"""
AI Supervisor Module
Defines the LangGraph ReAct agent (The Chief Scout) that reasons through 
user queries and interacts with the data tools.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from src.agents.tools import SCOUT_TOOLS
from langchain_core.messages import trim_messages
import json
import base64

# Load the API key from your hidden .env file
load_dotenv()

# ==========================================
# AGENT ORCHESTRATION LAYER
# ==========================================
def build_chief_scout():
    """
    Initializes the LangGraph ReAct agent with explicit directives
    for handling per-90 metrics, wages, ages, and custom tool sorting.
    """
    # Initialize the primary reasoning LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        max_retries=6, 
        temperature=0.8
    )

    system_prompt = (
        "You are the Chief Data Scout for a top-tier European football club. "
        "Your job is to discover and analyze talent using your unified analytics tools.\n\n"
        "FOOTBALL INTELLIGENCE & SEMANTIC MAPPING:\n"
        "Do not look for literal words. Use your advanced knowledge of football tactics and analytics "
        "to translate abstract user phrases into our specific database schema.\n"
        "• Map concepts like 'ball-progression' to xg_chain, 'ball-winner' to tackles/interceptions.\n"
        "• Map concepts like 'youngster', 'wonderkid', or 'prospect' to age thresholds.\n"
        "• Map concepts like 'cheap salary' or 'low earner' to annual_wage_eur.\n"
        "• Map physical constraints like 'aerial powerhouse', 'giant', or 'tall' to height_cm filters.\n"
        "• Map technical constraints like 'ambidextrous' or 'two-footed' to min_weak_foot filters.\n\n"
        "YOUR AVAILABLE SCHEMA MATRIX:\n"
        "When filtering or sorting, you MUST map your tactical, financial, or physical intent to these exact columns:\n"
        "• Attacking Output: 'goals_per90', 'xg_per90', 'shots_per90'\n"
        "• Playmaking & Progression: 'assists_per90', 'xa_per90', 'key_passes_per90', 'xg_chain_per90', 'xg_buildup_per90'\n"
        "• Defensive Steel & Discipline: 'performance_tklw_per90', 'performance_int_per90', 'ball_recoveries_per90', 'performance_fls_per90', 'performance_crdy_per90'\n"
        "• Biometrics & Technical Attributes: 'height_cm', 'weight_kg', 'preferred_foot', 'weak_foot', 'skill_moves'\n"
        "• Financial, Geographic & Environmental Constants: 'market_value_mln', 'contract_expiry', 'annual_wage_eur', 'age', 'nation', 'league', 'team'\n\n"
        "QUALITATIVE & RISK VETTING:\n"
        "After finding a shortlist of players using your structured data tools, you MUST use the "
        "'query_player_narrative_tool' to check their qualitative profiles. \n"
        "• If the user asks about injuries, character, or tactical fit, query the narrative tool for the specific player.\n"
        "• Synthesize both the raw numbers and the text reports into your final scouting brief.\n\n"
        "VISUALIZATION & PROFILING:\n"
        "When a user asks for a comparison chart, radar chart, visual profile, or to compare players side-by-side, "
        "you MUST ALWAYS call the 'generate_percentile_comparison_chart' tool. "
        "Intelligently select the appropriate `metric_group` argument ('attacking', 'defending', or 'comprehensive') based on the players' natural positions on the pitch.\n\n"
        "CRITICAL RULES:\n"
        "1. Never invent or hallucinate metrics, injury histories, or character traits. Rely STRICTLY on your tools.\n"
        "2. When the user sets a specific parameter threshold (e.g., maximum age, maximum wage, maximum transfer value, minimum height, or minimum weak foot ratings), you MUST explicitly pass those numeric values into the corresponding arguments of the discovery tool.\n"
        "3. For geographic or physical traits like 'league', 'preferred_foot', or 'height', map them explicitly to the tool's dedicated filtering parameters rather than treating them as loose text search values.\n"
        "4. Do not run generic searches dropping filters unless explicitly asked to broaden the range by the user.\n"
        "5. When asked for visual charts, radar charts, or percentile comparisons, DO NOT hallucinate stats or EA Sports/video game ratings. You MUST trigger the chart generation tool and wait for its output."
    )

    # In-memory checkpointer to persist conversation state/threads seamlessly
    checkpointer = MemorySaver()

    def state_trimmer_modifier(state) -> list:
        """Intercepts agent state, scrubs heavy base64 images, and trims old messages before calling Gemini."""
        
        # 1. First, scrub any base64 data out of the tool messages
        cleaned_messages = []
        for msg in state["messages"]:
            if getattr(msg, "type", "") == "tool":
                # Convert content to string safely just to check if our key is in it
                content_str = str(msg.content)
                
                if "image_base64" in content_str:
                    try:
                        # Create a lightweight version of the payload for the LLM
                        light_data = {
                            "status": "success",
                            "message": "Chart successfully generated and displayed to user. Summarize the comparison briefly.",
                            "image_base64": "[IMAGE_DATA_SCRUBBED_FOR_TOKEN_EFFICIENCY]"
                        }
                        
                        # Clone the message but with the tiny payload
                        # msg.__class__ ensures we keep the exact message type (ToolMessage)
                        clean_msg = msg.__class__(
                            content=json.dumps(light_data),
                            name=getattr(msg, "name", "generate_percentile_comparison_chart"),
                            tool_call_id=getattr(msg, "tool_call_id", "")
                        )
                        cleaned_messages.append(clean_msg)
                        continue  # Skip appending the original heavy message
                    except Exception as e:
                        print(f"Scrubber failed: {e}")
                        pass
            
            # Keep all other normal messages exactly as they are
            cleaned_messages.append(msg)

        # 2. Now apply the token trimmer to the newly cleaned list of messages
        trimmed = trim_messages(
            cleaned_messages,
            max_tokens=150000,          
            strategy="last",            
            token_counter=len,          
            start_on="human",           
            include_system=False,       
            allow_partial=True,
        )
        
        # 3. Prepend the system instructions
        return [("system", system_prompt)] + trimmed

    # Create the compilation graph engine
    app = create_react_agent(
        model=llm,
        tools=SCOUT_TOOLS,
        checkpointer=checkpointer,
        prompt=state_trimmer_modifier # Use state_modifier instead of prompt for standard LangGraph
    )
    
    return app

scout_app = build_chief_scout()

# ==========================================
# LOCAL TESTING LOOP
# ==========================================
if __name__ == "__main__":
    print("Initializing Stateful LangGraph Chief Scout...")
    
    print("Agent is online! Memory system activated. Type 'quit' to exit.\n")
    
    session_config = {"configurable": {"thread_id": "scout_terminal_session_alpha"}}
    
    while True:
        user_query = input("\nAsk the Chief Scout:\n> ")
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Shutting down scouting department...")
            break
            
        if not user_query.strip():
            continue
            
        print("\n--- CHIEF SCOUT EXECUTION TRACE ---")
        
        inputs = {"messages": [("user", user_query)]}
        
        for chunk in scout_app.stream(inputs, config=session_config, stream_mode="updates"):
            for node, data in chunk.items():
                if "messages" in data:
                    last_msg = data["messages"][-1]
                    
                    if node == "agent" and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tool in last_msg.tool_calls:
                            print(f"🤖 [Reasoning] Chief Scout using tool: {tool['name']}({tool['args']})")
                    
                    elif node == "tools":
                        print(f"⚙️ [Data Layer] Processing completed successfully. Returning state to LLM...")
                        
                    elif node == "agent":
                        print("\n--- CHIEF SCOUT REPORT ---")
                        content = last_msg.content
                        
                        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
                            clean_text = content[0].get('text', str(content))
                            print(clean_text)
                        else:
                            print(content)
                        print("--------------------------")