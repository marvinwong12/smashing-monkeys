import os
import uuid
import json
import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

import chromadb
from chromadb.utils import embedding_functions
from matplotlib.font_manager import FontProperties  
from mplsoccer import Radar

# --- LangChain Core & Community Imports ---
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Internal Project Imports ---
from src import ScoutEngine 
from src.helper_functions import normalize_name
from dotenv import load_dotenv

load_dotenv()

# Instantiate the global internal engine reference
_engine = ScoutEngine()


from mplsoccer import Radar
from matplotlib.font_manager import FontProperties

# Use standard system sans-serif fonts natively supported by matplotlib
font_normal = FontProperties(family="sans-serif", size=11)
font_bold = FontProperties(family="sans-serif", size=16, weight="bold")


# ==========================================
# 1. VECTOR DATA ACCESS LAYER
# ==========================================
class NarrativeRepository:
    """
    Singleton repository for vector database operations. 
    Initializes the Chroma client and heavy embedding models exactly once.
    """
    def __init__(self):
        # Dynamically calculate the project root to ensure safe pathing
        current_file_path = os.path.abspath(__file__)
        src_dir = os.path.dirname(os.path.dirname(current_file_path)) 
        project_root = os.path.dirname(src_dir)
        
        # Point to the dedicated vector storage layer
        self.db_path = os.path.join(project_root, "data/vector_store")
        os.makedirs(self.db_path, exist_ok=True)
        
        # 1. Initialize persistent client once
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 2. Load embedding model into memory once
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        # 3. Connect to (or create) the collection
        self.collection = self.client.get_or_create_collection(
            name="player_narratives", 
            embedding_function=self.embedding_fn
        )

    def get_narrative(self, clean_name: str) -> Optional[str]:
        """Performs a fast O(1) vector metadata lookup."""
        results = self.collection.get(where={"player_match_name": clean_name})
        if results and results.get('documents') and len(results['documents']) > 0:
            return results['documents'][0]
        return None

    def save_narrative(self, clean_name: str, original_name: str, dossier: str) -> None:
        """Caches a newly generated narrative into the vector store."""
        self.collection.add(
            documents=[dossier],
            metadatas=[{"player_match_name": clean_name, "original_name": original_name}],
            ids=[str(uuid.uuid4())]
        )

# Initialize the global thread-safe repository singleton
narrative_repo = NarrativeRepository()


# ==========================================
# 2. PRODUCTION REFINED LLM TOOLS
# ==========================================
@tool
def search_player_tactical_tool(player_name: str) -> str:
    """
    Searches the database for a specific football player's structural tactical profile and metrics.
    Use this when you need detailed stats about a single named individual.
    """
    try:
        # Delegate entirely to the ScoutEngine
        player_profile = _engine.lookup_player(player_name)
        
        if isinstance(player_profile, str) and player_profile.startswith("No scout target found"):
            return f"Strategic Lookup Notification: No profile found for '{player_name}' inside the scouting matrices."
            
        return json.dumps([player_profile], ensure_ascii=False)
        
    except Exception as e:
        return f"System Execution Error processing player profile query: {str(e)}"


@tool
def discovery_scout_tool(
    *,
    position: Optional[str] = None,
    league: Optional[str] = None,
    nation: Optional[str] = None,
    team: Optional[str] = None,
    min_height: Optional[int] = None,
    max_height: Optional[int] = None,
    min_weight: Optional[int] = None,
    max_weight: Optional[int] = None,
    min_weak_foot: Optional[int] = None,
    min_skill_moves: Optional[int] = None,
    preferred_foot: Optional[str] = None,
    max_value_millions: Optional[float] = None, 
    max_wage: Optional[float] = None,  
    max_age: Optional[int] = None,
    target_metric: Optional[str] = None, 
    min_metric_value: Optional[float] = None,
    sort_by_metric: str = "xg_per90",
    highest_first: bool = True
) -> str:
    """
    Scans the database to dynamically discover and rank hidden talent profiles matching financial, 
    positional, geographic, biometric, and professional performance limits.
    """
    try:
        # 1. Map all arguments dynamically into a dictionary
        # We let the ScoutEngine handle ALL column normalization and _per90 mapping.
        query_filters = {
            "position": position,
            "league": league,
            "nation": nation,
            "team": team,
            "min_height": min_height,
            "max_height": max_height,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "min_weak_foot": min_weak_foot,
            "min_skill_moves": min_skill_moves,
            "preferred_foot": preferred_foot,
            "max_value_mln": max_value_millions,
            "max_wage": max_wage,
            "max_age": max_age,
        }
        
        # Dynamically append the requested target metric if provided
        if target_metric and min_metric_value is not None:
            query_filters[f"min_{target_metric}"] = min_metric_value

        # 2. Delegate directly to the Engine
        matches = _engine.discover_players(
            filters=query_filters, 
            sort_by=sort_by_metric, 
            ascending=(not highest_first), 
            limit=5
        )
        
        if not matches or (isinstance(matches, str) and "Error" in matches):
            return (
                "CRITICAL: Zero players matched your exact tactical or financial configuration filters. "
                "Report back to the user that zero database rows satisfied these boundaries."
            )
        
        # Return JSON. ScoutEngine already replaced NaNs with Nones, so it parses cleanly.
        return json.dumps(matches, ensure_ascii=False)
        
    except Exception as e:
        return f"System Execution Error filtering discovery matrices: {str(e)}"

@tool
def generate_percentile_comparison_chart(
    player1_name: str,
    player2_name: Optional[str] = None,
    # Define metric groups suitable for different roles
    metric_group: str = "attacking" # Default to common attacking comparison
) -> str:
    """
    Generates a professional percentile-based radar/spider chart comparing one player 
    or two players side-by-side. Returns the image as a base64 encoded PNG string.

    Args:
        player1_name: First player to visualize (required).
        player2_name: Second player for side-by-side comparison (optional).
        metric_group: 'attacking', 'defending', or 'comprehensive' profile.
    """
    try:
        # 1. Define the metric sets (assumes engine pre-calculates percentiles as _pctl)
        # 1. Define the metric sets using EXACT raw column names from your CSV
        attacking_metrics = [
            'goals_per90', 'xg_per90', 'shots_per90', 
            'assists_per90', 'key_passes_per90', 'np_goals_per90', 
            'np_xg_per90', 'xg_chain_per90'
        ]
        defending_metrics = [
            'performance_tklw_per90', 'performance_int_per90', 'ball_recoveries_per90', 
            'performance_fls_per90', 'performance_crdy_per90', 'xg_buildup_per90'
        ]
        comprehensive_metrics = attacking_metrics + defending_metrics

        if metric_group.lower() == "defending":
            active_metrics = defending_metrics
        elif metric_group.lower() == "comprehensive":
            active_metrics = comprehensive_metrics
        else:
            active_metrics = attacking_metrics # Default
            
        # 2. Delegate data lookup and dynamic percentile calculation to the Engine
        pctl_data1 = _engine.get_player_percentiles(player1_name, active_metrics)
        if not pctl_data1:
            return f"Strategic Notification: Data lookup failure for primary player '{player1_name}'."
            
        pctl_values1 = list(pctl_data1.values())
        param_labels = [label.title().replace('_', ' ') for label in pctl_data1.keys()]

        # 3. Handle comparison player data optionally
        pctl_values2 = None
        player2_label = "Average Player"
        if player2_name:
            pctl_data2 = _engine.get_player_percentiles(player2_name, active_metrics)
            if pctl_data2:
                pctl_values2 = list(pctl_data2.values())
                player2_label = player2_name.upper()

        # 4. Generate the Chart via mplsoccer (isolated plotting logic)
        num_vars = len(active_metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # The radar is circular, so close the loop by appending the start value
        pctl_values1 += pctl_values1[:1]
        if pctl_values2:
            pctl_values2 += pctl_values2[:1]
        angles += angles[:1]
        param_labels += param_labels[:1] # Close the label loop for the spider look

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2) # Start at the top center
        ax.set_theta_direction(-1) # Clockwise
        
        # Configure the percentile grid rings (0-100)
        ax.set_rgrids([25, 50, 75, 90, 99], labels=["25", "50", "75", "90", "99"], 
                     color="grey", size=10, fontproperties=font_normal)
        ax.set_ylim(0, 100)
        
        # Configure and rotate metric labels
        ax.set_xticks(angles[:-1])
        # Calculate label rotations dynamically for dynamic metric counts
        label_angles = [float(angle * 180 / np.pi) for angle in angles[:-1]]
        ax.set_xticklabels(param_labels[:-1], color='black', size=12, 
                          fontproperties=font_normal)
        for label, angle in zip(ax.get_xticklabels(), label_angles):
            label.set_rotation(angle - 180)

        # Plot Player 1 (Attacking colors)
        ax.plot(angles, pctl_values1, color='#1A78CF', linewidth=2, linestyle='solid')
        ax.fill(angles, pctl_values1, color='#1A78CF', alpha=0.25)
        
        # Plot Player 2 if provided (Defending colors)
        if pctl_values2:
            ax.plot(angles, pctl_values2, color='#CF4C1A', linewidth=2, linestyle='solid')
            ax.fill(angles, pctl_values2, color='#CF4C1A', alpha=0.15)
        else:
            # Fallback: Plot average line explicitly at 50%
            ax.plot(angles, [50] * len(angles), color='#CCCCCC', linewidth=1, linestyle='dashed')

        # Formatting titles, legends, and styling
        comparison_text = f" vs. {player2_label}" if pctl_values2 else " - Solo Profile"
        title_text = f"{player1_name.upper()}{comparison_text}\n{metric_group.title()} Percentile Comparison"
        
        fig.text(0.5, 0.95, title_text, ha='center', color='black', fontproperties=font_bold)
        
        if pctl_values2:
            # Side-by-side Legend
            fig.text(0.35, 0.05, player1_name.upper(), color='#1A78CF', fontproperties=font_bold)
            fig.text(0.5, 0.05, "vs.", color='black', fontproperties=font_normal)
            fig.text(0.55, 0.05, player2_label.upper(), color='#CF4C1A', fontproperties=font_bold)
        else:
            # Solo Profile disclaimer
            fig.text(0.5, 0.05, "Dashed line represents performance at the 50th percentile.", 
                     ha='center', color='#888888', fontproperties=font_normal)
            
        fig.text(0.95, 0.02, "[Source: SQLite Master Database]", ha='right', color='#888888', fontproperties=font_normal)
        
        # 5. Output Management: Convert the matplotlib plot directly to an in-memory Base64 string
        buffer = io.BytesIO()
        
        # Save the figure directly into the RAM buffer instead of the hard drive
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
        plt.close(fig)
        
        # Rewind the buffer to the beginning and encode it as a text string
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Return the base64 string inside the JSON payload
        return json.dumps({
            "status": "success",
            "image_base64": image_base64
        }, ensure_ascii=False)
        
    except Exception as e:
        return f"System Execution Error: {str(e)}"


# ==========================================
# 4. BEHAVIORAL ANALYST TOOLS (The RAG Component)
# ==========================================
@tool
def query_player_narrative_tool(player_name: str) -> str:
    """
    Queries the local vector database for qualitative scouting profiles concerning 
    a player's character, leadership, temperament, and background.
    """
    if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" in os.environ:
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
        
    clean_name = normalize_name(player_name)
    
    # 1. Query the unified repository layer
    cached_dossier = narrative_repo.get_narrative(clean_name)
    
    if cached_dossier:
        return f"--- LOCAL DOSSIER FOR {player_name.upper()} ---\n\n{cached_dossier}"
        
    # 2. Cache MISS: Trigger autonomous search
    try:
        search = DuckDuckGoSearchResults(max_results=4)
        raw_web_data = search.run(f"{player_name} football character personality attitude injury")
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        
        prompt = PromptTemplate(
            input_variables=["player_name", "web_context"],
            template="""
            You are a senior executive director of football intelligence. 
            Synthesize the provided raw internet search results into a concise 4-sentence 
            qualitative intelligence profile for {player_name}.
            
            Structure the dossier precisely like this:
            - Sentence 1: General career/tactical context and current status.
            - Sentence 2: Deep dive into character, leadership traits, or dressing room presence.
            - Sentence 3: Known injury risks, temperament flags, or physical robustness.
            - Sentence 4: Summary recommendation on their psychological profile.
            
            RAW WEB DATA CONTEXT:
            {web_context}
            """
        )
        
        chain = prompt | llm
        generated_dossier = chain.invoke({"player_name": player_name, "web_context": raw_web_data}).content
        
        # Cache it to the vector store
        narrative_repo.save_narrative(clean_name, player_name, generated_dossier)
        
        return f"--- NEWLY RECONSTRUCTED DOSSIER FOR {player_name.upper()} (Now Cached) ---\n\n{generated_dossier}"
        
    except Exception as e:
        import traceback
        print(f"\n🚨 TOOL CRASH DETECTED IN NARRATIVE SEARCH 🚨")
        print(f"Error Type: {type(e)}")
        print(f"Details: {str(e)}")
        traceback.print_exc()
        return f"System Error: Failed to execute tool due to {str(e)}"

# Export for LangGraph Supervisor integration
SCOUT_TOOLS = [
    search_player_tactical_tool,
    discovery_scout_tool,
    query_player_narrative_tool,
    generate_percentile_comparison_chart
]