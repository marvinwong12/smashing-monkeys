import streamlit as st
from src.agents.supervisor import build_chief_scout
import json
import os
import re
from PIL import Image
import base64

# 1. Page Configuration
st.set_page_config(
    page_title="Chief Scout OS",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ Chief Scout OS")
st.markdown("Interact with your AI scouting department.")

# 2. Initialize the LangGraph Agent (Run once per session)
if "scout_agent" not in st.session_state:
    st.session_state.scout_agent = build_chief_scout()
    
# 3. Initialize Chat History in Streamlit Session State (Keep this as you had it)
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "I am the Chief Scout. Who are we looking for today?"
    })

# 4. Render the existing chat history on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # First render any conversational text
        if message.get("content"):
            st.markdown(message["content"])
        
        # Decode and redraw the image directly from the base64 string stored in memory!
        if "image_base64" in message and message["image_base64"]:
            try:
                img_bytes = base64.b64decode(message["image_base64"])
                st.image(img_bytes, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render historical chart: {e}")

# 5. Capture user input
user_query = st.chat_input("Ask your scouting agent a question...")

if user_query:
    # A. Display the user's message immediately
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # B. Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # C. Display a loading spinner while the agent thinks
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        inputs = {"messages": [("user", user_query)]}
        config = {"configurable": {"thread_id": "streamlit_session_1"}}
        
        with st.spinner("Scouting database and generating visualization..."):
            final_state = st.session_state.scout_agent.invoke(inputs, config=config)
            
            raw_content = final_state["messages"][-1].content
            
            if isinstance(raw_content, list):
                if len(raw_content) > 0 and isinstance(raw_content[0], dict) and "text" in raw_content[0]:
                    response_payload = raw_content[0]["text"]
                else:
                    response_payload = str(raw_content)
            else:
                response_payload = raw_content

            print("\n--- NEW MESSAGE CHAIN ---")
            for m in final_state["messages"]:
                print(f"TYPE: {getattr(m, 'type', 'unknown')} | CONTENT: {str(m.content)[:100]}...")
            print("-------------------------\n")

            image_base64_data = None
            clean_text = response_payload  # AI's final text summary

            # Scan the conversation history backwards to find the tool output
            for msg in reversed(final_state["messages"]):
                
                # Stop looking once we hit the user's prompt for this turn ---
                if getattr(msg, "type", "") == "human":
                    break 

                if getattr(msg, "type", "") == "tool":
                    # Convert content to string safely just to check if our key is in it
                    content_str = str(msg.content)
                    
                    if "image_base64" in content_str:
                        try:
                            # 1. If LangGraph already parsed it into a dictionary automatically
                            if isinstance(msg.content, dict):
                                image_base64_data = msg.content.get("image_base64")
                            
                            # 2. If it's a raw JSON string
                            elif isinstance(msg.content, str):
                                parsed_json = json.loads(msg.content)
                                image_base64_data = parsed_json.get("image_base64")
                                
                            if image_base64_data:
                                break  # We successfully extracted the image, stop searching!
                                
                        except (json.JSONDecodeError, TypeError) as e:
                            print(f"Extraction failed on this message: {e}")
                            pass

        # D. Display and save the response
        if image_base64_data:
            try:
                # Add base64 padding (==) just in case the string length got slightly clipped
                padded_base64 = image_base64_data + "=" * ((4 - len(image_base64_data) % 4) % 4)
                img_bytes = base64.b64decode(padded_base64)
                
                # Render the AI's conversational text FIRST
                if clean_text:
                    message_placeholder.markdown(clean_text)
                else:
                    message_placeholder.empty()
                    
                # Render the pixel data natively BELOW the text
                st.image(img_bytes, use_container_width=True)
                
                # Save BOTH to Streamlit's history so it stays when you refresh
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": clean_text,
                    "image_base64": padded_base64 
                })
            except Exception as e:
                message_placeholder.error(f"Error decoding the chart visualization: {str(e)}")
                
        else:
            # Fall back to standard markdown text if absolutely no image was found
            message_placeholder.markdown(clean_text)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": clean_text
            })