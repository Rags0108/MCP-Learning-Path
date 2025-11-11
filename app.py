import streamlit as st
from utils import run_agent_sync
from langchain_core.messages import AIMessage

st.set_page_config(page_title="MCP POC", page_icon="🤖", layout="wide")
st.title("Model Context Protocol(MCP) - Learning Path Generator")

# Initialize session state for progress
if 'current_step' not in st.session_state:
    st.session_state.current_step = ""
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'last_section' not in st.session_state:
    st.session_state.last_section = ""
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False

# Sidebar for API and URL configuration
st.sidebar.header("Configuration")
st.sidebar.subheader("Pipedream URLs")
youtube_pipedream_url = st.sidebar.text_input("YouTube URL (Required)", placeholder="Enter your Pipedream YouTube URL")

# Secondary tool selection
secondary_tool = st.sidebar.radio("Select Secondary Tool:", ["Drive", "Notion"])

# Secondary tool URL input
if secondary_tool == "Drive":
    drive_pipedream_url = st.sidebar.text_input("Drive URL", placeholder="Enter your Pipedream Drive URL")
    notion_pipedream_url = None
else:
    notion_pipedream_url = st.sidebar.text_input("Notion URL", placeholder="Enter your Pipedream Notion URL")
    drive_pipedream_url = None

# Quick guide before goal input
st.info("""
**Quick Guide:**
1. Enter your YouTube URL (required)
2. Select and configure your secondary tool (Drive or Notion)
3. Enter a clear learning goal, for example:
4. I used Local LLM Model in this learning path generator
   - "I want to learn python basics in 3 days"
   - "I want to learn data science basics in 10 days"
""")

# Main content area
st.header("Enter Your Goal")
user_goal = st.text_input("Enter your learning goal:", help="Describe what you want to learn, and we'll generate a structured path using YouTube content and your selected tool.")

# Progress area
progress_container = st.container()
progress_bar = st.empty()

def update_progress(message: str):
    """Update progress in the Streamlit UI"""
    st.session_state.current_step = message
    # Determine section and update progress (simplified logic)
    if "Setting up agent" in message:
        section = "Setup"
        st.session_state.progress = 0.1
    elif "Added" in message:
        section = "Integration"
        st.session_state.progress = 0.2
    elif "Creating AI agent" in message:
        section = "Setup"
        st.session_state.progress = 0.3
    elif "Generating" in message:
        section = "Generation"
        st.session_state.progress = 0.5
    elif "complete" in message:
        section = "Complete"
        st.session_state.progress = 1.0
        st.session_state.is_generating = False
    else:
        section = st.session_state.last_section or "Progress"
    st.session_state.last_section = section

    progress_bar.progress(st.session_state.progress)
    with progress_container:
        if section != st.session_state.last_section and section != "Complete":
            st.write(f"**{section}**")
        prefix = "✓" if st.session_state.progress >= 0.5 else "→"
        if "complete" in message:
            st.success("All steps completed! 🎉")
        else:
            st.write(f"{prefix} {message}")
    st.session_state.last_section = section

# Generate Learning Path button
if st.button("Generate Learning Path", type="primary", disabled=st.session_state.is_generating):
    if not youtube_pipedream_url:
        st.error("YouTube URL is required. Please enter your Pipedream YouTube URL in the sidebar.")
    elif (secondary_tool == "Drive" and not drive_pipedream_url) or (secondary_tool == "Notion" and not notion_pipedream_url):
        st.error(f"Please enter your Pipedream {secondary_tool} URL in the sidebar.")
    elif not user_goal:
        st.warning("Please enter your learning goal.")
    else:
        try:
            st.session_state.is_generating = True
            st.session_state.current_step = ""
            st.session_state.progress = 0
            st.session_state.last_section = ""
            result = run_agent_sync(
                youtube_pipedream_url=youtube_pipedream_url,
                drive_pipedream_url=drive_pipedream_url,
                notion_pipedream_url=notion_pipedream_url,
                user_goal=user_goal,
                progress_callback=update_progress
            )
            st.header("Your Learning Path")

            if result:    
                # Handle LangChain message-style output
                if isinstance(result, dict) and "messages" in result:
                    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
                    
                    if ai_messages:
                        content = ai_messages[-1].content  # Use last AIMessage
                    
                    else:
                        content = str(result)

                elif isinstance(result, list) and any(isinstance(m, AIMessage) for m in result):
                    ai_messages = [m for m in result if isinstance(m, AIMessage)]
                    content = ai_messages[-1].content if ai_messages else str(result)                
                else:
                    content = str(result)

                    # Extract only the final answer (ignore <think> parts if any)
                if "Here is your comprehensive" in content:
                    content = content.split("Here is your comprehensive", 1)[1]
                    content = "Here is your comprehensive" + content

                st.markdown(f"📚 **Your Learning Path**", unsafe_allow_html=True)
                st.markdown(content, unsafe_allow_html=True)
            else:
                st.error("No results were generated. Please try again.")

                st.session_state.is_generating = False
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your URLs and try again.")
            st.session_state.is_generating = False
            
