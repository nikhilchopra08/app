import streamlit as st
from phi.agent import Agent
from phi.model.huggingface import HuggingFaceChat
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_agent(role, instructions):
    """Create an agent with error handling"""
    try:
        return Agent(
            role=role,
            model=HuggingFaceChat(
                id="meta-llama/Meta-Llama-3-8B-Instruct",
                max_tokens=4096,
            ),
            tools=[DuckDuckGo()] if role == "research" else [],
            instructions=instructions,
            markdown=True
        )
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        return None

def safe_run_agent(agent, input_text):
    """Safely run agent with error handling"""
    if agent is None:
        return "Error: Agent not properly initialized"
    
    try:
        response = agent.run(input_text)
        return response.content if hasattr(response, 'content') else str(response)
    except AttributeError as e:
        # Handle the specific model_dump error
        if "model_dump" in str(e):
            try:
                # Alternative approach to get response
                response = agent.model.chat(input_text)
                return str(response)
            except Exception as inner_e:
                logger.error(f"Alternative approach failed: {str(inner_e)}")
                return f"Error processing request: {str(inner_e)}"
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}\n{traceback.format_exc()}")
        return f"Error: {str(e)}"

def main():
    st.title("Research & YouTube Script Generator")
    
    # Initialize agents with error handling
    if 'agents_initialized' not in st.session_state:
        st.session_state.research_agent = create_agent(
            role="research",
            instructions=["Always include sources and give answer to the point."]
        )
        st.session_state.script_agent = create_agent(
            role="script",
            instructions=["Always start with a question to increase curiosity and at end of script ask a question to listeners"]
        )
        st.session_state.agents_initialized = True

    # Input section
    st.header("Research Topic")
    query = st.text_input("Enter your research topic:", 
                         placeholder="E.g., why govt. should not print more money")
    
    if st.button("Generate Research & Script"):
        if query:
            try:
                # Research phase
                with st.spinner("Researching the topic..."):
                    research_results = safe_run_agent(st.session_state.research_agent, query)
                    
                    if not research_results.startswith("Error"):
                        st.header("Research Results")
                        st.markdown(research_results)
                        
                        # Script generation phase
                        with st.spinner("Generating YouTube script..."):
                            script_content = safe_run_agent(st.session_state.script_agent, research_results)
                            
                            if not script_content.startswith("Error"):
                                st.header("YouTube Script")
                                st.markdown(script_content)
                                
                                # Download buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button(
                                        label="Download Research",
                                        data=research_results,
                                        file_name="research.md",
                                        mime="text/markdown"
                                    )
                                with col2:
                                    st.download_button(
                                        label="Download Script",
                                        data=script_content,
                                        file_name="youtube_script.md",
                                        mime="text/markdown"
                                    )
                            else:
                                st.error(script_content)
                    else:
                        st.error(research_results)
                        
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        else:
            st.warning("Please enter a research topic")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This app helps you:
        1. Research topics using AI and web search
        2. Generate YouTube scripts based on the research
        3. Download both research and scripts in markdown format
        """)
        
        st.header("Instructions")
        st.write("""
        1. Enter your research topic in the text input
        2. Click 'Generate Research & Script'
        3. Wait for both the research and script to be generated
        4. Download the results using the buttons below the content
        """)
        
        # Add status indicator
        st.header("System Status")
        if st.session_state.get('agents_initialized'):
            if st.session_state.research_agent and st.session_state.script_agent:
                st.success("All systems operational")
            else:
                st.error("Agent initialization failed")
        else:
            st.warning("System initializing...")

if __name__ == "__main__":
    main()