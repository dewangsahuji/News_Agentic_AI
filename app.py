import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.asknews import AskNewsSearch

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
import re

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv(dotenv_path=r"C:\Users\dewan\Coding\GenAIKN\Langchain\.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["ASKNEWS_CLIENT_SECRET"] = os.getenv("ASKNEWS_CLIENT_ID")
os.environ["ASKNEWS_API_KEY"] = os.getenv("ASKNEWS_API_KEY")

# -------------------------------
# Initialize the model
# -------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# -------------------------------
# Define Tools
# -------------------------------
search_tool = DuckDuckGoSearchResults(num_results=15)

@tool
def search(query: str) -> str:
    """Search for information."""
    try:
        results = search_tool.invoke(query)
        if not results:
            return "No search results found. Try rephrasing your query."
        return str(results)
    except Exception as e:
        return f"Error fetching search results: {e}"
    
news = AskNewsSearch(max_results=2)
@tool
def news (query:str,hours_back:int = 24)->str:
    """For Daily News"""
    results = news.invoke({"query":query,"hours_back":hours_back})
    return results



# -------------------------------
# Error Handling Middleware
# -------------------------------
@wrap_tool_call
def handle_tool_error(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(content=str(e), tool_call_id=request.tool_call["id"])

# -------------------------------
# Create Agent
# -------------------------------
tools = [search,news]
agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[handle_tool_error],
    system_prompt="You are a AI that summarizes each one of the news in few lines with details,time and sources "
)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI News Summarizer", page_icon="🦾", layout="centered")

st.title("🦾 AI News Summarizer for Accessibility")
st.markdown("This app uses **LangChain**, **OpenAI**, and **DuckDuckGo** to summarize the latest news for visually impaired users.")

# -------------------------------
# User Input + State
# -------------------------------
user_input = st.text_area("💬 Ask about news:", placeholder="e.g., Tell me about today's cybersecurity news")

if st.button("Run Agent", use_container_width=True):
    if user_input.strip():
        with st.spinner("🤖 Summarizing the latest news..."):
            try:
                # Invoke the agent with user message
                response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
                
                # Safely access the latest message content (last one from the list)
                messages = response.get("messages", [])
                if not messages:
                    st.error("No messages returned from agent.")
                else:
                    # Extract the final AI response
                    final_msg = messages[-1]  # last message (AIMessage)
                    
                    # If it's an object, get .content, else treat as dict or string
                    if hasattr(final_msg, "content"):
                        content = final_msg.content
                    elif isinstance(final_msg, dict) and "content" in final_msg:
                        content = final_msg["content"]
                    else:
                        content = str(final_msg)

                    # Show clean summary
                    st.session_state.latest_output = content
                    st.success("✅ Latest Summary:")
                    st.markdown(content)
                    # st.write(response)

            except Exception as e:
                st.session_state.latest_output = f"❌ Error: {e}"
                st.error(st.session_state.latest_output)
    else:
        st.warning("Please enter a query first.")

# -------------------------------
# Show Only Latest Output
# -------------------------------
# if st.session_state.latest_output:
#     st.success("✅ Latest Summary:")
#     # st.markdown(st.session_state.latest_output)

st.markdown("---")
st.caption("Powered by LangChain • OpenAI • DuckDuckGo Search")
