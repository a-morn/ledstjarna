import streamlit as st
from agent import create_agent, create_rag_tool
import os
import torch

torch.classes.__path__ = []

# Set page config
st.set_page_config(
    page_title="Ledstjärna",
    page_icon="✯",
    layout="wide"
)

# Initialize session state for agent
if 'agent' not in st.session_state:
    st.session_state.agent = create_agent()

# Title and description
st.title("✯ Ledstjärna")
st.markdown("""
A leading light in the dark.
""")

# Create two columns for the main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Timeline Analysis")
    with st.spinner("Analyzing timelines..."):
        # Get timeline analysis from agent
        timeline_response = st.session_state.agent.invoke({
            "input": "Are there any conflicting timelines in the project? Please analyze and explain any conflicts you find."
        })
        st.markdown(timeline_response["output"])

with col2:
    st.subheader("Team Dependency Analysis")
    with st.spinner("Analyzing team dependencies..."):
        # Enhanced prompt: first list teams/members, then analyze dependencies
        team_dep_prompt = (
            "Find out if there is a team that needs to be notified of an ongoing project of another team. "
            "Present the results in a clear, structured format. Focus on who needs to be notified of what."
        )
        dependency_response = st.session_state.agent.invoke({
            "input": team_dep_prompt
        })
        st.markdown(dependency_response["output"])