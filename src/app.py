import streamlit as st
from agent import create_agent
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
    st.subheader("TPM")
    with st.spinner("Analyzing timelines..."):
        # Get timeline analysis from agent
        timeline_response = st.session_state.agent.invoke({
            "input": "Are there any conflicting timelines in the project? Are there any dependencies that are not aligned? Keep your answer short and concise. Focus on actions required, and which teams are involved for each action."
        })
        st.markdown(timeline_response["output"])