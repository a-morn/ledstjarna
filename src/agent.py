from rag import rag
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def create_rag_tools():
    """Create separate RAG tools for each data source."""
    rag_instance = rag()
    
    def search_slack_messages(query: str) -> str:
        """Search through Slack messages."""
        retriever = rag_instance.get_slack_retriever()
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant Slack messages found."
        return "\n\n".join([doc.page_content for doc in docs])
    
    def search_google_docs(query: str) -> str:
        """Search through Google Docs."""
        retriever = rag_instance.get_google_docs_retriever()
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant Google Docs found."
        return "\n\n".join([doc.page_content for doc in docs])
    
    def search_teams(query: str) -> str:
        """Search through team information."""
        retriever = rag_instance.get_teams_retriever()
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant team information found."
        return "\n\n".join([doc.page_content for doc in docs])
    
    def search_company_info(query: str) -> str:
        """Search through company information."""
        retriever = rag_instance.get_company_info_retriever()
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant company information found."
        return "\n\n".join([doc.page_content for doc in docs])
    
    return [
        Tool(
            name="slack_search",
            description="Includes ad-hoc discussions that may or may not relate to an ongoing project.",
            func=search_slack_messages
        ),
        Tool(
            name="google_docs_search",
            description="Includes project plans, specifications, and other documentation.",
            func=search_google_docs
        ),
        Tool(
            name="teams_search",
            description="Includes all teams and their members.",
            func=search_teams
        ),
        Tool(
            name="company_info_search",
            description="Basic information about the company.",
            func=search_company_info
        )
    ]

def create_agent():
    """Create an agent with RAG capabilities."""
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )
    
    # Create the RAG tools
    tools = create_rag_tools()
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior technical project manager.
        
        You are a technical project manager.
        Your task is to analyze communication and documentation across product teams to determine if further alignment is needed.

        Identify if one team's work depends on another team's work. Only consider teams available in the teams_search tool.

        Output in this format:

        - Source Team
        - Dependent Team
        - Dependency Description
        - Risk Level (None / Low / Medium / High)
        - Alignment Required (Yes / No)
        
        Only include the output if alignment is required and risk level is Low or above.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create the agent executor
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )

def analyze_project():
    """Run the analysis using the agent."""
    agent = create_agent()
    
    # Analyze timelines
    timeline_response = agent.invoke({
        "input": "Are there any conflicting timelines in the project? Please analyze and explain any conflicts you find."
    })
    print("\n=== Timeline Analysis ===")
    print(timeline_response["output"])
    
    # Analyze dependencies
    dependency_response = agent.invoke({
        "input": "Are there any dependencies that need to be flagged? Please identify and explain any critical dependencies or blockers."
    })
    print("\n=== Dependency Analysis ===")
    print(dependency_response["output"])

if __name__ == "__main__":
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    analyze_project() 