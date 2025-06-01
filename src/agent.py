from rag import rag
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def create_rag_tool():
    """Create a RAG tool for the agent to use."""
    rag_instance = rag()
    retriever = rag_instance.get_retriever()
    
    def search_documents(query: str) -> str:
        """Search through documents using RAG."""
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant information found."
        return "\n\n".join([doc.page_content for doc in docs])
    
    return Tool(
        name="document_search",
        description="Useful for searching through documents to find information about teams, timelines and dependencies.",
        func=search_documents
    )

def create_agent():
    """Create an agent with RAG capabilities."""
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )
    
    # Create the RAG tool
    tools = [create_rag_tool()]
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant that helps analyze and organization, their project timelines and dependencies.
        Use the document_search tool to find relevant information.
        When analyzing timelines, look for:
        - Conflicting or overlapping schedules
        - Timeline inconsistencies
        - Critical path issues
        
        When analyzing dependencies, look for:
        - Inter-team dependencies
        - Potential blockers

        When identifying teams and team members, look for:
        - Who leads the team
        - Who is in the team
        - What is the team's name
        
        Always provide clear, structured responses with specific examples from the documents. Focus on required actions and be very succinct"""),
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