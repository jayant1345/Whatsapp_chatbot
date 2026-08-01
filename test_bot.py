import os
import sys
from dotenv import load_dotenv

# Load env variables from local directory
load_dotenv()

# Verify that API keys are set before running the interactive test
AI_PROVIDER = os.getenv("AI_PROVIDER", "anthropic").lower()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

if AI_PROVIDER == "anthropic" and (not ANTHROPIC_KEY or ANTHROPIC_KEY.startswith("sk-ant-xx")):
    print("⚠️ WARNING: ANTHROPIC_API_KEY is not set or is using the default template in your .env file!")
    print("Please configure your keys in the '.env' file to run actual AI tests.")
    print("Alternatively, set AI_PROVIDER='openrouter' and configure OPENROUTER_API_KEY.")
    sys.exit(1)

if AI_PROVIDER == "openrouter" and (not OPENROUTER_KEY or OPENROUTER_KEY.startswith("sk-or-xx")):
    print("⚠️ WARNING: OPENROUTER_API_KEY is not set or is using the default template in your .env file!")
    print("Please configure your keys in the '.env' file to run actual AI tests.")
    sys.exit(1)

print("🚀 Starting LangGraph local chatbot test...")
print(f"Using Provider: {AI_PROVIDER.upper()}")
print(f"Model Selected: {os.getenv('SELECTED_MODEL')}\n")

# Import LangChain / LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Annotated, TypedDict

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: AgentState):
    messages = state["messages"]
    user_query = messages[-1].content
    
    # Retrieve context from pgvector PostgreSQL database
    try:
        from rag_helper import query_knowledge_base
        context = query_knowledge_base(user_query)
    except Exception:
        context = ""
        
    system_content = "You are a professional customer service assistant for JK Data Lab. Keep your answers concise, accurate, and under 3 sentences."
    if context:
        system_content += f"\n\nUse the following verified context from the company database to answer the query:\n{context}\nIf the answer is not found in this context, politely say that you don't know."
        
    system_prompt = SystemMessage(content=system_content)
    full_messages = [system_prompt] + messages
    
    if AI_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=os.getenv("SELECTED_MODEL"),
            temperature=0.5,
            anthropic_api_key=ANTHROPIC_KEY
        )
    elif AI_PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=os.getenv("SELECTED_MODEL"),
            temperature=0.5,
            openai_api_key=OPENROUTER_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"Invalid AI_PROVIDER: {AI_PROVIDER}")
        
    response = llm.invoke(full_messages)
    return {"messages": [response]}

# Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# Use SqliteSaver for session persistence
import sqlite3
test_db_conn = sqlite3.connect("test_checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(test_db_conn)
checkpointer.setup()

chatbot_graph = workflow.compile(checkpointer=checkpointer)

# Run interactive terminal loop
thread_id = "local_test_user_001"
config = {"configurable": {"thread_id": thread_id}}

print("="*60)
print(f"Chatbot initialized. Thread ID set to: '{thread_id}'")
print("Memory is stored in 'test_checkpoints.sqlite'.")
print("Type 'exit' or 'quit' to stop.")
print("="*60)

while True:
    try:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Exiting test session. Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        # Send user message to the graph
        inputs = {"messages": [HumanMessage(content=user_input)]}
        output_state = chatbot_graph.invoke(inputs, config=config)
        
        # Get response
        bot_response = output_state["messages"][-1].content
        print(f"Bot: {bot_response}")
        
    except KeyboardInterrupt:
        print("\nExiting test session. Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")
