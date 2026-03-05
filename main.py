import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# --- LangGraph Imports ---
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

app = FastAPI()

# --- 1. VECTOR DB SETUP ---
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ["HF_TOKEN"]
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# --- 2. THE TOOL ---
@tool
def get_isl_video(word: str) -> str:
    """Use this tool ONLY when the user asks how to sign a specific word or phrase in ISL."""
    docs = vectorstore.similarity_search(word, k=1)
    if docs:
        url = docs[0].metadata.get("video_url")
        if url:
            return f"VIDEO_FOUND: {url}"
    return "VIDEO_NOT_FOUND"

tools = [get_isl_video]

# --- 3. THE LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

# --- 4. LANGGRAPH STATE & NODES ---
# State strictly holds our message history
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot_node(state: State):
    system_prompt = SystemMessage(content="""You are an accessible, helpful, and direct AI assistant designed for a mobile app used by mute and deaf individuals whose primary language is Indian Sign Language (ISL).

Follow these strict rules for every response:
1. Forgive Grammar and Syntax: Focus entirely on their intent.
2. Plain and Direct Language: Use simple, everyday vocabulary. Keep sentences short.
3. Highly Visual Formatting: Break up your responses using standard dashes (-) instead of asterisks. Avoid long paragraphs.
4. Professional Tone: Be empathetic and helpful. Focus strictly on delivering the information.
5. If user is talking in Hindi reply in Hindi.
6. If you use the get_isl_video tool and it returns a URL, simply say "Here is the video demonstrating how to sign [word]." DO NOT put the raw URL in your text response.""")
    
    # Prepend the system prompt to the message history before sending to Gemini
    messages_to_send = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages_to_send)
    
    # LangGraph will automatically append this to the state's message list
    return {"messages": [response]}

# --- 5. COMPILE THE GRAPH ---
graph_builder = StateGraph(State)

# Add our custom chatbot node and the prebuilt ToolNode
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", ToolNode(tools=tools))

# Define the flow
graph_builder.add_edge(START, "chatbot")
# tools_condition automatically checks if Gemini decided to call a tool
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot") # Loop back to Gemini after the tool finishes

# Add short-term memory
memory = MemorySaver()
agent_graph = graph_builder.compile(checkpointer=memory)

# --- 6. THE ENDPOINT ---
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # thread_id is how LangGraph tracks separate users
    config = {"configurable": {"thread_id": "default_test_session"}}
    
    # Run the graph
    response = await agent_graph.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config
    )
    
    final_message_content = response["messages"][-1].content
    
    # 2. Safely extract the text, handling Gemini's weird list-of-blocks format
    if isinstance(final_message_content, list):
        text_parts = []
        for block in final_message_content:
            # If it's a dictionary block, grab the 'text' key
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            # Fallback in case it's just a string in a list
            elif isinstance(block, str):
                text_parts.append(block)
        raw_text = " ".join(text_parts)
    else:
        # If LangChain behaves and returns a normal string
        raw_text = str(final_message_content)
        
    # 3. Clean the text
    clean_text = raw_text.replace('*', '')
    
    # 4. Search the history explicitly for the exact ToolMessage
    video_url = None
    for msg in response["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "get_isl_video":
            if "VIDEO_FOUND:" in msg.content:
                video_url = msg.content.replace("VIDEO_FOUND: ", "").strip()
    
    return {
        "reply": clean_text,
        "video_url": video_url
    }