"""
MCP Server implementation for UdaPlay Game Research Assistant.
Exposes semantic search, collection inspection, evaluation, and web search tools
over the Model Context Protocol (stdio transport).
"""

import os
import sys
import importlib.util
from datetime import datetime
from typing import List, Dict, Any, Optional, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# SQLite compatibility shim for environments requiring pysqlite3 (e.g. Linux/Udacity)
if importlib.util.find_spec("pysqlite3") is not None:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.api.types import QueryResult
from tavily import TavilyClient
from fastmcp import FastMCP

from lib.llm import LLM
from lib.vector_db import VectorStore
from lib.messages import UserMessage, SystemMessage

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize FastMCP server
mcp = FastMCP("udaplay-research-server")


class GameDocument(BaseModel):
    """Structured data model for video game records retrieved from vector store."""
    Platform: Annotated[str, Field(description="Console or system platform (e.g., PC, PlayStation 5, Switch)")]
    Name: Annotated[str, Field(description="Official title of the video game")]
    YearOfRelease: Annotated[int, Field(description="Calendar release year")]
    Description: Annotated[str, Field(description="Summary description and genre details")]


class RetrievalEvaluation(BaseModel):
    """Data model assessing if retrieved documents sufficiently answer the user query."""
    useful: Annotated[bool, Field(description="True if documents contain adequate factual information to answer")]
    description: Annotated[str, Field(description="Detailed rationale and justification for the assessment")]


@mcp.tool()
def retrieve_game(query: str, num_matches: int = 5) -> List[GameDocument]:
    """
    Perform semantic similarity search over the UdaPlay video games ChromaDB vector database.
    
    Args:
        query (str): The natural language query or game-related question.
        num_matches (int): Maximum number of top matching documents to return (default: 5).
        
    Returns:
        List[GameDocument]: Top matching game documents with metadata.
    """
    client = chromadb.PersistentClient(path="chromadb")
    collection = client.get_collection("udaplay")
    store = VectorStore(chroma_collection=collection)
    
    query_result: QueryResult = store.query(
        query_texts=[query],
        n_results=num_matches
    )
    
    matched_docs: List[GameDocument] = []
    if query_result and query_result.get("metadatas"):
        for meta in query_result["metadatas"][0]:
            matched_docs.append(GameDocument(**meta))
            
    return matched_docs


@mcp.tool()
def retrieve_all_games() -> List[GameDocument]:
    """
    Retrieve all cataloged game entries currently indexed in the local vector database.
    
    Returns:
        List[GameDocument]: Full catalog of indexed video games.
    """
    client = chromadb.PersistentClient(path="chromadb")
    collection = client.get_collection("udaplay")
    all_data = collection.get()
    
    catalog: List[GameDocument] = []
    if all_data and all_data.get("metadatas"):
        for meta in all_data["metadatas"]:
            catalog.append(GameDocument(**meta))
            
    return catalog


@mcp.tool()
def evaluate_retrieval(question: str, retrieved_docs: List[GameDocument]) -> RetrievalEvaluation:
    """
    Evaluate whether the retrieved local documents provide sufficient factual evidence
    to accurately and completely answer the user's specific query.
    
    Args:
        question (str): The original question asked by the user.
        retrieved_docs (List[GameDocument]): The candidate documents retrieved from vector storage.
        
    Returns:
        RetrievalEvaluation: Evaluation verdict (useful boolean flag and qualitative analysis).
    """
    evaluator_llm = LLM(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model="gpt-4o",
        temperature=0.0
    )
    
    system_prompt = (
        "You are an impartial, highly rigorous verification evaluator for a video game research agent.\n"
        "Assess whether the provided candidate documents contain sufficient, direct factual evidence to answer "
        "the user's inquiry accurately without speculating or hallucinating.\n"
        "Return useful=True only if the factual details needed to answer the question are present; otherwise useful=False."
    )
    
    user_prompt = (
        f"User Inquiry: {question}\n\n"
        f"Retrieved Documents Context:\n"
        f"{[doc.model_dump() if hasattr(doc, 'model_dump') else str(doc) for doc in retrieved_docs]}\n\n"
        "Provide your evaluation verdict and rationale."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        UserMessage(content=user_prompt)
    ]
    
    response = evaluator_llm.invoke(
        input=messages,
        response_format=RetrievalEvaluation
    )
    
    return response.content


@mcp.tool()
def web_search(query: str, search_depth: str = "advanced") -> Dict[str, Any]:
    """
    Search the live web using Tavily Search API when internal vector knowledge is insufficient.
    
    Args:
        query (str): The search query to look up on the web.
        search_depth (str): Tavily search depth mode ('basic' or 'advanced', default: 'advanced').
        
    Returns:
        Dict[str, Any]: Formatted search results with extracted answer, source URLs, and query metadata.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "answer": "TAVILY_API_KEY is not configured.",
            "results": [],
            "search_metadata": {"timestamp": datetime.now().isoformat(), "query": query, "error": "Missing API key"}
        }
        
    client = TavilyClient(api_key=api_key)
    raw_search = client.search(
        query=query,
        search_depth=search_depth,
        include_answer=True,
        include_raw_content=False,
        include_images=False
    )
    
    return {
        "answer": raw_search.get("answer", ""),
        "results": [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0)
            }
            for r in raw_search.get("results", [])
        ],
        "search_metadata": {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "total_results": len(raw_search.get("results", []))
        }
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
