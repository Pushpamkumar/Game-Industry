"""
MCP Client Agent for UdaPlay.
Connects over stdio to mcp_server.py, lists exposed research tools, and
orchestrates conversational Q&A using OpenAI-compatible function calling.
"""

import os
import sys
import json
import asyncio
import importlib.util
from typing import List, Dict, Any, Optional

# SQLite compatibility shim for environments requiring pysqlite3
if importlib.util.find_spec("pysqlite3") is not None:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

import chromadb
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import nest_asyncio

nest_asyncio.apply()
load_dotenv()

console = Console()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")


class UdaPlayMCPClient:
    """
    Intelligent research client that connects to the UdaPlay MCP server
    and executes multi-turn tool-calling workflows.
    """

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.session: Optional[ClientSession] = None
        self.openai_client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        self.available_tools: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, Any]] = []

    def _build_system_prompt(self) -> str:
        db_count = "N/A"
        try:
            client = chromadb.PersistentClient(path="chromadb")
            if "udaplay" in [c.name for c in client.list_collections()]:
                db_count = str(client.get_collection("udaplay").count())
        except Exception:
            pass

        return (
            "You are UdaPlay, an expert AI Research Assistant specialized in the video game industry. "
            "Your objective is to provide comprehensive, factual, and well-structured answers to user queries.\n\n"
            "Operating Protocol:\n"
            "1. Primary Retrieval: Always query the local ChromaDB vector database first using 'retrieve_game'.\n"
            "2. Evidence Evaluation: Always invoke 'evaluate_retrieval' on retrieved documents to verify whether they "
            "sufficiently answer the inquiry without ambiguity.\n"
            "3. Secondary Fallback: If evaluation indicates the information is insufficient or missing, perform a "
            "web search using 'web_search'.\n"
            "4. Synthesis & Citation: Synthesize the final answer clearly with bullet points, stating the release year, "
            "platform, publisher, and specific source attribution (including URLs for web findings).\n"
            f"Note: Local database currently indexes {db_count} curated video game records."
        )

    async def connect_and_start(self):
        """Connect to the MCP server via stdio transport and launch interactive chat."""
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["mcp_server.py"],
            env=os.environ.copy()
        )

        console.print("[cyan]Establishing stdio connection to UdaPlay MCP Server...[/cyan]")
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                self.session = session
                await session.initialize()

                tools_resp = await session.list_tools()
                exposed_tools = tools_resp.tools
                tool_names = [t.name for t in exposed_tools]
                console.print(f"[bold green]Connected successfully![/bold green] Registered tools: {tool_names}")

                # Format tools into OpenAI function calling schema
                self.available_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    }
                    for tool in exposed_tools
                ]

                # Initialize conversation history with system instructions
                self.messages = [{"role": "system", "content": self._build_system_prompt()}]
                await self.interactive_chat_loop()

    async def process_user_query(self, user_query: str):
        """Execute a full tool-calling cycle for a given user query."""
        self.messages.append({"role": "user", "content": user_query})
        console.print(f"\n[bold green]User Query:[/bold green] {user_query}")

        turn_active = True
        max_iterations = 8
        iteration = 0

        while turn_active and iteration < max_iterations:
            iteration += 1
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                tools=self.available_tools,
                tool_choice="auto",
                temperature=0.0
            )

            choice = response.choices[0]
            message = choice.message
            tool_calls = message.tool_calls

            if tool_calls:
                # Append assistant tool call message
                self.messages.append(message)

                for call in tool_calls:
                    func_name = call.function.name
                    raw_args = call.function.arguments
                    try:
                        parsed_args = json.loads(raw_args)
                    except Exception:
                        parsed_args = {}

                    console.print(
                        f"[yellow]>> Dispatching MCP Tool:[/yellow] [bold]{func_name}[/bold] | Args: {parsed_args}"
                    )

                    try:
                        tool_res = await self.session.call_tool(func_name, arguments=parsed_args)
                        content_str = (
                            tool_res.content[0].text
                            if tool_res.content and hasattr(tool_res.content[0], "text")
                            else str(tool_res.content)
                        )
                    except Exception as err:
                        content_str = json.dumps({"error": str(err)})

                    console.print(f"[dim]   Result length: {len(content_str)} chars[/dim]")

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": content_str
                    })
            else:
                # LLM produced final response
                turn_active = False
                final_content = message.content or ""
                self.messages.append({"role": "assistant", "content": final_content})
                console.print(Panel(Markdown(final_content), title="[bold green]UdaPlay Final Answer[/bold green]", expand=False))

    async def interactive_chat_loop(self):
        """CLI interactive prompt loop."""
        console.print("\n[bold]UdaPlay Research Assistant Ready[/bold]. Type 'exit' or 'quit' to quit.\n")
        while True:
            try:
                user_input = input("UdaPlay > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    console.print("[dim]Exiting UdaPlay MCP session.[/dim]")
                    break
                await self.process_user_query(user_input)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    client = UdaPlayMCPClient()
    asyncio.run(client.connect_and_start())
