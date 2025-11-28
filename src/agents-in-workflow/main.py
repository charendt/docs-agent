import asyncio
import os

from agent_framework import (
    ChatAgent,
    MCPStreamableHTTPTool,
    Workflow,
    WorkflowOutputEvent,
)
from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework_devui import serve
from agents.docs import create_docs_agent
from agents.rewrite import create_rewrite_agent
from agents.summarize import create_summarize_agent
from azure.ai.agentserver.agentframework import from_agent_framework
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from workflows.documentation import create_documentation_workflow

# Monkey patch Workflow.run_stream to accept checkpoint_storage
original_run_stream = Workflow.run_stream

async def patched_run_stream(self, message, **kwargs):
    # Ignore checkpoint_storage and other kwargs not supported by original
    async for event in original_run_stream(self, message):
        yield event

Workflow.run_stream = patched_run_stream

MODE = os.getenv("MODE", "devui")  # "devui", "cli", or "hosted_agent"

async def create_workflow() -> tuple[Workflow, ChatAgent, ChatAgent, ChatAgent, MCPStreamableHTTPTool]:
    load_dotenv(override=True)

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME")
    
    print(f"Loaded environment variables: ENDPOINT={endpoint}, DEPLOYMENT={deployment}, API_KEY={'***' if api_key else None}")

    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT must be set in .env")

    chat_client = AzureOpenAIResponsesClient(
        endpoint=endpoint,
        deployment_name=deployment,
        # api_key=api_key,
        credential=DefaultAzureCredential()
    )

    rewrite_agent = create_rewrite_agent(chat_client)
    docs_agent, mcp_tool = create_docs_agent(chat_client)
    summarize_agent = create_summarize_agent(chat_client)

    workflow = create_documentation_workflow(rewrite_agent, docs_agent, summarize_agent)

    return workflow, rewrite_agent, docs_agent, summarize_agent, mcp_tool


def main() -> None:
    workflow, rewrite_agent, docs_agent, summarize_agent, mcp_tool = asyncio.run(create_workflow())

    if MODE == "hosted_agent":
        print("Starting in Hosted Agent mode...")
        workflow_agent = workflow.as_agent()
        # Run the agent as a hosted agent
        from_agent_framework(workflow_agent).run()

    elif MODE == "cli":
        print("Starting in CLI mode...")
        print("Microsoft Docs Agent Workflow. Ask a question about Microsoft services.")
        print("Type 'exit' to quit.\n")

        async def repl() -> None:
            async with mcp_tool, rewrite_agent, docs_agent, summarize_agent:
                while True:
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in {"exit", "quit"}:
                        print("Goodbye!")
                        break

                    async for event in workflow.run_stream(user_input):
                        if isinstance(event, WorkflowOutputEvent):
                            print(f"Agent: {event.data}\n")
        
        asyncio.run(repl())

    elif MODE == "devui": # Default to devui
        print("Starting in Dev UI mode...")
        print("Microsoft Docs Agent Workflow. Dev UI will open in your browser.")
        serve(entities=[workflow, rewrite_agent, docs_agent, summarize_agent], auto_open=True)


if __name__ == "__main__":
    main()
