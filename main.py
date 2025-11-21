import os
import asyncio

from agent_framework_devui import serve
from dotenv import load_dotenv

from agent_framework import MCPStreamableHTTPTool, Workflow, ChatAgent, WorkflowOutputEvent
from agent_framework.azure import AzureOpenAIResponsesClient

from agents.rewrite import create_rewrite_agent
from agents.docs import create_docs_agent
from agents.summarize import create_summarize_agent
from workflows.documentation import create_documentation_workflow

from azure.ai.agentserver.agentframework import from_agent_framework

MODE = os.getenv("MODE", "cli")  # "devui", "cli", or "hosted_agent"

async def create_workflow() -> tuple[Workflow, ChatAgent, ChatAgent, ChatAgent, MCPStreamableHTTPTool]:
    load_dotenv(override=True)

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME")

    if not endpoint or not api_key:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env")

    chat_client = AzureOpenAIResponsesClient(
        endpoint=endpoint,
        deployment_name=deployment,
        api_key=api_key,
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
