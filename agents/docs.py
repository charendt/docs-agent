from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.azure import AzureOpenAIResponsesClient
from models.responses import DocsResponse

def create_docs_agent(chat_client: AzureOpenAIResponsesClient) -> tuple[ChatAgent, MCPStreamableHTTPTool]:
    mcp_tool = MCPStreamableHTTPTool(
        name="Microsoft Learn MCP",
        url="https://learn.microsoft.com/api/mcp",
        allowed_tools=["microsoft_docs_search"],
    )
    
    agent = ChatAgent(
        chat_client=chat_client,
        instructions=(
            'You are an expert assistant that answers questions about '
            'Microsoft products and services. Use the Microsoft Learn MCP '
            'server to ground your answers in official Microsoft '
            'documentation whenever possible.'
        ),
        name='microsoft-docs-agent',
        tools=mcp_tool,
        response_format=DocsResponse,
    )
    
    return agent, mcp_tool
