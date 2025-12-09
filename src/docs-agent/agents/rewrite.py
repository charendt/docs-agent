from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIResponsesClient

def create_rewrite_agent(chat_client: AzureOpenAIResponsesClient) -> ChatAgent:
    return ChatAgent(
        chat_client=chat_client,
        instructions=(
            'You are an expert editor. Rewrite the user\'s input to be precise, '
            'correcting grammar and spelling. Optimize it for a documentation '
            'search query while preserving the original intent.'
        ),
        name='rewrite-agent',
    )
