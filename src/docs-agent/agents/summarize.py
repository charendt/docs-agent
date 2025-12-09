from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIResponsesClient

def create_summarize_agent(chat_client: AzureOpenAIResponsesClient) -> ChatAgent:
    return ChatAgent(
        chat_client=chat_client,
        instructions=(
            'You are a helpful assistant that summarizes the provided text '
            'into a concise bulleted list. Do not lose important information. '
            'Always output your response in Markdown format.'
        ),
        name='summarize-agent',
    )
