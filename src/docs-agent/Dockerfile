FROM python:3.13-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for building some python packages)
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODE=hosted_agent
ENV PYTHONUNBUFFERED=1
ENV AZURE_OPENAI_ENDPOINT=https://proj-hosted-agents-resource.openai.azure.com/openai/v1/
ENV AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4.1-mini
ENV AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME=gpt-4.1-mini
ENV ENABLE_HOSTED_AGENTS=true

CMD ["python", "./main.py"]
