# Agents in Microsoft Agent Workflow published as Hosted Agents

This project demonstrates a workflow agent that responds to product launch strategy inquiries by concurrently leveraging insights from three specialized agents:
- **Rewrite Agent**: Refines the user query.
- **Docs Agent**: Retrieves relevant documentation.
- **Summarize Agent**: Synthesizes the information into a coherent response.

## Prerequisites

- [Azure Developer CLI (azd)](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd) installed.
- An Azure subscription.
- Access to Azure AI Foundry.

## Setup and Deployment

Follow these steps to initialize and deploy the agent workflow:

1.  **Initialize the project template:**
    ```bash
    azd init -t https://github.com/Azure-Samples/azd-ai-starter-basic
    ```

2.  **Initialize the AI Agent:**
    Connect to your existing Azure AI Foundry project. Replace the resource ID with your specific project details.
    ```bash
    azd ai agent init --project-id /subscriptions/02ba4597-4783-4faf-aaa3-00814c82b7b9/resourceGroups/rg-foundry-nextgen-ncus/providers/Microsoft.CognitiveServices/accounts/proj-hosted-agents-resource/projects/proj-hosted-agents
    ```

3.  **Configure the Agent Definition:**
    Initialize the agent configuration using the provided `agent.yaml`.
    ```bash
    azd ai agent init -m https://github.com/charendt/docs-agent/blob/master/agent.yaml
    azd up
    ```

4.  **Deploy:**
    Deploy the agent workflow to Azure.
    ```bash
    azd deploy agents-in-workflow
    ```
    *Note: If you haven't provisioned the infrastructure yet, you can use `azd up` to provision and deploy in one step.*

    Sample output:
    ```
    Deploying services (azd deploy)

    (✓) Done: Deploying service agents-in-workflow
    - Agent playground (portal): https://ai.azure.com/nextgen/r/ArpFl0eDT6-qowCBTIK3uQ,rg-foundry-nextgen-ncus,,proj-hosted-agents-resource,proj-hosted-agents/build/agents/agents-in-workflow/build?version=4 
    - Agent endpoint: https://proj-hosted-agents-resource.services.ai.azure.com/api/projects/proj-hosted-agents/agents/agents-in-workflow/versions/4 

    ```

## Architecture

The workflow is defined in `src/workflows/documentation.py` and orchestrated using the Microsoft Agent Framework. It runs as a Hosted Agent on Foundry.