from agent_framework import Workflow, WorkflowBuilder, ChatAgent

def create_documentation_workflow(rewrite_agent: ChatAgent, docs_agent: ChatAgent, summarize_agent: ChatAgent) -> Workflow:
    workflow = WorkflowBuilder()
    workflow.add_agent(summarize_agent, output_response=True)
    workflow.add_edge(rewrite_agent, docs_agent)
    workflow.add_edge(docs_agent, summarize_agent)
    workflow.set_start_executor(rewrite_agent)
    return workflow.build()
