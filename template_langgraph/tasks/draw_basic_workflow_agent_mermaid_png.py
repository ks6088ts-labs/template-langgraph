import sys

from template_langgraph.agents.basic_workflow_agent.agent import BasicWorkflowAgent

if __name__ == "__main__":
    png_path = "data/basic_workflow_agent.png"
    if len(sys.argv) > 1:
        png_path = sys.argv[1]

    with open(png_path, "wb") as f:
        f.write(BasicWorkflowAgent().draw_mermaid_png())
