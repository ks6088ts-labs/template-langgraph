import sys

from template_langgraph.agents.chat_with_tools_agent.agent import ChatWithToolsAgent

if __name__ == "__main__":
    png_path = "data/chat_with_tools_agent.png"
    if len(sys.argv) > 1:
        png_path = sys.argv[1]

    with open(png_path, "wb") as f:
        f.write(ChatWithToolsAgent().draw_mermaid_png())
