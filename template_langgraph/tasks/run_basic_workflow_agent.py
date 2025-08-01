import logging
import sys

from template_langgraph.agents.basic_workflow_agent.agent import AgentInput, BasicWorkflowAgent
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    question = "「鬼灯」を実行すると、KABUTOが急に停止します。原因と対策を教えてください。"
    if len(sys.argv) > 1:
        # sys.argv[1] が最初の引数
        question = sys.argv[1]

    # Agentのインスタンス化
    agent = BasicWorkflowAgent()

    # AgentInputの作成
    agent_input = AgentInput(
        request=question,
    )

    # エージェントの実行
    logger.info(f"Running BasicWorkflowAgent with input: {agent_input.model_dump_json(indent=2)}")
    agent_output = agent.run_agent(input=agent_input)
    logger.info(f"Agent output: {agent_output.model_dump_json(indent=2)}")
