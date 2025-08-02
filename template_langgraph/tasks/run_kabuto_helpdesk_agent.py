import logging
import sys

from template_langgraph.agents.kabuto_helpdesk_agent import KabutoHelpdeskAgent
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    question = "「鬼灯」を実行すると、KABUTOが急に停止します。原因と対策を教えてください。"
    if len(sys.argv) > 1:
        # sys.argv[1] が最初の引数
        question = sys.argv[1]

    logger.info(f"質問: {question}")

    agent = KabutoHelpdeskAgent(
        tools=None,  # ツールはカスタムせず、デフォルトのツールを使用
    )
    response = agent.run(
        question=question,
    )
    logger.info(f"Agent result: {response}")

    # エージェントの応答を表示
    logger.info(f"Answer: {response['messages'][-1].content}")
