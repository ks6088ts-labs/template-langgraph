import logging

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from template_langgraph.agents.chat_with_tools_agent.agent import AgentState
from template_langgraph.agents.chat_with_tools_agent.agent import graph as chat_with_tools_agent
from template_langgraph.loggers import get_logger

router = APIRouter()
logger = get_logger(
    name=__name__,
    verbosity=logging.DEBUG,
)


class RunChatWithToolsAgentRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    question: str


class RunChatWithToolsAgentResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    response: str


@router.post(
    "/chat_with_tools_agent/",
    response_model=RunChatWithToolsAgentResponse,
    operation_id="create_transcription_job",
)
async def run_chat_with_tools_agent(
    request: RunChatWithToolsAgentRequest,
) -> RunChatWithToolsAgentResponse:
    try:
        async for event in chat_with_tools_agent.astream(
            input=AgentState(
                messages=[
                    {
                        "role": "user",
                        "content": request.question,
                    },
                ],
            ),
            config={
                "recursion_limit": 30,
            },
        ):
            logger.debug(f"Event received: {event}")
        response = event["chat_with_tools"]["messages"][0].content
        return RunChatWithToolsAgentResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing event: {e}")
        return RunChatWithToolsAgentResponse(
            response=f"An error occurred while processing your request with {e}",
        )
