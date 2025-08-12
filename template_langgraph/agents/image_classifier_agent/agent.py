import os
from base64 import b64encode

import httpx
from langgraph.graph import StateGraph
from langgraph.types import Send

from template_langgraph.agents.image_classifier_agent.models import (
    AgentState,
    ClassifyImageState,
    Result,
    Results,
)
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


def load_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return b64encode(image_file.read()).decode("utf-8")


class MockNotifier:
    def notify(self, id: str, body: dict) -> None:
        """Simulate sending a notification to the user."""
        logger.info(f"Notification sent for request {id}: {body}")


class MockClassifier:
    def predict(
        self,
        prompt: str,
        image: str,
        llm=AzureOpenAiWrapper().chat_model,
    ) -> Result:
        """Simulate image classification."""
        return Result(
            title="Mocked Image Title",
            summary=f"Mocked summary of the prompt: {prompt}",
            labels=["mocked_label_1", "mocked_label_2"],
            reliability=0.95,
        )


class LlmClassifier:
    def predict(
        self,
        prompt: str,
        image: str,
        llm=AzureOpenAiWrapper().chat_model,
    ) -> Result:
        """Use the LLM to classify the image."""
        logger.info(f"Classifying image with LLM: {prompt}")
        return llm.with_structured_output(Result).invoke(
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image",
                            "source_type": "base64",
                            "data": image,
                            "mime_type": "image/png",
                        },
                    ],
                },
            ]
        )


class ImageClassifierAgent:
    def __init__(
        self,
        llm=AzureOpenAiWrapper().chat_model,
        notifier=MockNotifier(),
        classifier=MockClassifier(),
    ):
        self.llm = llm
        self.notifier = notifier
        self.classifier = classifier

    def create_graph(self):
        """Create the main graph for the agent."""
        # Create the workflow state graph
        workflow = StateGraph(AgentState)

        # Create nodes
        workflow.add_node("initialize", self.initialize)
        workflow.add_node("classify_image", self.classify_image)
        workflow.add_node("notify", self.notify)

        # Create edges
        workflow.set_entry_point("initialize")
        workflow.add_conditional_edges(
            source="initialize",
            path=self.run_subtasks,
            path_map={
                "classify_image": "classify_image",
            },
        )
        workflow.add_edge("classify_image", "notify")
        workflow.set_finish_point("notify")
        return workflow.compile(
            name=ImageClassifierAgent.__name__,
        )

    def initialize(self, state: AgentState) -> AgentState:
        """Initialize the agent state."""
        logger.info(f"Initializing state: {state}")
        # FIXME: retrieve urls from user request
        return state

    def run_subtasks(self, state: AgentState) -> list[Send]:
        """Run the subtasks for the agent."""
        logger.info(f"Running subtasks with state: {state}")
        return [
            Send(
                node="classify_image",
                arg=ClassifyImageState(
                    prompt=state.input.prompt,
                    file_path=state.input.file_paths[idx],
                ),
            )
            for idx, _ in enumerate(state.input.file_paths)
        ]

    def classify_image(self, state: ClassifyImageState):
        logger.info(f"Classify file: {state.file_path}")
        if state.file_path.endswith((".png", ".jpg", ".jpeg")) and os.path.isfile(state.file_path):
            try:
                logger.info(f"Loading file: {state.file_path}")
                base64_image = load_image_to_base64(state.file_path)

                logger.info(f"Classifying file: {state.file_path}")
                result = self.classifier.predict(
                    prompt=state.prompt,
                    image=base64_image,
                    llm=self.llm,
                )

                logger.info(f"Classification result: {result.model_dump_json(indent=2)}")
                return {
                    "results": [
                        Results(
                            file_path=state.file_path,
                            result=result,
                        ),
                    ]
                }
            except httpx.RequestError as e:
                logger.error(f"Error fetching web content: {e}")

    def notify(self, state: AgentState) -> AgentState:
        """Send notifications to the user."""
        logger.info(f"Sending notifications with state: {state}")
        # Simulate sending notifications
        summary = {}
        for i, result in enumerate(state.results):
            summary[i] = result.model_dump()
        self.notifier.notify(
            id=state.input.id,
            body=summary,
        )
        return state


# For testing
# graph = ImageClassifierAgent().create_graph()

graph = ImageClassifierAgent(
    llm=AzureOpenAiWrapper().chat_model,
    notifier=MockNotifier(),
    classifier=LlmClassifier(),
).create_graph()
