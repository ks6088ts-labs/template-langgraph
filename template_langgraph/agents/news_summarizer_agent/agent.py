import httpx
from langgraph.graph import StateGraph
from langgraph.types import Send

from template_langgraph.agents.news_summarizer_agent.models import (
    AgentState,
    Article,
    StructuredArticle,
    SummarizeWebContentState,
)
from template_langgraph.internals.notifiers import get_notifier
from template_langgraph.internals.scrapers import get_scraper
from template_langgraph.internals.summarizers import get_summarizer
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class NewsSummarizerAgent:
    def __init__(
        self,
        llm=AzureOpenAiWrapper().chat_model,
        notifier=get_notifier(),
        scraper=get_scraper(),
        summarizer=get_summarizer(),
    ):
        self.llm = llm
        self.notifier = notifier
        self.scraper = scraper
        self.summarizer = summarizer

    def create_graph(self):
        """Create the main graph for the agent."""
        # Create the workflow state graph
        workflow = StateGraph(AgentState)

        # Create nodes
        workflow.add_node("initialize", self.initialize)
        workflow.add_node("summarize_web_content", self.summarize_web_content)
        workflow.add_node("notify", self.notify)

        # Create edges
        workflow.set_entry_point("initialize")
        workflow.add_conditional_edges(
            source="initialize",
            path=self.run_subtasks,
            path_map={
                "summarize_web_content": "summarize_web_content",
            },
        )
        workflow.add_edge("summarize_web_content", "notify")
        workflow.set_finish_point("notify")
        return workflow.compile(
            name=NewsSummarizerAgent.__name__,
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
                node="summarize_web_content",
                arg=SummarizeWebContentState(
                    url=state.input.urls[idx],
                    prompt=state.input.prompt,
                ),
            )
            for idx, _ in enumerate(state.input.urls)
        ]

    def summarize_web_content(self, state: SummarizeWebContentState):
        is_valid_url = state.url.startswith("http")
        is_valid_content = False
        content = ""

        # Check if the URL is valid
        if not is_valid_url:
            logger.error(f"Invalid URL: {state.url}")
            is_valid_content = False
        else:
            # Scrape the web content
            try:
                logger.info(f"Scraping URL: {state.url}")
                content = self.scraper.scrape(state.url)
                is_valid_content = True
            except httpx.RequestError as e:
                logger.error(f"Error fetching web content: {e}")

        if is_valid_content:
            logger.info(f"Summarizing content with LLM: {state.url}")
            structured_article: StructuredArticle = self.summarizer.summarize(
                prompt=state.prompt,
                content=content,
            )
            return {
                "articles": [
                    Article(
                        is_valid_url=is_valid_url,
                        is_valid_content=is_valid_content,
                        content=content,
                        url=state.url,
                        structured_article=structured_article,
                    ),
                ]
            }

    def notify(self, state: AgentState) -> AgentState:
        """Send notifications to the user."""
        logger.info(f"Sending notifications with state: {state}")
        # Simulate sending notifications
        # convert list of articles to a dictionary for notification
        summary = {}
        for i, article in enumerate(state.articles):
            summary[i] = article.model_dump()
        self.notifier.notify(
            text=summary.__str__(),
        )
        return state


graph = NewsSummarizerAgent(
    notifier=get_notifier(),
    scraper=get_scraper(),
    summarizer=get_summarizer(),
).create_graph()
