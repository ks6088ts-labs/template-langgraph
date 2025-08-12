import httpx
from langgraph.graph import StateGraph
from langgraph.types import Send

from template_langgraph.agents.news_summarizer_agent.models import AgentState, Article, StructuredArticle
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class MockNotifier:
    def notify(self, request_id: str, body: dict) -> None:
        """Simulate sending a notification to the user."""
        logger.info(f"Notification sent for request {request_id}: {body}")


class MockScraper:
    def scrape(self, url: str) -> str:
        """Simulate scraping a web page."""
        return "<html><body><h1>Mocked web content</h1></body></html>"


class HttpxScraper:
    def scrape(self, url: str) -> str:
        """Retrieve the HTML content of a web page."""
        with httpx.Client() as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text


class MockSummarizer:
    def summarize(
        self,
        prompt: str,
        content: str,
    ) -> StructuredArticle:
        """Simulate summarizing the input."""
        return StructuredArticle(
            title="Mocked Title",
            date="2023-01-01",
            summary=f"Mocked summary of the content: {content}, prompt: {prompt}",
            keywords=["mock", "summary"],
            score=75,
        )


class LlmSummarizer:
    def __init__(self, llm=AzureOpenAiWrapper().chat_model):
        self.llm = llm

    def summarize(
        self,
        prompt: str,
        content: str,
    ) -> StructuredArticle:
        """Use the LLM to summarize the input."""
        logger.info(f"Summarizing input with LLM: {prompt}")
        return self.llm.with_structured_output(StructuredArticle).invoke(
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ]
        )


class NewsSummarizerAgent:
    def __init__(
        self,
        llm=AzureOpenAiWrapper().chat_model,
        notifier=MockNotifier(),
        scraper=MockScraper(),
        summarizer=MockSummarizer(),
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
        workflow.add_node("fetch_web_content", self.fetch_web_content)
        workflow.add_node("notify", self.notify)

        # Create edges
        workflow.set_entry_point("initialize")
        workflow.add_conditional_edges(
            source="initialize",
            path=self.run_subtasks,
        )
        workflow.add_edge("fetch_web_content", "notify")
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
                node="fetch_web_content",
                arg=AgentState(
                    input=state.input,
                    output=state.output,
                    target_url_index=idx,
                ),
            )
            for idx, _ in enumerate(state.input.urls)
        ]

    def fetch_web_content(self, state: AgentState):
        url: str = state.input.urls[state.target_url_index]
        is_valid_url = url.startswith("http")
        is_valid_content = False
        content = ""

        # Check if the URL is valid
        if not is_valid_url:
            logger.error(f"Invalid URL: {url}")
            is_valid_content = False
        else:
            # Scrape the web content
            try:
                logger.info(f"Scraping URL: {url}")
                content = self.scraper.scrape(url)
                is_valid_content = True
            except httpx.RequestError as e:
                logger.error(f"Error fetching web content: {e}")

        if is_valid_content:
            logger.info(f"Summarizing content with LLM @ {state.target_url_index}: {url}")
            structured_article: StructuredArticle = self.summarizer.summarize(
                prompt=state.input.request,
                content=content,
            )
            state.output.articles.append(
                Article(
                    is_valid_url=is_valid_url,
                    is_valid_content=is_valid_content,
                    content=content,
                    url=url,
                    structured_article=structured_article,
                ),
            )

    def notify(self, state: AgentState) -> AgentState:
        """Send notifications to the user."""
        logger.info(f"Sending notifications with state: {state}")
        # Simulate sending notifications
        # convert list of articles to a dictionary for notification
        summary = {}
        for i, article in enumerate(state.output.articles):
            summary[i] = article.model_dump()
        self.notifier.notify(
            request_id=state.input.request_id,
            body=summary,
        )
        return state


# For testing
# graph = NewsSummarizerAgent().create_graph()

graph = NewsSummarizerAgent(
    notifier=MockNotifier(),
    scraper=HttpxScraper(),
    summarizer=LlmSummarizer(),
).create_graph()
