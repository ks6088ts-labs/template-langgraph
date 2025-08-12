from pydantic import BaseModel, Field


class StructuredArticle(BaseModel):
    title: str = Field(..., description="Title of the article")
    date: str = Field(..., description="Publication date of the article")
    summary: str = Field(..., description="Summary of the article")
    keywords: list[str] = Field(..., description="Keywords extracted from the article")
    score: int = Field(..., description="Score of the article based on user request from 0 to 100")


class Article(BaseModel):
    is_valid_url: bool = Field(..., description="Indicates if the article URL is valid")
    is_valid_content: bool = Field(..., description="Indicates if the article content is valid")
    content: str = Field(..., description="Original content of the article")
    url: str = Field(..., description="URL of the article")
    structured_article: StructuredArticle = Field(..., description="Structured representation of the article")


class AgentInputState(BaseModel):
    request: str = Field(..., description="Request from the user")
    request_id: str = Field(..., description="Unique identifier for the request")
    urls: list[str] = Field(..., description="List of article URLs")


class AgentOutputState(BaseModel):
    articles: list[Article] = Field(..., description="List of articles processed by the agent")


class AgentState(BaseModel):
    input: AgentInputState = Field(..., description="Input state for the agent")
    output: AgentOutputState = Field(..., description="Output state for the agent")
    target_url_index: int | None = Field(..., description="Index of the target URL being processed")
