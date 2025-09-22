import dspy

from template_langgraph.llms.azure_openais import Settings


def get_lm(settings: Settings = None) -> dspy.LM:
    if settings is None:
        settings = Settings()

    return dspy.LM(
        model=f"azure/{settings.azure_openai_model_chat}",
        api_key=settings.azure_openai_api_key,
        api_base=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
        max_tokens=1000,
    )
