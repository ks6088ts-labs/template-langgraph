from functools import lru_cache

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    otel_service_name: str = "<OTEL_SERVICE_NAME>"
    otel_collector_endpoint: str = "<OTEL_COLLECTOR_ENDPOINT>"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_otel_settings() -> Settings:
    """Get OpenTelemetry settings."""
    return Settings()


class OtelWrapper:
    def __init__(
        self,
        settings: Settings = None,
    ):
        if settings is None:
            settings = get_otel_settings()
        self.settings = settings

    def initialize(self):
        provider = TracerProvider(
            resource=Resource(
                attributes={
                    "service.name": self.settings.otel_service_name,
                }
            )
        )
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.settings.otel_collector_endpoint,
        )
        provider.add_span_processor(
            span_processor=BatchSpanProcessor(otlp_exporter),
        )
        trace.set_tracer_provider(provider)

    def get_tracer(self, name: str):
        return trace.get_tracer(name)
