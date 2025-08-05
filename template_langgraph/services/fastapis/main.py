from fastapi import FastAPI

from template_langgraph.services.fastapis.routers import agents as agents_router

app = FastAPI()


app.include_router(
    agents_router.router,
    prefix="/agents",
    tags=["agents"],
)
