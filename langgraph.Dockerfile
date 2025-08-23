FROM langchain/langgraph-api:3.11



# -- Adding local package . --
ADD . /deps/template-langgraph
# -- End of local package . --

# -- Installing all local dependencies --
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e /deps/*
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{"chat_with_tools_agent": "template_langgraph.agents.chat_with_tools_agent.agent:graph", "demo_agents_parallel_rag_agent": "template_langgraph.agents.demo_agents.parallel_rag_agent.agent:graph", "demo_agents_multi_agent": "template_langgraph.agents.demo_agents.multi_agent:graph", "demo_agents_research_deep_agent": "template_langgraph.agents.demo_agents.research_deep_agent:graph", "demo_agents_weather_agent": "template_langgraph.agents.demo_agents.weather_agent:graph", "image_classifier_agent": "template_langgraph.agents.image_classifier_agent.agent:graph", "issue_formatter_agent": "template_langgraph.agents.issue_formatter_agent.agent:graph", "kabuto_helpdesk_agent": "template_langgraph.agents.kabuto_helpdesk_agent.agent:graph", "news_summarizer_agent": "template_langgraph.agents.news_summarizer_agent.agent:graph", "supervisor_agent": "template_langgraph.agents.supervisor_agent.agent:graph", "task_decomposer_agent": "template_langgraph.agents.task_decomposer_agent.agent:graph"}'



# -- Ensure user deps didn't inadvertently overwrite langgraph-api
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --
# -- Removing build deps from the final image ~<:===~~~ --
RUN pip uninstall -y pip setuptools wheel
RUN rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* && find /usr/local/bin -name "pip*" -delete || true
RUN rm -rf /usr/lib/python*/site-packages/pip* /usr/lib/python*/site-packages/setuptools* /usr/lib/python*/site-packages/wheel* && find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel && rm /usr/bin/uv /usr/bin/uvx

WORKDIR /deps/template-langgraph
