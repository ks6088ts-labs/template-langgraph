FROM python:3.13-slim-bookworm AS requirements-stage

WORKDIR /tmp

RUN pip install --no-cache-dir uv==0.7.12

COPY ./pyproject.toml ./uv.lock /tmp/

# Delete `-e .` line from requirements.txt
RUN uv export --format requirements-txt --no-dev --no-hashes --output-file requirements.txt && \
    sed -i '/^-e .*/d' requirements.txt

FROM python:3.13-slim-bookworm

ARG GIT_REVISION="0000000"
ARG GIT_TAG="x.x.x"

WORKDIR /app

# Copy requirements first for better cache efficiency
COPY --from=requirements-stage /tmp/requirements.txt /app/requirements.txt

# Install dependencies in a separate layer for caching
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy application code after dependencies are installed
COPY . .

CMD ["python", "template_python/core.py"]
