FROM python:3.11

WORKDIR /workspace

COPY pyproject.toml poetry.lock ./
RUN pip install poetry

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN poetry install --no-root --no-directory
