#FROM ghcr.io/open-webui/pipelines:cuda FIXME -> latest image returns error, had to move back few commits
FROM ghcr.io/open-webui/pipelines:git-adec657-cuda

USER root

COPY requirements.txt /app/pipelines

RUN pip install --no-cache-dir -r /app/pipelines/requirements.txt

# Only prod pipelines are copied automatically
COPY ./prod/*.py /app/pipelines
COPY ./libs /app/libs

EXPOSE 9099