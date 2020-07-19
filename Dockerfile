FROM python:3.7.2-slim

ENV SERVICE=face-recognition

# ENV DD_AGENT_HOST=172.17.0.1
# ENV DD_TRACE_AGENT_PORT=8125

WORKDIR /app

COPY requirements.txt /app


# Template: https://github.com/puckel/docker-airflow/blob/master/Dockerfile
# python3-dev / build-essential / gcc are required for lightgbm
RUN set -ex \
    && buildDeps=' \
    python3-dev \
    build-essential \
    gcc \
    ' \
    && apt-get update -yqq \
    && apt-get install -yqq --no-install-recommends cmake $buildDeps \
    && apt-get install curl \
    && apt-get -y install libglib2.0-0 \
    && apt-get install -y libsm6 libxext6 libxrender-dev \
    && pip install --no-cache-dir --upgrade --upgrade-strategy=eager -r requirements.txt \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf \
    /var/lib/apt/lists/* \
    /tmp/* \
    /var/tmp/* \
    /usr/share/man \
    /usr/share/doc \
    /usr/share/doc-base

# EXPOSE 5000

# run the app
# ENTRYPOINT ["python", "entrypoint.py"]
# Make Jupyter Notebook Work...

RUN apt-get update && \
    apt-get install -yqq --no-install-recommends git && \
    pip install --upgrade -q black && \
    pip install -U altair vega_datasets jupyterlab && \
    pip install seaborn nb_black pyarrow

RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -&& \
    apt-get install -yqq --no-install-recommends nodejs && \
    jupyter labextension install @jupyterlab/toc && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
    jupyter labextension install jupyterlab-spreadsheet

RUN apt-get install libglib2.0-0
