FROM rocker/r-ver:4.3.2

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libsodium-dev libpng-dev libglpk-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY python/requirements.txt python/requirements.txt
RUN python3 -m venv /opt/venv/d606-nlp && \
    /opt/venv/d606-nlp/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/d606-nlp/bin/pip install --no-cache-dir -r python/requirements.txt

ENV RETICULATE_PYTHON=/opt/venv/d606-nlp/bin/python

COPY install_packages.R install_packages.R
RUN Rscript install_packages.R

COPY . .

EXPOSE 3838
CMD ["R", "-e", "shiny::runApp('/app', host='0.0.0.0', port=3838)"]