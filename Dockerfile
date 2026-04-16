FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base r-base-dev libssl-dev libcurl4-openssl-dev libxml2-dev dos2unix \
    && rm -rf /var/lib/apt/lists/*

RUN Rscript -e "install.packages('irace', repos='https://cloud.r-project.org/')"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY validator/ validator/
COPY input/ input/
COPY app.py .
COPY irace_runner.py .
COPY target-runner.sh .
COPY parameters.txt .
COPY scenario.txt .
COPY scenario-test.txt .
COPY instances.txt .
COPY instances-test.txt .

RUN dos2unix target-runner.sh \
    && chmod +x irace_runner.py target-runner.sh

RUN mkdir -p output irace_output

CMD ["Rscript", "-e", "library(irace); irace(scenario=readScenario('scenario.txt'))"]
