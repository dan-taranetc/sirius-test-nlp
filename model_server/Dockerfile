FROM python:3.11-slim-buster

WORKDIR ./server

COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

RUN apt-get update && apt-get -y install git-lfs
RUN git lfs install
RUN git clone https://huggingface.co/taranetsdan/ruDialoGPT_v2_medium /tmp/model

COPY ./app ./app

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "80"]