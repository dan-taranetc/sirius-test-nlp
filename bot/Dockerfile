FROM python:3.11-slim-buster

WORKDIR ./bot

COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

RUN apt-get update && apt-get -y install ffmpeg flac

COPY ./app ./app

ARG BOT_TOKEN
ENV BOT_TOKEN=${BOT_TOKEN}

CMD python3 ./app/bot.py