FROM python:3.9-alpine

ENV PYTHONDONTWRITEBYCODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
ENV PYTHONPATH /app

COPY . /app
RUN pip install -r requirements.txt