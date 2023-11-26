FROM python:3.8.18

WORKDIR /workspace
COPY ./requirements.txt . 

RUN pip install -r requirements.txt
