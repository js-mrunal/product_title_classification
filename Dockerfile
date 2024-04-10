FROM python:3.11-slim
ENV PYTHONUNBUFFERED True

RUN apt-get update
RUN apt-get install -y gcc python3-dev

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install pipenv
RUN pipenv lock
RUN pipenv install --deploy --system

ENV PORT 8080
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1