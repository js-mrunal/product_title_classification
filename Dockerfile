FROM python:3.11

RUN apt-get update

RUN apt-get install -y graphviz

RUN pip install pipenv

COPY . /opt/

WORKDIR /opt/

RUN make init

ENV PYTHONPATH="${PYTHONPATH}:/opt/"

RUN make train

ENTRYPOINT ["make", "run-local"]