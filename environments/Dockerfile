FROM python:3.8-slim-buster AS build-env
WORKDIR /build

RUN apt-get update
RUN apt-get install -y libglpk-dev
RUN apt-get install -y curl unzip gcc vim
RUN pip3 install --upgrade pip

# AWS S3 CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install

# install deps
COPY environments/requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install -U numpy
RUN rm /tmp/requirements.txt

COPY . .
RUN pip3 install /build/

FROM python:3.8-slim-buster
WORKDIR /pybalance/
COPY --from=build-env /usr/local /usr/local
COPY --from=build-env /usr/local/bin/aws /usr/local/bin/
COPY --from=build-env /usr/local/aws-cli /usr/local/aws-cli
COPY --from=build-env /build /pybalance

# expose jupyter
EXPOSE 8888
EXPOSE 8080
