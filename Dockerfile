FROM python:3.6.9-stretch

RUN pip install torch==1.8.1 torchvision==0.9.1 efficientnet-pytorch==0.7.1

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace

COPY ./src ./src
COPY ./model ./model
COPY ./setup.py .
RUN pip install -e .

ENV FLASK_APP=src/app
ENV FLASK_ENV=production
ENV CAPTAIN_EMAIL=allenwu9453@gmail.com
ENV SALT=my_salt
ENV MODEL_PATH=model/efficientnet_b4.pt

CMD flask run --host=0.0.0.0 >> /tmp/esun_ai_2021_summer_tradevan.log 2>&1
