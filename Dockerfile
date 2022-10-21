from python:3.9.15-slim-buster
RUN apt-get update -y && apt-get install -y python3-pip python3-dev build-essential git ffmpeg libsm6 libxext6

COPY requirements.txt /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

COPY .env /app/
COPY src/load_weights.py /app/src/
RUN /bin/sh -c "python3 /app/src/load_weights.py"

COPY . /app

CMD [ "python3" , "/app/src/app.py" ]