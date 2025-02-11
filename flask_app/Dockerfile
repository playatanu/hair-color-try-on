FROM python:3.12.9

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip install opencv-python-headless

COPY src src
COPY run.py run.py

EXPOSE 5000

CMD ["python3", "run.py"]
