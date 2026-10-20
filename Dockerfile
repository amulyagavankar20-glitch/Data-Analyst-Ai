FROM python:3.12.8-slim@sha256:2199a62885a12290dc9c5be3ca0681d367576ab7bf037da120e564723292a2f0

WORKDIR /app

ENV HOME=/home/analyst PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt ./
COPY src/ ./src/
COPY app.py .
COPY .streamlit/ ./.streamlit/

RUN pip3 install --no-cache-dir -r requirements.txt \
    && groupadd --system --gid 10001 analyst \
    && useradd --system --uid 10001 --gid analyst --create-home analyst

USER 10001:10001

EXPOSE 8501

HEALTHCHECK CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=2).read()"]

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
