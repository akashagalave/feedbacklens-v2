FROM python:3.11-slim

WORKDIR /app

COPY services/insight_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/insight_agent/app ./app
COPY shared ./shared

EXPOSE 8003

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]