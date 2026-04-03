FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default run path uses deterministic rules policy so the container works without secrets.
CMD ["python", "baseline/run_agent.py", "--policy", "rules"]
