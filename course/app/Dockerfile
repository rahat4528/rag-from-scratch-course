# Lightweight container for the Novasphere RAG From Scratch Capstone Web App
FROM python:3.11-slim

WORKDIR /app

COPY course/requirements.txt /app/course/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/course/requirements.txt

COPY course/ /app/course/

ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV RAG_PROVIDER=offline_demo

EXPOSE 7860

CMD ["python", "course/app/app.py"]

