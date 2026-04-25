FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY /src ./src/
COPY /app ./app
COPY /data ./data
COPY /best_model.joblib ./best_model.joblib
COPY /transformer.joblib ./transformer.joblib
COPY /selected_features.joblib ./selected_features.joblib

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]