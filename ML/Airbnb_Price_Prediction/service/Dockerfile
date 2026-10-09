FROM python:3.12-slim
WORKDIR /app
COPY ML/Airbnb_Price_Prediction/service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY ML/Airbnb_Price_Prediction/airbnb_coastal_prices.csv /app/data.csv
COPY ML/Airbnb_Price_Prediction/service/train.py /app/train.py
COPY ML/Airbnb_Price_Prediction/service/app.py /app/app.py
RUN python train.py --data /app/data.csv --output /app/model.joblib --metrics /app/metrics.json
ENV MODEL_PATH=/app/model.joblib
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
