# 🚀 Deployment Guide

This guide covers different deployment options for the Customer Churn Prediction project.

## 📋 Prerequisites

- Python 3.8+
- Git
- Virtual environment tool
- (Optional) Docker
- (Optional) Cloud account (AWS/GCP/Azure)

## 🏠 Local Development Deployment

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# Setup environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run analysis
python demo_kaggle.py
```

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 jupyter

# Run tests
pytest tests/

# Format code
black .

# Lint code
flake8 .
```

## 🐳 Docker Deployment

### Build Docker Image
```bash
# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Default command
CMD ["python", "demo_kaggle.py"]
EOF

# Build image
docker build -t customer-churn-prediction .

# Run container
docker run -p 8501:8501 customer-churn-prediction
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  churn-analysis:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./results:/app/results
    environment:
      - PYTHONPATH=/app
    command: python demo_kaggle.py

  dashboard:
    build: .
    ports:
      - "8502:8501"
    volumes:
      - ./data:/app/data
    command: streamlit run dashboard/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

## ☁️ Cloud Deployment

### AWS Deployment

#### EC2 Instance
```bash
# Launch EC2 instance (Ubuntu 20.04)
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv git -y

# Clone and setup project
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run analysis
python demo_kaggle.py

# For dashboard (install nginx for production)
sudo apt install nginx -y
streamlit run dashboard/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

#### AWS Lambda (Serverless)
```python
# lambda_function.py
import json
import sys
import os

# Add src to path
sys.path.append('/opt/python/src')

from data_preprocessing_kaggle import KaggleDataPreprocessor
from financial_impact import FinancialImpactAnalyzer

def lambda_handler(event, context):
    try:
        # Process customer data
        preprocessor = KaggleDataPreprocessor()
        # ... analysis logic ...
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'churn_rate': 0.265,
                'revenue_at_risk': 3338773.57,
                'recommendations': ['Contract conversion', 'Payment optimization']
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Google Cloud Platform

#### Cloud Run
```bash
# Create cloudbuild.yaml
cat > cloudbuild.yaml << EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/\$PROJECT_ID/churn-prediction', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/\$PROJECT_ID/churn-prediction']
  - name: 'gcr.io/cloud-builders/gcloud'
    args: ['run', 'deploy', 'churn-prediction', 
           '--image', 'gcr.io/\$PROJECT_ID/churn-prediction',
           '--platform', 'managed',
           '--region', 'us-central1',
           '--allow-unauthenticated']
EOF

# Deploy
gcloud builds submit --config cloudbuild.yaml
```

#### Vertex AI
```python
# vertex_ai_deployment.py
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project="your-project-id", location="us-central1")

# Create custom training job
job = aiplatform.CustomTrainingJob(
    display_name="churn-prediction-training",
    script_path="src/churn_models.py",
    container_uri="gcr.io/cloud-aiplatform/training/pytorch-gpu.1-9:latest",
    requirements=["pandas", "scikit-learn", "kagglehub"],
    model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/pytorch-gpu.1-9:latest",
)

# Run training
model = job.run(
    dataset=dataset,
    model_display_name="churn-prediction-model",
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
)

# Deploy model
endpoint = model.deploy(
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=10,
)
```

### Microsoft Azure

#### Azure Container Instances
```bash
# Create resource group
az group create --name churn-prediction-rg --location eastus

# Deploy container
az container create \
    --resource-group churn-prediction-rg \
    --name churn-prediction \
    --image your-registry/customer-churn-prediction:latest \
    --cpu 2 \
    --memory 4 \
    --ports 8501 \
    --environment-variables PYTHONPATH=/app
```

## 🔧 Production Configuration

### Environment Variables
```bash
# .env file
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### Logging Configuration
```python
# logging_config.py
import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('churn_analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
```

### Database Integration
```python
# database.py
import psycopg2
import pandas as pd

class DatabaseManager:
    def __init__(self, connection_string):
        self.conn = psycopg2.connect(connection_string)
    
    def save_predictions(self, predictions_df):
        predictions_df.to_sql('churn_predictions', self.conn, if_exists='replace')
    
    def load_customer_data(self):
        return pd.read_sql('SELECT * FROM customers', self.conn)
```

## 📊 Monitoring & Maintenance

### Health Checks
```python
# health_check.py
from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/health')
def health_check():
    try:
        # Test core functionality
        df = pd.DataFrame({'test': [1, 2, 3]})
        return jsonify({'status': 'healthy', 'timestamp': pd.Timestamp.now().isoformat()})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### Performance Monitoring
```python
# monitoring.py
import time
import psutil
import logging

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        logging.info(f"Function {func.__name__} took {end_time - start_time:.2f}s")
        logging.info(f"Memory usage: {end_memory - start_memory:.2f}MB")
        
        return result
    return wrapper
```

## 🔒 Security Considerations

### API Security
```python
# security.py
from functools import wraps
import jwt
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.environ.get('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function
```

### Data Privacy
```python
# privacy.py
import hashlib

def anonymize_customer_id(customer_id):
    """Hash customer ID for privacy"""
    return hashlib.sha256(customer_id.encode()).hexdigest()[:16]

def remove_pii(df):
    """Remove personally identifiable information"""
    pii_columns = ['email', 'phone', 'address', 'name']
    return df.drop(columns=[col for col in pii_columns if col in df.columns])
```

## 📈 Scaling Considerations

### Horizontal Scaling
- Use load balancers (nginx, HAProxy)
- Implement caching (Redis, Memcached)
- Database read replicas
- Microservices architecture

### Vertical Scaling
- Optimize pandas operations
- Use vectorized computations
- Implement batch processing
- Memory-efficient data loading

### Auto-scaling
```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-prediction
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-prediction
  template:
    metadata:
      labels:
        app: churn-prediction
    spec:
      containers:
      - name: churn-prediction
        image: your-registry/churn-prediction:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: churn-prediction-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: churn-prediction
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 🎯 Deployment Checklist

### Pre-deployment
- [ ] Code review completed
- [ ] Tests passing
- [ ] Security scan completed
- [ ] Performance testing done
- [ ] Documentation updated

### Deployment
- [ ] Environment variables configured
- [ ] Database migrations run
- [ ] Health checks implemented
- [ ] Monitoring configured
- [ ] Backup strategy in place

### Post-deployment
- [ ] Smoke tests passed
- [ ] Monitoring alerts configured
- [ ] Performance metrics baseline established
- [ ] Rollback plan tested
- [ ] Team notified

---

**Choose the deployment option that best fits your infrastructure and requirements. Start with local development, then move to cloud deployment as your needs scale.**
