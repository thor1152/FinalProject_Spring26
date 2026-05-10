# FinalProject Spring 2026

This project builds an end-to-end ML inference pipeline for the breast cancer dataset using Airflow, S3, SQS, Docker, and Kubernetes.

## Architecture

1. Airflow generates the breast cancer dataset.
2. Airflow trains a logistic regression model.
3. Airflow promotes the trained model, metrics, and metadata to S3.
4. Airflow reads the test dataset and sends one SQS message per record.
5. A Kubernetes consumer polls SQS.
6. The consumer loads the trained model from S3 on startup.
7. The consumer runs inference for each message.
8. The consumer writes one prediction JSON file per record to S3.
9. The consumer deletes SQS messages only after successful processing.

## AWS Resources

- Region: `us-east-1`
- S3 bucket: `seis765-finalproject-560852306721-us-east-1-an`
- SQS queue: `MLOps-FinalProject-Spring26`
- SQS queue URL: `https://sqs.us-east-1.amazonaws.com/560852306721/MLOps-FinalProject-Spring26`
- ECR image: `560852306721.dkr.ecr.us-east-1.amazonaws.com/sqs-consumer:latest`
- EKS cluster: `mlops_final_project_spring26`

## Local Setup

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
source ./setup_airflow.sh
```

## Run Airflow

Quick development mode:

```bash
airflow standalone
```

Or run webserver and scheduler separately.

Terminal 1:

```bash
source venv/bin/activate
source ./setup_airflow.sh
airflow webserver --port 8080
```

Terminal 2:

```bash
source venv/bin/activate
source ./setup_airflow.sh
airflow scheduler
```

If port `8793` is already in use, stop stale Airflow processes or set another log server port:

```bash
export AIRFLOW__LOGGING__WORKER_LOG_SERVER_PORT=8794
```

## Airflow DAGs

Run these DAGs in order:

1. `end_to_end_ml_pipeline`
2. `populate_inference_queue`

The `end_to_end_ml_pipeline` DAG runs:

```text
generate_data -> train_model -> promote_model
```

It creates:

- `data/breast_cancer.csv`
- `data/test_data.csv`
- `models/breast_cancer_model.pkl`
- `eval/breast_cancer_metrics.json`
- `models/breast_cancer_metadata.json`

It promotes model artifacts to S3 under a versioned path and the stable latest path:

```text
models/20260509_053000/
models/latest/
```

The Kubernetes consumer uses:

```text
models/latest/model.pkl
```

## Docker Image

Authenticate Docker to ECR:

```bash
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 560852306721.dkr.ecr.us-east-1.amazonaws.com
```

Build, tag, and push the consumer image:

```bash
docker build -t sqs-consumer .
docker tag sqs-consumer:latest 560852306721.dkr.ecr.us-east-1.amazonaws.com/sqs-consumer:latest
docker push 560852306721.dkr.ecr.us-east-1.amazonaws.com/sqs-consumer:latest
```

## Kubernetes Deployment

Connect to the EKS cluster:

```bash
source kubectl-connect mlops_final_project_spring26
```

Deploy the consumer:

```bash
kubectl apply -f k8s/consumer-deployment.yaml
```

Check status:

```bash
kubectl get nodes
kubectl get pods
kubectl logs deployment/ml-sqs-consumer
```

Scale the consumer to demonstrate multiple replicas:

```bash
kubectl scale deployment/ml-sqs-consumer --replicas=2
kubectl get pods
```

Scale back to one replica:

```bash
kubectl scale deployment/ml-sqs-consumer --replicas=1
```

## Kubernetes AWS Credentials

The consumer uses `boto3`, so the pod needs AWS credentials to access S3 and SQS. The Kubernetes YAML provides the resource locations, but the pod still needs credentials for AWS API calls.

The deployment passes resource locations through environment variables in `k8s/consumer-deployment.yaml`, including the SQS queue URL, S3 bucket name, model key, and AWS region.

In the lab environment, create a temporary Kubernetes secret from the current AWS credentials:

```bash
aws configure export-credentials --format env
```

Run the printed `export` commands, then:

```bash
kubectl delete secret aws-creds --ignore-not-found

kubectl create secret generic aws-creds \
  --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --from-literal=AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN"

kubectl set env deployment/ml-sqs-consumer --from=secret/aws-creds
kubectl rollout restart deployment/ml-sqs-consumer
```

These credentials expire when the lab session expires or restarts. Recreate the secret after a lab restart.

## Message Format

Each SQS message should contain one record:

```json
{
  "batch_id": "20260509T053000Z",
  "record_id": "sample_001",
  "features": [30 numeric values]
}
```

The `features` list must contain exactly 30 values.

## Prediction Output

Each prediction is written to S3 as a separate JSON file:

```json
{
  "record_id": "sample_001",
  "prediction": 1,
  "timestamp": "2026-04-15T12:00:00Z"
}
```

Use unique S3 keys to avoid overwriting predictions across multiple DAG runs:

```text
predictions/20260509T053000Z_sample_001.json
```

For example:

```text
predictions/20260509T053000Z_sample_001.json
```

## Verification Commands

Check promoted model artifacts:

```bash
aws s3 ls s3://seis765-finalproject-560852306721-us-east-1-an/models/latest/
```

Check SQS queue counts:

```bash
aws sqs get-queue-attributes \
  --queue-url "https://sqs.us-east-1.amazonaws.com/560852306721/MLOps-FinalProject-Spring26" \
  --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible ApproximateNumberOfMessagesDelayed
```

Check prediction files:

```bash
aws s3 ls s3://seis765-finalproject-560852306721-us-east-1-an/predictions/
```

Count prediction files:

```bash
aws s3 ls s3://seis765-finalproject-560852306721-us-east-1-an/predictions/ | wc -l
```

View one prediction:

```bash
aws s3 cp s3://seis765-finalproject-560852306721-us-east-1-an/predictions/20260509T053000Z_sample_001.json -
```

## Troubleshooting

If the consumer pod crashes, check the pod logs first:

```bash
kubectl logs deployment/ml-sqs-consumer --tail=100
```

If the consumer pod crashes with `NoCredentialsError`, recreate the `aws-creds` secret and restart the deployment.

If the consumer pod crashes with S3 `404 Not Found`, confirm the model exists:

```bash
aws s3 ls s3://seis765-finalproject-560852306721-us-east-1-an/models/latest/model.pkl
```

If pods are pending, check cluster nodes and events:

```bash
kubectl get nodes
kubectl describe pods -l app=ml-sqs-consumer
kubectl get events -A --sort-by=.lastTimestamp | tail -80
```

If nodes are `NotReady` with `cni plugin not initialized`, install or repair the EKS add-ons:

```bash
aws eks create-addon --region us-east-1 --cluster-name mlops_final_project_spring26 --addon-name vpc-cni
aws eks create-addon --region us-east-1 --cluster-name mlops_final_project_spring26 --addon-name kube-proxy
aws eks create-addon --region us-east-1 --cluster-name mlops_final_project_spring26 --addon-name coredns
```
