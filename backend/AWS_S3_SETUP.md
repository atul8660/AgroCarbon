# AWS S3 Cloud Storage Integration Guide

This document defines the production-grade architecture for setting up and managing the AWS S3 storage backend for the AgroCarbonAI Engine.

## 1. AWS Account Setup
1. Log into your [AWS Management Console](https://aws.amazon.com/console/).
2. Navigate to the **IAM (Identity and Access Management)** dashboard.

## 2. S3 Bucket Creation
1. Navigate to **S3** inside AWS.
2. Click **Create bucket**.
3. Set the name to exactly match what's in your `.env` (e.g. `tree-carbon-system`).
4. Select `us-east-1` (or your preferred region).
5. **Disable** "Block all public access" to ensure images generate public-facing CDN URLs for the mobile app.
6. Acknowledge the warning and click **Create bucket**.
7. Go to **Permissions** -> **Bucket Policy** and grant public read access:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::tree-carbon-system/*"
        }
    ]
}
```

## 3. IAM User Setup
1. In **IAM**, click **Users** -> **Add users**.
2. Name user `agrocarbonai-svc`.
3. Select **Attach policies directly**.
4. Attach `AmazonS3FullAccess` (Or secure it precisely using inline json policies just to the bucket).
5. Open the user, navigate to **Security credentials**, and click **Create access key**.
6. Select **Application running outside AWS**.
7. Copy your `Access Key ID` and `Secret Access Key`.

## 4. Environment Variables
Add these values directly into `.env`:
```
AWS_ACCESS_KEY_ID="AKIAEXAMPLE..."
AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
AWS_REGION="us-east-1"
AWS_S3_BUCKET_NAME="tree-carbon-system"
MAX_UPLOAD_SIZE_MB=10
```

## 5. Running Locally & Testing
Since the `.env` variables abstract the physical hardware, your local desktop will automatically bridge seamlessly to AWS S3 without localized configurations.
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Execute unit tests directly against the local Moto engine:
```bash
pytest tests/test_storage.py
```

## 6. Troubleshooting
- **Missing Credentials**: Ensure your local active terminal has inherited the updated `.env` state, or forcefully insert `os.environ` prior to `import boto3`.
- **Bucket Not Found**: The region string in `AWS_REGION` strictly must match where the physical bucket was provisioned. Ensure no typos in `AWS_S3_BUCKET_NAME`.
- **Upload Timeout / Network Failures**: Ensure your corporate firewall or ISP is not aggressively shaping AWS-bound payloads. Validates maximum file sizes (`MAX_UPLOAD_SIZE_MB`).
