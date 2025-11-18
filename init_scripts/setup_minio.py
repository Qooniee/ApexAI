#!/usr/bin/env python3
"""MinIO setup script for MLflow artifact storage."""

import os
import time

import boto3
from botocore.exceptions import ClientError


def setup_minio_bucket():
    """Create MLflow bucket in MinIO."""
    # Get configuration from environment variables
    endpoint_url = os.getenv("MINIO_ENDPOINT_URL")
    access_key = os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD")
    bucket_name = os.getenv("MINIO_BUCKET_NAME")

    if not all([endpoint_url, access_key, secret_key, bucket_name]):
        print("❌ MinIO environment variables not properly set")
        return False

    print("MinIO Configuration:")
    print(f"  Endpoint: {endpoint_url}")
    print(f"  Bucket: {bucket_name}")

    # Create S3 client (for MinIO)
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",  # Specify any region for MinIO
    )

    # Wait for MinIO to start
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            s3_client.list_buckets()
            print("MinIO server connection successful!")
            break
        except Exception as e:
            print(f"MinIO connection attempt {attempt + 1}/{max_attempts}... ({e})")
            time.sleep(2)
    else:
        print("Failed to connect to MinIO server")
        return False

    # Create bucket
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists")
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            try:
                s3_client.create_bucket(Bucket=bucket_name)
                print(f"Created bucket '{bucket_name}'")
            except Exception as create_error:
                print(f"Failed to create bucket: {create_error}")
                return False
        else:
            print(f"Failed to check bucket: {e}")
            return False

    # Verify bucket
    try:
        s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
        print(f"Bucket '{bucket_name}' verification completed")
        return True
    except Exception as e:
        print(f"Failed to verify bucket: {e}")
        return False


if __name__ == "__main__":
    print("=== MinIO Setup ===")
    if setup_minio_bucket():
        print("MinIO setup completed successfully!")
    else:
        print("MinIO setup failed")
