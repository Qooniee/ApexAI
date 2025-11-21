#!/usr/bin/env python3
"""ApexAI Startup Database Health Check.

Database connection verification script for container startup.
"""

import os
import time

import psycopg2


def wait_for_postgres(
    host=None, port=None, user=None, password=None, database=None, max_attempts=30
):
    """Wait for PostgreSQL connection."""
    # Get configuration from environment variables (required, no defaults)
    host = host or os.getenv("DB_HOST")
    port = port or int(os.getenv("DB_PORT", "5432"))
    user = user or os.getenv("POSTGRES_USER")
    password = password or os.getenv("POSTGRES_PASSWORD")
    database = database or os.getenv("POSTGRES_DB_MLFLOW")

    if not all([host, user, password, database]):
        print("❌ Required environment variables are not set")
        print("Required variables: DB_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB_MLFLOW")
        return False

    print(f"Waiting for PostgreSQL ({database}) to start...")
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(
                host=host, port=port, user=user, password=password, database=database
            )
            conn.close()
            print(f"PostgreSQL ({database}) connection successful!")
            return True
        except psycopg2.OperationalError as e:
            print(f"Connection attempt {attempt + 1}/{max_attempts}... ({e})")
            time.sleep(2)

    print(f"❌ Failed to connect to PostgreSQL ({database})")
    return False


def verify_databases():
    """Verify ApexAI database connections."""
    print("=== ApexAI Database Connection Verification ===")

    # Verify MLflow database
    mlflow_db = os.getenv("POSTGRES_DB_MLFLOW")
    if not mlflow_db:
        print("❌ POSTGRES_DB_MLFLOW environment variable is not set")
        return False

    if wait_for_postgres(database=mlflow_db):
        print("✅ MLflow database connection OK")
    else:
        print("❌ MLflow database connection failed")
        return False

    # Verify Optuna database
    optuna_db = os.getenv("POSTGRES_DB_OPTUNA")
    if not optuna_db:
        print("❌ POSTGRES_DB_OPTUNA environment variable is not set")
        return False

    if wait_for_postgres(database=optuna_db):
        print("✅ Optuna database connection OK")
    else:
        print("❌ Optuna database connection failed")
        return False

    print("=== Environment Variables Check ===")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "Not set")
    db_host = os.getenv("DB_HOST", "Not set")
    db_user = os.getenv("POSTGRES_USER", "Not set")

    print(f"MLFLOW_TRACKING_URI: {mlflow_uri}")
    print(f"DB_HOST: {db_host}")
    print(f"POSTGRES_USER: {db_user}")

    return True


if __name__ == "__main__":
    if verify_databases():
        print("\n✅ ApexAI database startup verification completed!")
        print("\n🌐 Available services:")
        print("  📊 MLflow UI:        http://localhost:5001")
        print("  🎯 Optuna Dashboard: http://localhost:8081")
        print("  🗄️ MinIO Console:    http://localhost:9020")
        print("  💾 pgAdmin:          http://localhost:5051")
        print("\n🚀 Run experiment example:")
        print(
            "  docker exec apexai_predictor bash -c \"cd /workspace && python src/apexai_main.py experiment.name='ApexAI_Test'\""
        )
    else:
        print("\n❌ ApexAI database startup verification failed")
        print("💡 Solution: Run setup_apexai.py to initialize databases")
        exit(1)
