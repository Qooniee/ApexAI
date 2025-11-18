#!/usr/bin/env python3
"""MLflow Database Initialization Script."""

import os
import subprocess
import sys
import time

import psycopg2


def wait_for_postgres(
    host=None, port=None, user=None, password=None, database=None, max_attempts=30
):
    """Wait for PostgreSQL connection."""
    # Get configuration from environment variables
    host = host or os.getenv("DB_HOST")
    port = port or int(os.getenv("DB_PORT", "5433"))
    user = user or os.getenv("POSTGRES_USER")
    password = password or os.getenv("POSTGRES_PASSWORD")
    database = database or os.getenv("POSTGRES_DB_MLFLOW")

    print(f"Waiting for PostgreSQL to start at {host}:{port}...")
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(
                host=host, port=port, user=user, password=password, database=database
            )
            conn.close()
            print("PostgreSQL connection successful!")
            return True
        except psycopg2.OperationalError:
            print(f"Connection attempt {attempt + 1}/{max_attempts}...")
            time.sleep(2)
    return False


def setup_mlflow_database():
    """Initialize MLflow database."""
    # Get configuration from environment variables
    host = os.getenv("DB_HOST")
    port = int(os.getenv("DB_PORT", "5433"))
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("POSTGRES_DB_MLFLOW")

    if not all([host, user, password, database]):
        print("Required environment variables not set")
        return False

    db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    print("Checking MLflow database state...")

    # First, check if database is empty
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, database=database
        )
        cursor = conn.cursor()

        # Check for table existence
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'alembic_version'
            );
        """)
        has_alembic = cursor.fetchone()[0]

        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'experiments'
            );
        """)
        has_experiments = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        if has_alembic and has_experiments:
            print("MLflow database already initialized, skipping setup")
            return True
        if has_alembic and not has_experiments:
            print("Partial MLflow database detected, trying upgrade...")
        else:
            print("Empty database detected, initializing MLflow schema...")

    except Exception as e:
        print(f"Database check failed: {e}")
        print("Proceeding with initialization...")

    # Execute MLflow initialization
    try:
        # First try upgrade (normal initialization)
        print("Initializing MLflow database...")
        subprocess.run(
            [sys.executable, "-m", "mlflow", "db", "upgrade", db_url],
            capture_output=True,
            text=True,
            check=True,
        )

        print("✅ MLflow database initialization successful!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"⚠️  First initialization attempt failed: {e.returncode}")

        # If failed, force schema creation
        try:
            print("Attempting clean database initialization...")

            # Clean up database before initialization
            conn = psycopg2.connect(
                host=host, port=port, user=user, password=password, database=database
            )
            conn.autocommit = True
            cursor = conn.cursor()

            # Drop existing tables if any
            cursor.execute("DROP SCHEMA IF EXISTS public CASCADE;")
            cursor.execute("CREATE SCHEMA public;")
            cursor.execute("GRANT ALL ON SCHEMA public TO public;")

            cursor.close()
            conn.close()

            print("Database cleaned, retrying MLflow initialization...")

            # Retry initialization
            subprocess.run(
                [sys.executable, "-m", "mlflow", "db", "upgrade", db_url],
                capture_output=True,
                text=True,
                check=True,
            )

            print("✅ MLflow database initialization successful after cleanup!")
            return True

        except Exception as cleanup_error:
            print(f"❌ Clean initialization also failed: {cleanup_error}")
            return False


def main():
    """Run MLflow database setup."""
    print("=== MLflow Database Setup ===")

    # Wait for PostgreSQL connection
    if not wait_for_postgres():
        print("Failed to connect to PostgreSQL")
        sys.exit(1)

    # Initialize MLflow database
    if not setup_mlflow_database():
        print("Failed to initialize MLflow database")
        sys.exit(1)

    print("Setup completed!")


if __name__ == "__main__":
    main()
