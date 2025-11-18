#!/usr/bin/env python3
"""PostgreSQL Database Setup Script.

Initialize databases for MLflow and Optuna.
"""

import logging
import os
import time

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def wait_for_postgres(host, port, user, password, database, max_attempts=30):
    """Wait for PostgreSQL connection."""
    logging.info(f"Waiting for PostgreSQL at {host}:{port}...")

    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(
                host=host, port=port, user=user, password=password, database=database
            )
            conn.close()
            logging.info("PostgreSQL connection successful!")
            return True
        except psycopg2.OperationalError as e:
            logging.info(f"Connection attempt {attempt + 1}/{max_attempts}... ({e})")
            time.sleep(2)

    logging.error("Failed to connect to PostgreSQL")
    return False


def database_exists(host, port, user, password, admin_db, target_db):
    """Check if database exists."""
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, database=admin_db
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
        exists = cursor.fetchone() is not None

        cursor.close()
        conn.close()
        return exists
    except Exception as e:
        logging.exception(f"Failed to check database existence: {e}")
        return False


def create_database(host, port, user, password, admin_db, target_db):
    """Create database."""
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, database=admin_db
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        cursor = conn.cursor()
        # Database names must be built as direct strings
        # (psycopg2 does not support parameterization for DDL)
        cursor.execute(f'CREATE DATABASE "{target_db}"')
        cursor.execute(f'GRANT ALL PRIVILEGES ON DATABASE "{target_db}" TO "{user}"')

        cursor.close()
        conn.close()
        logging.info(f"Database '{target_db}' created successfully")
        return True
    except Exception as e:
        logging.exception(f"Failed to create database '{target_db}': {e}")
        return False


def setup_databases():
    """ApexAI PostgreSQL database setup."""
    # Get configuration from environment variables
    host = os.getenv("DB_HOST")
    port = int(os.getenv("DB_PORT", "5432"))
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    mlflow_db = os.getenv("POSTGRES_DB_MLFLOW")
    optuna_db = os.getenv("POSTGRES_DB_OPTUNA")
    admin_db = "postgres"  # Default admin database

    # Check required environment variables
    if not all([host, user, password, mlflow_db, optuna_db]):
        logging.error("Required environment variables not set")
        logging.error(
            "Required: DB_HOST, POSTGRES_USER, POSTGRES_PASSWORD, "
            "POSTGRES_DB_MLFLOW, POSTGRES_DB_OPTUNA"
        )
        return False

    logging.info("=== PostgreSQL Database Setup ===")
    logging.info(f"Host: {host}:{port}")
    logging.info(f"User: {user}")
    logging.info(f"MLflow DB: {mlflow_db}")
    logging.info(f"Optuna DB: {optuna_db}")

    # Wait for PostgreSQL connection
    if not wait_for_postgres(host, port, user, password, admin_db):
        return False

    success = True

    # Create MLflow database
    if database_exists(host, port, user, password, admin_db, mlflow_db):
        logging.info(f"Database '{mlflow_db}' already exists")
    else:
        logging.info(f"Creating database '{mlflow_db}'...")
        if create_database(host, port, user, password, admin_db, mlflow_db):
            logging.info(f"✅ Database '{mlflow_db}' created")
        else:
            logging.error(f"❌ Failed to create database '{mlflow_db}'")
            success = False

    # Create Optuna database
    if database_exists(host, port, user, password, admin_db, optuna_db):
        logging.info(f"Database '{optuna_db}' already exists")
    else:
        logging.info(f"Creating database '{optuna_db}'...")
        if create_database(host, port, user, password, admin_db, optuna_db):
            logging.info(f"✅ Database '{optuna_db}' created")
        else:
            logging.error(f"❌ Failed to create database '{optuna_db}'")
            success = False

    return success


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if setup_databases():
        print("✅ PostgreSQL databases setup completed successfully!")
    else:
        print("❌ PostgreSQL databases setup failed")
        exit(1)
