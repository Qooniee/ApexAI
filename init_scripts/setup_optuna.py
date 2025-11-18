#!/usr/bin/env python3
"""Optuna Study Initialization Script."""

import os
import time

import optuna


def wait_for_database(max_attempts=30):
    """Wait for database connection."""
    print("Waiting for Optuna database to be ready...")

    # Get DB connection information from environment variables
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("POSTGRES_DB_OPTUNA")

    storage_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

    for attempt in range(max_attempts):
        try:
            # Simple connection test
            optuna.storages.RDBStorage(storage_url)
            print("Optuna database connection successful!")
            return storage_url
        except Exception as e:
            print(f"Connection attempt {attempt + 1}/{max_attempts}... ({e})")
            time.sleep(2)

    raise Exception("Failed to connect to Optuna database")


def setup_optuna_studies():
    """Initialize Optuna Study."""
    print("\n=== Optuna Studies Setup ===")

    # Wait for database connection
    storage_url = wait_for_database()

    studies_config = [
        {
            "name": "ApexAI_Setup_Study",
            "direction": "minimize",
            "description": "Main study for ApexAI optimization and testing",
        }
    ]

    created_studies = []

    for config in studies_config:
        try:
            study = optuna.create_study(
                storage=storage_url,
                study_name=config["name"],
                direction=config["direction"],
                load_if_exists=True,
            )
            print(f"✅ Created/loaded study: {study.study_name}")
            created_studies.append(study.study_name)

        except Exception as e:
            print(f"❌ Failed to create study {config['name']}: {e}")
            return False

    print(f"\n✅ Optuna setup completed! Created {len(created_studies)} studies:")
    for study_name in created_studies:
        print(f"  - {study_name}")

    return True


if __name__ == "__main__":
    try:
        if setup_optuna_studies():
            print("Optuna studies setup completed successfully!")
        else:
            print("Optuna studies setup failed!")
            exit(1)
    except Exception as e:
        print(f"Setup error: {e}")
        exit(1)
