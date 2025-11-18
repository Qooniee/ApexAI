#!/usr/bin/env python3
"""ApexAI Services Health Check Script.

Integrated health check functionality for all services.
"""

import os
import sys
import time

import psycopg2
import requests


def check_database_connection() -> bool:
    """Check PostgreSQL database connection."""
    print("🔍 Checking PostgreSQL database connection...")

    try:
        # Get connection information from environment variables
        host = os.getenv("DB_HOST")
        port = int(os.getenv("DB_PORT", "5433"))
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")

        if not all([host, user, password]):
            print("  ❌ Database environment variables not set")
            return False

        # Test MLflow database connection
        mlflow_db = os.getenv("POSTGRES_DB_MLFLOW")
        if mlflow_db:
            conn_mlflow = psycopg2.connect(
                host=host, port=port, database=mlflow_db, user=user, password=password
            )
            conn_mlflow.close()
            print(f"  ✅ MLflow DB ({mlflow_db}) connection successful")

        # Test Optuna database connection
        optuna_db = os.getenv("POSTGRES_DB_OPTUNA")
        if optuna_db:
            conn_optuna = psycopg2.connect(
                host=host, port=port, database=optuna_db, user=user, password=password
            )
            conn_optuna.close()
            print(f"  ✅ Optuna DB ({optuna_db}) connection successful")

        return True

    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
        return False


def check_web_service(name: str, url: str, expected_status: int = 200, timeout: int = 10) -> bool:
    """Check web service health."""
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        if response.status_code == expected_status or (200 <= response.status_code < 400):
            print(f"  ✅ {name} is accessible ({response.status_code})")
            return True
        print(f"  ❌ {name} returned status {response.status_code}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"  ❌ {name} connection failed: {e}")
        return False


def check_minio_service() -> bool:
    """Check MinIO S3 service."""
    print("🔍 Checking MinIO S3 service...")

    try:
        # Connect directly to MinIO API endpoint and check status
        minio_endpoint = os.getenv("MINIO_ENDPOINT_URL")
        if not minio_endpoint:
            print("  ❌ MINIO_ENDPOINT_URL environment variable not set")
            return False
        response = requests.get(minio_endpoint, timeout=5)
        if response.status_code in [400, 403]:  # Normal MinIO response
            print("  ✅ MinIO is accessible and responding")
            return True
        print(f"  ⚠️  MinIO returned status {response.status_code} (may still be functional)")
        return True
    except requests.exceptions.ConnectionError:
        print("  ❌ MinIO connection refused")
        return False
    except requests.exceptions.Timeout:
        print("  ❌ MinIO connection timeout")
        return False
    except Exception as e:
        print(f"  ❌ MinIO check failed: {e}")
        return False


def check_pgadmin_service() -> bool:
    """Health check specifically for pgAdmin (with longer wait time)."""
    print("🔍 Checking pgAdmin service...")

    # pgAdmin takes longer to start up, so set a longer timeout
    for attempt in range(8):  # 8 attempts (up to 2 minutes)
        try:
            response = requests.get("http://apexai_pgadmin", timeout=10)
            if response.status_code == 200:
                print("  ✅ pgAdmin is accessible (200)")
                return True
            print(f"  ⚠️  pgAdmin returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            if attempt < 7:  # Wait quietly if not the last attempt
                print(f"  ⏳ pgAdmin starting up... (attempt {attempt + 1}/8)")
                time.sleep(15)  # Wait 15 seconds
            else:
                print("  ❌ pgAdmin connection refused after extended wait")
                return False
        except Exception as e:
            if attempt < 7:
                time.sleep(15)
            else:
                print(f"  ❌ pgAdmin check failed: {e}")
                return False

    return False


def check_all_services(max_attempts: int = 8, wait_between: int = 15) -> dict[str, bool]:
    """Integrated health check for all services."""
    services_config = [
        ("PostgreSQL", check_database_connection),
        ("MinIO", check_minio_service),
        ("MLflow UI", lambda: check_web_service("MLflow", "http://apexai_mlflow:5000")),
        (
            "Optuna Dashboard",
            lambda: check_web_service("Optuna", "http://apexai_optuna:8080"),
        ),
        ("pgAdmin", check_pgadmin_service),
    ]

    results = {}

    for attempt in range(max_attempts):
        print(f"\n🔍 Health Check Attempt {attempt + 1}/{max_attempts}")
        print("=" * 50)

        all_healthy = True

        for service_name, check_func in services_config:
            if service_name not in results or not results[service_name]:
                print(f"Checking {service_name}...")
                try:
                    results[service_name] = check_func()
                except Exception as e:
                    print(f"  ❌ {service_name} check failed: {e}")
                    results[service_name] = False

                if not results[service_name]:
                    all_healthy = False

        if all_healthy:
            print("\n🎉 All services are healthy!")
            return results

        if attempt < max_attempts - 1:
            print(f"\n⏳ Some services not ready. Waiting {wait_between} seconds...")
            time.sleep(wait_between)

    return results


def print_service_summary(results: dict[str, bool]):
    """Display service status summary."""
    print("\n" + "=" * 60)
    print("📊 ApexAI Services Health Summary")
    print("=" * 60)

    for service_name, is_healthy in results.items():
        status = "✅ HEALTHY" if is_healthy else "❌ UNHEALTHY"
        print(f"  {service_name:20} {status}")

    healthy_count = sum(1 for status in results.values() if status)
    total_count = len(results)

    print(f"\nOverall: {healthy_count}/{total_count} services healthy")

    if healthy_count == total_count:
        print("🎉 All ApexAI services are operational!")
        return True
    print("⚠️  Some services require attention")
    return False


def main():
    """Main execution function."""
    print("🚀" + "=" * 58)
    print("   ApexAI Services Health Check")
    print("=" * 60)

    # Execute all service checks
    results = check_all_services()

    # Display summary
    all_healthy = print_service_summary(results)

    # Display service URLs
    print("\n🌐 Service Access URLs:")
    print("  📊 MLflow UI:        http://localhost:5001")
    print("  🎯 Optuna Dashboard: http://localhost:8081")
    print("  🗄️ MinIO Console:    http://localhost:9020")
    print("  💾 pgAdmin:          http://localhost:5051")

    if not all_healthy:
        sys.exit(1)

    print("\n✅ Health check completed successfully!")


if __name__ == "__main__":
    main()
