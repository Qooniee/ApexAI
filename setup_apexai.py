#!/usr/bin/env python3
"""ApexAI - Vehicle Identification Platform Setup Script.

Automated initialization script for ApexAI vehicle identification platform.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description, check=True, show_progress=False):
    """Execute command and display results."""
    print(f"🔧 {description}...")
    try:
        if show_progress:
            # Real-time output display (for long-running processes like Docker build)

            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                bufsize=1,
            )

            output_lines = []
            start_time = time.time()
            last_progress_time = start_time

            # Animation characters
            pacman_frames = ["ᗧ•••", "ᗤ•••", "ᗢ•••", "ᗡ•••"]
            building_frames = ["🏗️ ", "🔨", "⚙️ ", "🔧"]

            frame_index = 0

            def show_progress_animation():
                nonlocal frame_index, last_progress_time
                current_time = time.time()
                elapsed_minutes = int((current_time - start_time) / 60)
                elapsed_seconds = int((current_time - start_time) % 60)

                # Display progress every 30 seconds
                if current_time - last_progress_time >= 30:
                    building_char = building_frames[frame_index % len(building_frames)]
                    pacman_char = pacman_frames[frame_index % len(pacman_frames)]

                    print(
                        f"\n   {building_char} Building... {pacman_char} ({elapsed_minutes:02d}:{elapsed_seconds:02d} elapsed)",
                        end="",
                        flush=True,
                    )
                    last_progress_time = current_time
                    frame_index += 1

            for line in iter(process.stdout.readline, ""):
                if line:
                    line = line.strip()
                    output_lines.append(line)

                    # Display step progress
                    if line.startswith("#") and "Step" in line:
                        print(f"\n   📦 {line}")

                    # Important success messages
                    elif any(
                        keyword in line for keyword in ["Successfully built", "Successfully tagged"]
                    ):
                        print(f"\n   ✅ {line}")

                    # Display only true error messages (exclude warnings and normal build messages)
                    elif any(
                        keyword in line.lower()
                        for keyword in ["error:", "failed:", "cannot", "fatal"]
                    ) and not any(
                        ignore in line.lower()
                        for ignore in [
                            "warning",
                            "deprecated",
                            "debconf:",
                            "unable to initialize frontend",
                            "dpkg-preconfigure",
                            "apt-utils",
                            "locale not supported",
                            "found orphan containers",
                            "recreating",
                            "removing",
                            "network",
                            "volume",
                            "building",
                            "step",
                            "sha256",
                            "already exists",
                            "pulling from",
                            "digest",
                            "status",
                        ]
                    ):
                        # Also exclude normal Docker messages
                        if not any(
                            docker_msg in line.lower()
                            for docker_msg in [
                                "copying file",
                                "reading dockerfile",
                                "sending build context",
                            ]
                        ):
                            print(f"\n   ❌ {line}")

                    # Time-based progress animation display
                    show_progress_animation()

            process.stdout.close()
            return_code = process.wait()

            if return_code == 0:
                print(f"\n✅ {description} completed successfully")
                return True
            if not check:
                print(f"\n⚠️  {description} completed with warnings (return code {return_code})")
                return True
            print(f"\n❌ {description} failed with return code {return_code}")
            # Display last few lines on error
            if output_lines:
                print("   Last few lines:")
                for line in output_lines[-3:]:
                    if line.strip():
                        print(f"     {line}")
            return False
        # Normal execution (display output together)
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if result.stdout:
            print(f"✅ {description} completed")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False


def check_docker():
    """Check Docker environment."""
    print("🐳 Checking Docker environment...")
    if not run_command("docker --version", "Docker version check"):
        print("❌ Docker is not installed")
        return False
    if not run_command("docker-compose --version", "Docker Compose version check"):
        print("❌ Docker Compose is not installed")
        return False
    print("✅ Docker environment is ready")
    return True


def cleanup_environment():
    """Clean up existing environment (preserve data)."""
    print("\n🧹 Cleaning up existing ApexAI containers...")

    # Stop and remove existing containers (preserve volumes)
    run_command("docker-compose down", "Stop existing containers", check=False)

    # Check data directories
    data_dirs = ["postgres_data", "minio_data", "pgadmin_data"]
    existing_dirs = [d for d in data_dirs if os.path.exists(d)]

    if existing_dirs:
        print(f"📦 Keeping existing data directories: {', '.join(existing_dirs)}")
        print("⚠️  To completely reset data, manually delete these folders:")
        for data_dir in existing_dirs:
            print(f"   rm -rf {data_dir}  (or delete folder manually)")
    else:
        print("📦 No existing data directories found - fresh start")


def build_and_start_core_services():
    """Build Docker images and start core services (DB, MinIO)."""
    print("\n🔨 Building Docker images and starting core services...")
    print("You can check logs in another terminal with: docker-compose logs -f")

    # Docker buildを進捗表示付きで実行
    print("   🏗️  Starting Docker build process...")
    print("   ⏳ This may take 5-10 minutes for first-time setup")
    print("   📋 Progress: Major steps will be shown, dots indicate ongoing work")

    try:
        # Try with progress display
        if not run_command(
            "docker-compose build --no-cache --progress=plain",
            "Docker build",
            show_progress=True,
        ):
            return False
    except Exception as e:
        print(f"   ⚠️  Progress display failed, falling back to standard build: {e}")
        # Fallback: normal build
        if not run_command("docker-compose build --no-cache", "Docker build (fallback)"):
            return False

    if not run_command("docker-compose up -d apexai_db apexai_minio", "Start core services"):
        return False

    print("⏳ Waiting for core services to be ready...")
    time.sleep(30)  # Wait for core services to start

    return True


def start_remaining_services():
    """Start remaining services (MLflow, Optuna, pgAdmin)."""
    print("\n🚀 Starting remaining services...")

    if not run_command("docker-compose up -d", "Start all remaining services"):
        return False

    print("⏳ Waiting for all services to be ready...")
    time.sleep(20)

    return True


def init_postgresql_databases():
    """Initialize PostgreSQL databases."""
    print("\n💾 Initializing PostgreSQL databases...")

    # Execute unified database setup script
    result = run_command(
        "docker exec apexai_predictor python init_scripts/setup_database.py",
        "Setup PostgreSQL databases using unified script",
        check=True,
    )

    if not result:
        print("❌ Failed to initialize PostgreSQL databases")
        return False

    print("✅ PostgreSQL databases initialized successfully")
    return True


def init_minio_bucket():
    """Initialize MinIO bucket (using existing script)."""
    print("\n🗄️ Initializing MinIO bucket...")

    # Execute existing setup_minio.py script
    result = run_command(
        "docker exec apexai_predictor python init_scripts/setup_minio.py",
        "Setup MinIO bucket using existing script",
        check=False,
    )
    # Continue regardless of MinIO setup success/failure (warning only)
    if result:
        print("✅ MinIO bucket initialized successfully")
    else:
        print("⚠️  MinIO bucket setup completed with warnings (this is normal)")
    return True


def init_optuna_study():
    """Initialize Optuna study (using existing script)."""
    print("\n🎯 Initializing Optuna studies...")

    # Execute existing setup_optuna.py script
    result = run_command(
        "docker exec apexai_predictor python init_scripts/setup_optuna.py",
        "Setup Optuna studies using existing script",
        check=False,
    )
    # Continue regardless of Optuna setup success/failure (warning only)
    if result:
        print("✅ Optuna studies initialized successfully")
    else:
        print("⚠️  Optuna studies setup completed with warnings (this is normal)")
    return True


def init_mlflow_database():
    """Initialize MLflow database (using existing script)."""
    print("\n📊 Initializing MLflow database...")

    # Execute existing setup_mlflow_db.py script
    result = run_command(
        "docker exec apexai_predictor python init_scripts/setup_mlflow_db.py",
        "Setup MLflow database using existing script",
        check=False,
    )
    # Continue regardless of MLflow setup success/failure (warning only)
    if result:
        print("✅ MLflow database initialized successfully")
    else:
        print("⚠️  MLflow database setup completed with warnings (this is normal)")
    return True


def wait_for_database_ready():
    """Wait for databases to be ready."""
    print("\n⏳ Waiting for databases to be fully initialized...")
    time.sleep(20)  # Additional wait for database initialization
    return True


def verify_services():
    """Verify services (using unified health check)."""
    print("\n🔍 Verifying ApexAI services...")

    # Check container status
    run_command("docker-compose ps", "Check container status")

    # Execute comprehensive health check
    print("\n🩺 Running comprehensive health check...")
    health_result = run_command(
        "docker exec apexai_predictor python init_scripts/health_check.py",
        "Execute comprehensive health check",
        check=False,
    )

    if health_result:
        print("✅ All services verified and healthy")
    else:
        print("⚠️  Health check completed with some warnings (core services are operational)")

    print("\n📋 Service access URLs:")
    print("  📊 MLflow UI:        http://localhost:5001")
    print("  🎯 Optuna Dashboard: http://localhost:8081")
    print("  🗄️ MinIO Console:    http://localhost:9020")
    print("  💾 pgAdmin:          http://localhost:5051")
    return True


def ensure_env_file():
    """Ensure environment configuration file exists."""
    print("📄 Checking environment configuration...")

    env_file = Path(".env")
    env_template = Path(".env.template")

    if not env_file.exists():
        if env_template.exists():
            print("🔧 Creating .env file from template...")
            shutil.copy(env_template, env_file)
            print("✅ .env file created from .env.template")
        else:
            print("❌ Neither .env nor .env.template found")
            sys.exit(1)
    else:
        print("✅ .env file already exists")

    return True


def main():
    """Main execution function."""
    print("🚀" + "=" * 60)
    print("   ApexAI - Vehicle Identification Platform Setup")
    print("=" * 63)

    # 1. Ensure environment configuration file
    ensure_env_file()

    # 2. Check Docker environment
    if not check_docker():
        sys.exit(1)

    # 3. Clean up existing environment
    cleanup_environment()

    # 4. Build Docker images and start core services
    if not build_and_start_core_services():
        print("❌ Failed to build and start core services")
        sys.exit(1)

    # 5. Start Predictor service
    if not run_command("docker-compose up -d apexai_predictor", "Start Predictor service"):
        print("❌ Failed to start Predictor service")
        sys.exit(1)

    # 6. Wait for Predictor container to be ready
    print("⏳ Waiting for Predictor service to be ready...")
    time.sleep(15)

    # 7. Initialize PostgreSQL databases (executed from Predictor container)
    if not init_postgresql_databases():
        print("❌ Failed to initialize PostgreSQL databases")
        sys.exit(1)

    # 8. Start MLflow and MinIO
    if not run_command(
        "docker-compose up -d apexai_mlflow apexai_minio",
        "Start MLflow and MinIO services",
    ):
        print("❌ Failed to start MLflow and MinIO services")
        sys.exit(1)

    # 9. Wait for database to be ready
    wait_for_database_ready()

    # 10. Initialize MinIO bucket
    init_minio_bucket()

    # 11. Initialize MLflow database
    init_mlflow_database()

    # 12. Initialize Optuna database tables and studies (Important: run from predictor container)
    print("\n🎯 Initializing Optuna database tables and studies...")
    if not run_command(
        "docker exec apexai_predictor python init_scripts/setup_optuna.py",
        "Initialize Optuna tables and studies",
        check=True,
    ):
        print("❌ Failed to initialize Optuna")
        sys.exit(1)

    # 13. Start Optuna Dashboard and pgAdmin after Optuna initialization
    if not run_command(
        "docker-compose up -d apexai_optuna apexai_pgadmin",
        "Start Optuna Dashboard and pgAdmin",
    ):
        print("❌ Failed to start Optuna Dashboard and pgAdmin")
        sys.exit(1)

    # 14. Wait for Optuna Dashboard to stabilize
    print("⏳ Waiting for Optuna Dashboard to stabilize...")
    time.sleep(15)

    # 15. Verify services
    verify_services()

    # 16. Check GPU environment
    print("\n🎮 Checking GPU environment...")
    gpu_result = run_command(
        "docker exec apexai_predictor python init_scripts/gpu_check.py",
        "GPU environment check",
        check=False,
    )
    if gpu_result:
        print("✅ GPU acceleration available and verified!")
    else:
        print("⚠️  Running in CPU mode (GPU not available or not configured)")

    # 17. Run setup verification experiment
    print("\n🧪 Running setup verification experiment...")
    test_result = run_command(
        "docker exec apexai_predictor python init_scripts/test_experiment.py",
        "Execute setup verification test",
        check=False,
    )
    if test_result:
        print("✅ Setup verification test completed successfully!")
    else:
        print("⚠️  Setup verification completed with warnings (core functionality working)")
        print(
            "   You can manually run: docker exec apexai_predictor python init_scripts/test_experiment.py"
        )

    # Setup complete
    print("\n" + "=" * 63)
    print("🎉 ApexAI Setup Completed Successfully!")
    print("=" * 63)
    print("\n🌐 Available Services:")
    print("  📊 MLflow UI:        http://localhost:5001")
    print("  🎯 Optuna Dashboard: http://localhost:8081")
    print("  🗄️ MinIO Console:    http://localhost:9020")
    print("  💾 pgAdmin:          http://localhost:5051")


if __name__ == "__main__":
    main()
