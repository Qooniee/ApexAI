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
    # 環境変数から設定を取得（デフォルト値なし、必須）
    host = host or os.getenv("DB_HOST")
    port = port or int(os.getenv("DB_PORT", "5432"))
    user = user or os.getenv("POSTGRES_USER")
    password = password or os.getenv("POSTGRES_PASSWORD")
    database = database or os.getenv("POSTGRES_DB_MLFLOW")

    if not all([host, user, password, database]):
        print("❌ 必須環境変数が設定されていません")
        print("必要な環境変数: DB_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB_MLFLOW")
        return False

    print(f"PostgreSQL ({database}) の起動を待機中...")
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(
                host=host, port=port, user=user, password=password, database=database
            )
            conn.close()
            print(f"PostgreSQL ({database}) 接続成功！")
            return True
        except psycopg2.OperationalError as e:
            print(f"接続試行 {attempt + 1}/{max_attempts}... ({e})")
            time.sleep(2)

    print(f"❌ PostgreSQL ({database}) への接続に失敗しました")
    return False


def verify_databases():
    """ApexAI データベースの接続確認."""
    print("=== ApexAI データベース接続確認 ===")

    # MLflowデータベース確認
    mlflow_db = os.getenv("POSTGRES_DB_MLFLOW")
    if not mlflow_db:
        print("❌ POSTGRES_DB_MLFLOW環境変数が設定されていません")
        return False

    if wait_for_postgres(database=mlflow_db):
        print("✅ MLflowデータベース接続OK")
    else:
        print("❌ MLflowデータベース接続失敗")
        return False

    # Optunaデータベース確認
    optuna_db = os.getenv("POSTGRES_DB_OPTUNA")
    if not optuna_db:
        print("❌ POSTGRES_DB_OPTUNA環境変数が設定されていません")
        return False

    if wait_for_postgres(database=optuna_db):
        print("✅ Optunaデータベース接続OK")
    else:
        print("❌ Optunaデータベース接続失敗")
        return False

    print("=== 環境変数確認 ===")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "未設定")
    db_host = os.getenv("DB_HOST", "未設定")
    db_user = os.getenv("POSTGRES_USER", "未設定")

    print(f"MLFLOW_TRACKING_URI: {mlflow_uri}")
    print(f"DB_HOST: {db_host}")
    print(f"POSTGRES_USER: {db_user}")

    return True


if __name__ == "__main__":
    if verify_databases():
        print("\n✅ ApexAI データベース起動確認完了！")
        print("\n🌐 利用可能なサービス:")
        print("  📊 MLflow UI:        http://localhost:5001")
        print("  🎯 Optuna Dashboard: http://localhost:8081")
        print("  🗄️ MinIO Console:    http://localhost:9020")
        print("  💾 pgAdmin:          http://localhost:5051")
        print("\n🚀 実験実行例:")
        print(
            "  docker exec apexai_predictor bash -c \"cd /workspace && python src/apexai_main.py experiment.name='ApexAI_Test'\""
        )
    else:
        print("\n❌ ApexAI データベース起動確認に失敗しました")
        print("💡 対処法: setup_apexai.pyを実行してデータベースを初期化してください")
        exit(1)
