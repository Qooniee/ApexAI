#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE optuna_db'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'optuna_db')\gexec
    GRANT ALL PRIVILEGES ON DATABASE optuna_db TO $POSTGRES_USER;
EOSQL
