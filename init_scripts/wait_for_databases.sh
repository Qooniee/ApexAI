#!/bin/bash
set -e

host="$1"
shift
cmd="$@"

until psql -h "$host" -U "$POSTGRES_USER" -d "$POSTGRES_DB_MLFLOW" -c '\q'; do
  >&2 echo "MLflow database is unavailable - sleeping"
  sleep 1
done

until psql -h "$host" -U "$POSTGRES_USER" -d "$POSTGRES_DB_OPTUNA" -c '\q'; do
  >&2 echo "Optuna database is unavailable - sleeping"
  sleep 1
done

>&2 echo "Both databases are up - executing command"
exec $cmd
