import os
import subprocess
import sys
import time

import psycopg2
from psycopg2 import sql


def _make_db_url(
    user: str, pwd: str, host: str, port: int, db: str, extra: str = ""
) -> str:
    auth = f"{user}:{pwd}@" if pwd else f"{user}@"
    return f"postgresql+psycopg2://{auth}{host}:{port}/{db}{extra}"


def _check_remote_reachable(db_conf: dict) -> None:
    host = db_conf.get("host", "localhost")
    port = db_conf.get("port", 5432)
    user = db_conf.get("user", "optuna_user")
    pwd = os.getenv("PGPASSWORD", db_conf.get("password", ""))
    maintenance_db = db_conf.get("maintenance_db", "postgres")
    timeout = db_conf.get("connect_timeout", 5)
    try:
        conn = psycopg2.connect(
            dbname=maintenance_db,
            user=user,
            password=pwd,
            host=host,
            port=port,
            connect_timeout=timeout,
        )
        conn.close()
    except psycopg2.OperationalError as e:
        raise ConnectionError(f"Cannot reach remote Postgres at {host}:{port} ({e}).")


# ---------- LOCAL MODE ----------


def _check_postgres_running_local(config: dict) -> None:
    db_config = config.get("postgres_config", {})
    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    user = db_config.get("user", "optuna_user")
    pwd = os.getenv("PGPASSWORD", db_config.get("password", ""))
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user=user,
            password=pwd,
            host=host,
            port=port,
            connect_timeout=5,
        )
        conn.close()
    except psycopg2.OperationalError:
        raise Exception("PostgreSQL server is not running.")


def _start_postgres_server_local(config: dict) -> None:
    db_config = config.get("postgres_config", {})
    data_dir = db_config.get("data_dir", os.path.expanduser("~/postgres/data"))
    log_file = db_config.get("log_file", os.path.expanduser("~/postgres/logfile"))
    database_folder = db_config.get("database_folder", os.path.expanduser("~/postgres"))
    pg_ctl_path = os.path.join(database_folder, "bin", "pg_ctl")

    if not os.path.exists(data_dir):
        raise Exception(f"PostgreSQL data directory '{data_dir}' does not exist.")
    if not os.path.exists(pg_ctl_path):
        raise Exception(f"pg_ctl not found at '{pg_ctl_path}'.")

    try:
        print("Starting PostgreSQL server...")
        subprocess.run(
            [pg_ctl_path, "-D", data_dir, "-l", log_file, "start"], check=True
        )
        print("PostgreSQL server started successfully.")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to start PostgreSQL server: {e}")


def _initialize_postgres_local(config: dict, study_folder_name: str) -> str:
    db_config = config.get("postgres_config", {})
    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    user = db_config.get("user", "optuna_user")
    pwd = os.getenv("PGPASSWORD", db_config.get("password", ""))
    db_name = db_config.get("db_name", study_folder_name)

    try:
        conn = psycopg2.connect(
            dbname="postgres", user=user, password=pwd, host=host, port=port
        )
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", [db_name])
        if cur.fetchone():
            print(f"Database '{db_name}' already exists.")
            while True:
                choice = (
                    input(f"Database '{db_name}' exists. (U)se or (O)verwrite? [U/O]: ")
                    .strip()
                    .lower()
                )
                if choice == "u":
                    print(f"Using existing database '{db_name}'.")
                    break
                if choice == "o":
                    print(f"Overwriting database '{db_name}'.")
                    cur.execute(
                        """
                        SELECT pg_terminate_backend(pg_stat_activity.pid)
                        FROM pg_stat_activity
                        WHERE pg_stat_activity.datname = %s
                          AND pid <> pg_backend_pid();
                        """,
                        [db_name],
                    )
                    cur.execute(
                        sql.SQL("DROP DATABASE {}").format(sql.Identifier(db_name))
                    )
                    cur.execute(
                        sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
                    )
                    print(f"Database '{db_name}' recreated.")
                    break
                print("Invalid choice. Enter 'U' or 'O'.")
        else:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
            print(f"Created PostgreSQL database '{db_name}'.")
        cur.close()
        conn.close()
    except psycopg2.Error as e:
        print(f"Error initializing PostgreSQL: {e}")
        raise

    return _make_db_url(user, pwd, host, port, db_name)


# ---------- REMOTE MODE ----------


def _initialize_postgres_remote(config: dict, study_folder_name: str) -> str:
    db_config = config.get("postgres_config", {})
    host = db_config["host"]
    port = db_config.get("port", 5432)
    user = db_config.get("user", "optuna_user")
    pwd = os.getenv("PGPASSWORD", db_config.get("password", ""))
    db_name = db_config.get("db_name", "optuna_global")
    sslmode = db_config.get("sslmode")

    _check_remote_reachable(db_config)

    extra = f"?sslmode={sslmode}" if sslmode else ""
    return _make_db_url(user, pwd, host, port, db_name, extra)


def initialize_optuna_database(config: dict, study_folder_name: str) -> str:
    mode = config.get("postgres_config", {}).get("mode", "local").lower()
    if mode == "local":
        try:
            _check_postgres_running_local(config)
            print("PostgreSQL server is running.")
        except Exception as e:
            print(e)
            print("Attempting to start PostgreSQL server...")
            try:
                _start_postgres_server_local(config)
                time.sleep(2)
                _check_postgres_running_local(config)
                print("PostgreSQL server is now running.")
            except Exception as e_start:
                print(f"Failed to start PostgreSQL server: {e_start}")
                sys.exit(1)
        return _initialize_postgres_local(config, study_folder_name)

    if mode == "remote":
        try:
            return _initialize_postgres_remote(config, study_folder_name)
        except Exception as e:
            print(e)
            sys.exit(1)

    raise ValueError(f"Unknown postgres_config.mode '{mode}'. Use 'local' or 'remote'.")
