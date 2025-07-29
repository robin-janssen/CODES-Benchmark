import subprocess
import getpass
import pytest
import psycopg2
from psycopg2 import OperationalError

# Import all functions to test
from codes.tune import (
    _make_db_url,
    _check_remote_reachable,
    _check_postgres_running_local,
    _start_postgres_server_local,
    _initialize_postgres_local,
    _initialize_postgres_remote,
    initialize_optuna_database,
)


# --- _make_db_url tests ---
def test_make_db_url_with_password():
    url = _make_db_url("user", "pass", "host", 5432, "db", "?sslmode=require")
    assert url == "postgresql+psycopg2://user:pass@host:5432/db?sslmode=require"


def test_make_db_url_without_password():
    url = _make_db_url("user", "", "localhost", 5433, "test", "")
    assert url == "postgresql+psycopg2://user@localhost:5433/test"


# --- _check_remote_reachable tests ---
def test_check_remote_reachable_success(monkeypatch):
    # monkeypatch psycopg2.connect to succeed
    monkeypatch.setattr(
        psycopg2,
        "connect",
        lambda **kw: type("Conn", (), {"close": lambda self: None})(),
    )
    # should not raise
    _check_remote_reachable({"host": "h", "port": 1111, "user": "u", "password": "p"})


def test_check_remote_reachable_failure(monkeypatch):
    # monkeypatch connect to raise OperationalError
    def bad_connect(**kw):
        raise OperationalError("cant connect")

    monkeypatch.setattr(psycopg2, "connect", bad_connect)
    with pytest.raises(ConnectionError) as exc:
        _check_remote_reachable({"host": "h", "port": 2222})
    assert "Cannot reach remote Postgres" in str(exc.value)


# --- _check_postgres_running_local tests ---
def test_check_postgres_running_local_success(monkeypatch):
    monkeypatch.setattr(
        psycopg2,
        "connect",
        lambda **kw: type("Conn", (), {"close": lambda self: None})(),
    )
    _check_postgres_running_local({"postgres_config": {"host": "h", "port": 5432}})


def test_check_postgres_running_local_failure(monkeypatch):
    monkeypatch.setattr(
        psycopg2, "connect", lambda **kw: (_ for _ in ()).throw(OperationalError())
    )
    with pytest.raises(Exception) as exc:
        _check_postgres_running_local({"postgres_config": {}})
    assert "PostgreSQL server is not running" in str(exc.value)


# --- _start_postgres_server_local tests ---
def test_start_postgres_server_local_missing_data_dir(tmp_path):
    cfg = {
        "postgres_config": {
            "data_dir": str(tmp_path / "nope"),
            "database_folder": str(tmp_path),
            "log_file": "log",
        }
    }
    with pytest.raises(Exception) as exc:
        _start_postgres_server_local(cfg)
    assert "data directory" in str(exc.value)


def test_start_postgres_server_local_missing_pg_ctl(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = {
        "postgres_config": {
            "data_dir": str(data_dir),
            "database_folder": str(tmp_path),
            "log_file": "log",
        }
    }
    with pytest.raises(Exception) as exc:
        _start_postgres_server_local(cfg)
    assert "pg_ctl not found" in str(exc.value)


def test_start_postgres_server_local_success(tmp_path, monkeypatch, capsys):
    # create data_dir and pg_ctl
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pg_ctl = bin_dir / "pg_ctl"
    pg_ctl.write_text("")
    cfg = {
        "postgres_config": {
            "data_dir": str(data_dir),
            "database_folder": str(tmp_path),
            "log_file": "lf",
        }
    }
    # monkeypatch subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *args, **kw: None)
    _start_postgres_server_local(cfg)
    captured = capsys.readouterr()
    assert "Starting PostgreSQL server" in captured.out
    assert "started successfully" in captured.out


# --- _initialize_postgres_local tests ---
class FakeCursor:
    def __init__(self, exists):
        self.exists = exists

    def execute(self, q, params=None):
        pass

    def fetchone(self):
        return (1,) if self.exists else None

    def close(self):
        pass


class FakeConn:
    def __init__(self, exists):
        self.exists = exists
        self.autocommit = False

    def cursor(self):
        return FakeCursor(self.exists)

    def close(self):
        pass


@pytest.mark.parametrize("exists,choice", [(False, None), (True, "u"), (True, "o")])
def test_initialize_postgres_local(monkeypatch, tmp_path, exists, choice):
    # monkeypatch connect
    def fake_connect(**kw):
        return FakeConn(exists)

    monkeypatch.setattr(psycopg2, "connect", fake_connect)
    # monkeypatch input when db exists
    if exists and choice:
        monkeypatch.setattr("builtins.input", lambda prompt: choice)
    cfg = {
        "postgres_config": {
            "host": "h",
            "port": 5432,
            "user": "u",
            "password": "p",
            "data_dir": "d",
            "database_folder": "x",
            "log_file": "l",
        }
    }
    url = _initialize_postgres_local(cfg, "mydb")
    assert "postgresql+psycopg2://" in url


# --- _initialize_postgres_remote tests ---
def test_initialize_postgres_remote_interactive(monkeypatch):
    cfg = {
        "postgres_config": {
            "mode": "remote",
            "host": "rh",
            "port": 1234,
            "user": "ru",
            "password": None,
            "db_name": "dbn",
            "sslmode": "require",
        }
    }
    # ensure no env var
    monkeypatch.delenv("PGPASSWORD", raising=False)
    # patch getpass
    monkeypatch.setattr(getpass, "getpass", lambda prompt: "pwd")
    # patch reachability
    monkeypatch.setattr(
        "codes.tune.postgres_fcts._check_remote_reachable", lambda conf: None
    )
    url = _initialize_postgres_remote(cfg, "ignored")
    assert "sslmode=require" in url


# --- initialize_optuna_database tests ---
def test_initialize_optuna_database_local(monkeypatch):
    cfg = {"postgres_config": {"mode": "local"}}
    monkeypatch.setattr(
        "codes.tune.postgres_fcts._check_postgres_running_local", lambda c: None
    )
    monkeypatch.setattr(
        "codes.tune.postgres_fcts._initialize_postgres_local", lambda c, n: "URL_LOCAL"
    )
    val = initialize_optuna_database(cfg, "sf")
    assert val == "URL_LOCAL"


def test_initialize_optuna_database_remote(monkeypatch):
    cfg = {"postgres_config": {"mode": "remote"}}
    monkeypatch.setattr(
        "codes.tune.postgres_fcts._initialize_postgres_remote", lambda c, n: "URL_REM"
    )
    val = initialize_optuna_database(cfg, "sf")
    assert val == "URL_REM"


def test_initialize_optuna_database_bad_mode():
    with pytest.raises(ValueError):
        initialize_optuna_database({"postgres_config": {"mode": "foo"}}, "sf")
