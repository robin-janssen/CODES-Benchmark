import shutil
from pathlib import Path

import optuna


def yes_no(prompt: str, default: bool = False) -> bool:
    """Simple Y/N prompt. default=False -> [y/N], True -> [Y/n]."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        ans = input(prompt + suffix).strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")


def copy_config(src: Path, dst_folder: Path) -> None:
    dst = dst_folder / "optuna_config.yaml"
    shutil.copy2(src, dst)


def build_study_names(config: dict, main_study_name: str) -> list[str]:
    if "surrogates" in config:
        return [f"{main_study_name}_{s['name'].lower()}" for s in config["surrogates"]]
    return [main_study_name]


def prepare_workspace(master_cfg_path: Path, config: dict) -> None:
    tuning_id = config["tuning_id"]
    run_dir = Path("tuned") / tuning_id
    run_dir_exists = run_dir.exists()

    if run_dir_exists:
        overwrite = yes_no(f"Folder '{run_dir}' exists. Overwrite?", default=False)
        if overwrite:
            shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            copy_config(master_cfg_path, run_dir)
            config["_overwrite_run"] = True
        else:
            config["_overwrite_run"] = False
            # ensure config is present for reproducibility (copy if missing)
            if not (run_dir / "optuna_config.yaml").exists():
                copy_config(master_cfg_path, run_dir)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        copy_config(master_cfg_path, run_dir)
        config["_overwrite_run"] = True  # brand new


def delete_studies_if_requested(config: dict, main_study_name: str, db_url: str):
    """Delete studies if user overwrote run folder."""
    if not config.get("_overwrite_run", False):
        return
    mode = config.get("postgres_config", {}).get("mode", "local").lower()
    if mode == "local":
        return  # no need to delete studies in local mode, DB is handled later in this mode (overwrite or use existing)
    names = build_study_names(config, main_study_name)
    for name in names:
        try:
            optuna.delete_study(name=name, storage=db_url)
            print(f"Deleted existing study '{name}'.")
        except KeyError or TypeError:
            # study didn't exist
            pass
