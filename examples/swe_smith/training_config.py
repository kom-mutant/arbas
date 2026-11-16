"""Load a SWE-smith training config, optionally based on one shared YAML file."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

BASE_CONFIG = Path(__file__).resolve().with_name("training-base.yaml")


def load_training_config(path: Path) -> DictConfig:
    path = path.expanduser().resolve()
    experiment = OmegaConf.load(path)
    base_file = experiment.pop("base", None)
    if base_file is None:
        config = experiment
        base_path = path
    else:
        base_path = (path.parent / Path(base_file).expanduser()).resolve()
        base = OmegaConf.load(base_path)
        if "base" in base:
            raise ValueError(f"Base config {base_path} cannot include another base config.")
        config = OmegaConf.merge(base, experiment)

    if base_path != BASE_CONFIG:
        defaults = OmegaConf.masked_copy(OmegaConf.load(BASE_CONFIG), ["agents", "experiment", "ray_kwargs"])
        config = OmegaConf.merge(defaults, config)

    experiment_data = experiment.get("data", {})
    data = config.get("data", {})
    for field in ("train_files", "val_files"):
        if field in data:
            directory = path.parent if field in experiment_data else base_path.parent
            data[field] = str(directory / Path(data[field]).expanduser())
    return config
