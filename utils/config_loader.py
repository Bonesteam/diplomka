import yaml, os

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs(config):
    for key in ["saved_models", "results"]:
        os.makedirs(config["paths"][key], exist_ok=True)
