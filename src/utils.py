import mlflow
import yaml


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def setup_mlflow():
    config = load_config()
    mlflow.set_tracking_uri(config["logging"]["mlflow_tracking_uri"])
    mlflow.set_experiment("Financial_Analyzer")
