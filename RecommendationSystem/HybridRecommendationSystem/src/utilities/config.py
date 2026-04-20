import os
import yaml
from pathlib import Path

# ----------------------------
# Environment detection
# ----------------------------
def running_on_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

# Determine project root
project_root = Path.cwd().parent if running_on_databricks() else Path.cwd()

# ----------------------------
# Load YAML configs
# ----------------------------
with open(project_root / "config.yaml") as f:
    yml_config = yaml.safe_load(f)

with open(project_root / "src/paths.yaml") as f:
    paths_yaml = yaml.safe_load(f)

# ----------------------------
# Config class
# ----------------------------
class Config:
    def __init__(self):
        # 1️⃣ Load config.yaml variables as attributes
        for k, v in yml_config.items():
            setattr(self, k.lower(), v)

        # 2️⃣ Resolve paths.yaml placeholders
        self.resolved_paths = {}
        for key, path in paths_yaml.items():
            # Replace placeholders from config.yaml
            val = path.format(**{k.upper(): getattr(self, k.lower()) for k in yml_config.keys()})
            # Replace placeholders referencing other resolved paths
            for pk, pv in self.resolved_paths.items():
                val = val.replace(f"{{{pk.upper()}}}", pv)
            self.resolved_paths[key.lower()] = val
            setattr(self, key.lower(), val)

        # 3️⃣ Environment flag
        self.running_on_databricks = running_on_databricks()

# ----------------------------
# Instantiate single config object
# ----------------------------
config = Config()
