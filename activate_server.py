import datasets
import uvicorn
import yaml
from transformers.utils import logging

from src.create_api import CreateAPI
from src.download_n_prepare_models import PreparePipeline

# Disable huggingface progress bar
datasets.disable_progress_bar()
logging.disable_progress_bar()

# Load project settings
with open("setup.yaml", "r") as file:
    config = yaml.load(file, yaml.Loader)

# Download local LLM to be used in various pipelines
initiator = PreparePipeline()
initiator.run()

# Create API
app = CreateAPI().get_api()

# Activate server
uvicorn.run(
    app,
    host=config["SERVER"]["HOST"],
    port=config["SERVER"]["PORT"],
    log_level="info"
)
