# utils/logging_utils.py
import logging
import os
from rich.logging import RichHandler

def setup_logging(log_dir: str, log_name: str = "pipeline.log", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_name)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            RichHandler(rich_tracebacks=True, markup=True, show_time=False, show_path=False)
        ]
    )
