import logging

def setup_logging(log_level="INFO"):
    logging.basicConfig(level=log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

def format_duration(seconds: float) -> str:
    return f"{seconds:.2f}s"
