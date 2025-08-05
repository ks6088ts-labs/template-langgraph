import logging


def get_logger(
    name: str = "default",
    verbosity: int = logging.INFO,
) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name (str): The name of the logger.
        verbosity (int): The logging level (default: logging.INFO).
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(verbosity)
    formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
