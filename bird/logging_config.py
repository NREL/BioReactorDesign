import logging
from pathlib import Path


class ConditionalFormatter(logging.Formatter):
    def format(self, record):
        """Change how to log if it is an INFO or a WARNING/ERROR"""
        if record.levelno >= logging.WARNING:
            self._style._fmt = "%(asctime)s [%(levelname)s] bird: %(message)s"
        else:
            self._style._fmt = "%(asctime)s [%(levelname)s] bird: %(message)s"
        return super().format(record)


def setup_logging(logfile: str | None = None, level=logging.INFO):
    """Setup logging"""
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = ConditionalFormatter()

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler
    if logfile:
        fh = logging.FileHandler(Path(logfile))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
