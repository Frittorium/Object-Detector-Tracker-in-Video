import logging
import sys

from applog import initialize_logging
from application import Application

logger = logging.getLogger(__name__)


def main() -> int:
    initialize_logging()
    try:
        return Application().run()
    except Exception:
        logger.exception("Unhandled application exception")
        return 1


if __name__ == "__main__":
    sys.exit(main())