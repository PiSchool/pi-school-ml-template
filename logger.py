import logging
from logging.config import dictConfig


def getLogger():
    logging_config = dict(
        version=1,
        formatters={
            'f': {'format':
                  '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}
        },
        handlers={
            'h': {'class': 'logging.StreamHandler',
                  'formatter': 'f',
                  'level': logging.INFO}
        },
        root={
            'handlers': ['h'],
            'level': logging.INFO,
        },
    )

    dictConfig(logging_config)
    return logging.getLogger()
