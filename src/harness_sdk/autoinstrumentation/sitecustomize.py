import os

import psutil
from harness_sdk.agent import Agent # pylint:disable=C0413

from harness_sdk.config.config import Config  # pylint:disable=C0413,C0412
from harness_sdk.custom_logger import get_custom_logger  # pylint:disable=C0413,C0412,C0411

config = Config()
logger = get_custom_logger(__name__)

a = Agent()
a.instrument(None, None, auto_instrument=True)

__POST_INIT = False
POST_FORK_SERVERS = ['gunicorn']

original_process = psutil.Process(os.getpid())
args = original_process.cmdline()

for entry in POST_FORK_SERVERS:
    for arg in args:
        if entry in arg:
            __POST_INIT = True
            logger.info('Detected server %s - deferring filter loading until post fork', entry)
            break


if __POST_INIT is not True:
    logger.info("Control plugins loaded during autoinstrumentation agent init")
