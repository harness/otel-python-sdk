import os

import psutil
from agent_trace.agent import Agent # pylint:disable=C0413

from agent_trace.config.config import Config  # pylint:disable=C0413,C0412
from agent_trace.custom_logger import get_custom_logger  # pylint:disable=C0413,C0412,C0411

config = Config()
logger = get_custom_logger(__name__)

# if using AWS_LAMBDA_EXEC_WRAPPER we will invoke instrumentation here
# and skip it in our wrapper handler
# otherwise(on py3.6 + 3.7) instrumentation + processor registration will occur within our wrapper handler
a = Agent()
if "_HANDLER" in os.environ:
    a.is_lambda = True

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
