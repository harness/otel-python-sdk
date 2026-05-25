'''constants'''
from agent_trace import version

EXCEPTION_MESSAGE = 'Failed to initialize %s instrumentation wrapper: exception=%s, stacktrace=%s'
INST_WRAP_EXCEPTION_MSSG = 'Failed to initialize %s instrumentation wrapper: exception=%s, stacktrace=%s' # pylint: disable=C0301
INST_RUNTIME_EXCEPTION_MSSG = 'An error occurred in %s: exception=%s, stacktrace=%s'
TELEMETRY_SDK_VERSION = version.__version__
TELEMETRY_SDK_NAME = 'agent_trace'
TELEMETRY_SDK_LANGUAGE = 'python'
