'''Hypertrace wrapper around OTel botocore Instrumentor''' # pylint: disable=R0801
import logging
from opentelemetry.instrumentation.botocore import BotocoreInstrumentor # pylint:disable=E0611,E0401
from agent_trace.instrumentation import BaseInstrumentorWrapper


from agent_trace.custom_logger import get_custom_logger
logger = get_custom_logger(__name__)

class BotocoreInstrumentationWrapper(BotocoreInstrumentor, BaseInstrumentorWrapper):
    '''Hypertrace wrapper around OTel Botocore Instrumentor class'''
