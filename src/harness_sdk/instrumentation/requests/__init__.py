'''Hypertrace wrapper around OTel instrumentation class'''
import logging
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from harness_sdk.plugins.control import get_control_registry
from harness_sdk.instrumentation import BaseInstrumentorWrapper

from harness_sdk.custom_logger import get_custom_logger
logger = get_custom_logger(__name__)

class RequestsInstrumentorWrapper(RequestsInstrumentor, BaseInstrumentorWrapper):
    '''Hypertrace wrapper around OTel requests instrumentor class'''
    # Constructor
    def __init__(self):
        logger.debug('Entering RequestsInstrumentorWrapper.__init__().')
        RequestsInstrumentor.__init__(self)
        BaseInstrumentorWrapper.__init__(self)

    def _instrument(self, **kwargs) -> None:
        '''internal enable instrumentation'''
        super()._instrument(
            tracer_provider=kwargs.get("tracer_provider"),
            request_hook=self.request_hook,
            response_hook=self.response_hook,
        )

    def request_hook(self, span, request_obj):
        '''capture request data'''
        url = request_obj.url
        self.generic_request_handler(request_obj.headers, request_obj.body, span)
        get_control_registry().evaluate(span, url, request_obj.headers, request_obj.body, False)


    def response_hook(self, span, _, response):
        '''capture response data'''
        self.generic_response_handler(response.headers, response.text, span)

    def _uninstrument(self, **kwargs) -> None:
        '''internal disable instrumentation'''
        super()._uninstrument()
