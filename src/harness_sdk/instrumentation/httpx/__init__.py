'''Traceable wrapper around OpenTelemetry httpx instrumentation.'''
# Mirrors the requests instrumentor wrapper pattern (init + OTel hooks).
# pylint: disable=duplicate-code
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from harness_sdk.plugins.control import get_control_registry
from harness_sdk.instrumentation import BaseInstrumentorWrapper
from harness_sdk.instrumentation.httpx.utils import (
    headers_from_httpx,
    read_request_body,
    url_from_request_info,
)

from harness_sdk.custom_logger import get_custom_logger

logger = get_custom_logger(__name__)


class HTTPXClientInstrumentorWrapper(HTTPXClientInstrumentor, BaseInstrumentorWrapper):
    '''Traceable wrapper around OpenTelemetry httpx instrumentor class.'''

    def __init__(self):
        logger.debug('Entering HTTPXClientInstrumentorWrapper.__init__().')
        HTTPXClientInstrumentor.__init__(self)
        BaseInstrumentorWrapper.__init__(self)

    def _instrument(self, **kwargs) -> None:
        '''Enable instrumentation with request/response hooks.'''
        super()._instrument(
            tracer_provider=kwargs.get("tracer_provider"),
            request_hook=self.request_hook,
            response_hook=self.response_hook,
            async_request_hook=self.async_request_hook,
            async_response_hook=self.async_response_hook,
        )

    def _process_request(self, span, request_info):
        url = url_from_request_info(request_info)
        headers = headers_from_httpx(request_info.headers)
        body = read_request_body(request_info.stream)
        self.generic_request_handler(headers, body, span)
        get_control_registry().evaluate(span, url, headers, body, False)

    def _process_response(self, span, response_info):
        headers = headers_from_httpx(response_info.headers)
        self.generic_response_handler(headers, None, span)

    def request_hook(self, span, request_info):
        '''Capture sync client request data and run evaluation.'''
        self._process_request(span, request_info)

    def response_hook(self, span, request_info, response_info):  # pylint: disable=unused-argument
        '''Capture sync client response headers.'''
        self._process_response(span, response_info)

    async def async_request_hook(self, span, request_info):
        '''Capture async client request data and run evaluation.'''
        self._process_request(span, request_info)

    async def async_response_hook(self, span, request_info, response_info):  # pylint: disable=unused-argument
        '''Capture async client response headers.'''
        self._process_response(span, response_info)
