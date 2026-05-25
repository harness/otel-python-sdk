'''Hypertrace django instrumentor module wrapper.''' # pylint: disable=R0401
import logging
import traceback
from http.client import HTTPException
from types import MethodType

from opentelemetry.instrumentation.django import DjangoInstrumentor
from opentelemetry.trace import Span
from django.core.exceptions import PermissionDenied  # pylint:disable=C0415

from agent_trace import constants
from agent_trace.plugins.control import get_control_registry
from agent_trace.instrumentation import BaseInstrumentorWrapper

from agent_trace.custom_logger import get_custom_logger
logger = get_custom_logger(__name__)


class TraceablePermissionDenied(PermissionDenied):
    def __init__(self, message='', status_code=403, headers=None):
        self.message = message
        self.status_code = status_code
        self.headers = headers or {}
        super().__init__(message)

class DjangoInstrumentationWrapper(BaseInstrumentorWrapper):
    """wrapped class around django instrumentation"""
    def instrument(self):
        """configure django instrumentor w hooks"""
        DjangoInstrumentor().instrument(request_hook=self.request_hook,
                                        response_hook=self.response_hook)

    def uninstrument(self):
        """need this to match wrapper interface for specs"""
        return

    def request_hook(self, span: Span, request):
        """django request hook before request is processed by app"""
        try:
            body = request.body
            self.generic_request_handler(request.headers, body, span)
            full_url = request.build_absolute_uri()
            control_result = get_control_registry().evaluate(span,
                                                    full_url,
                                                    request.headers,
                                                    body,
                                                    False)
            if control_result.block:
                logger.debug('should block evaluated to true, aborting')
                status_code = control_result.response_status_code or 403
                # since middleware chain is halted the status code is not set when blocked
                span.set_attribute('http.status_code', status_code)
                span.end()
                raise TraceablePermissionDenied(
                    message=control_result.response_message or 'Permission denied',
                    status_code=status_code,
                    headers={}
                )
        except TraceablePermissionDenied as block_exception:
            raise block_exception
        except Exception as err:  # pylint:disable=W0703
            logger.debug(constants.INST_RUNTIME_EXCEPTION_MSSG,
                         'django request hook',
                         err,
                         traceback.format_exc())


    def response_hook(self, span, _request, response):
        """django response hook before response is written out"""
        try:
            body = response.content
            self.generic_response_handler(response.headers, body, span)
        except Exception as err:  # pylint:disable=W0703
            logger.debug(constants.INST_RUNTIME_EXCEPTION_MSSG,
                         'django response hook',
                         err,
                         traceback.format_exc())
