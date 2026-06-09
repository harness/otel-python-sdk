'''Hypertrace django instrumentor module wrapper.''' # pylint: disable=R0401
import logging
import traceback

from django.conf import settings  # pylint:disable=C0415
from django.http import HttpResponse  # pylint:disable=C0415
from opentelemetry.instrumentation.django import DjangoInstrumentor
from opentelemetry.instrumentation.django.middleware.otel_middleware import (
    _DjangoMiddleware,
)
from opentelemetry.instrumentation.wsgi import collect_request_attributes
from opentelemetry.semconv._incubating.attributes.http_attributes import HTTP_TARGET
from opentelemetry.trace import Span

from harness_sdk import constants
from harness_sdk.plugins.control import ControlResult, get_control_registry
from harness_sdk.instrumentation import BaseInstrumentorWrapper

from harness_sdk.custom_logger import get_custom_logger
logger = get_custom_logger(__name__)

_IS_BLOCKED_ATTR = '_is_blocked'
_BLOCKING_MIDDLEWARE_PATH = 'harness_sdk.instrumentation.django.BlockingMiddleware'


class BlockingMiddleware:
    """Return a block response before the view runs (OTel swallows request_hook errors)."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        control_result = getattr(request, _IS_BLOCKED_ATTR, None)
        if control_result is not None:
            return HttpResponse(
                control_result.response_message,
                status=control_result.response_status_code,
                headers={},
            )
        return self.get_response(request)


def _install_blocking_middleware():
    if _BLOCKING_MIDDLEWARE_PATH in settings.MIDDLEWARE:
        return
    otel_index = next(
        (
            index
            for index, entry in enumerate(settings.MIDDLEWARE)
            if 'opentelemetry' in entry.lower()
        ),
        None,
    )
    if otel_index is not None:
        settings.MIDDLEWARE.insert(otel_index + 1, _BLOCKING_MIDDLEWARE_PATH)
    else:
        settings.MIDDLEWARE.insert(0, _BLOCKING_MIDDLEWARE_PATH)


def _wsgi_environ_with_request_uri(request):
    """WSGI environ with REQUEST_URI when the server omitted it (e.g. Django runserver)."""
    environ = request.META
    if environ.get("RAW_URI") or environ.get("REQUEST_URI"):
        return environ
    path = environ.get("PATH_INFO") or request.path
    query_string = environ.get("QUERY_STRING", "")
    request_uri = f"{path}?{query_string}" if query_string else path
    return {**environ, "REQUEST_URI": request_uri}


def _apply_missing_wsgi_request_attributes(span: Span, request) -> None:
    """Backfill span attrs using OTel WSGI collection (same path Flask/Werkzeug gets)."""
    if not span.is_recording():
        return
    if span.attributes.get(HTTP_TARGET):
        return
    collected = collect_request_attributes(
        _wsgi_environ_with_request_uri(request),
        sem_conv_opt_in_mode=_DjangoMiddleware._sem_conv_opt_in_mode,
    )
    for key, value in collected.items():
        if value is not None and span.attributes.get(key) is None:
            span.set_attribute(key, value)


class DjangoInstrumentationWrapper(BaseInstrumentorWrapper):
    """wrapped class around django instrumentation"""
    def instrument(self):
        """configure django instrumentor w hooks"""
        DjangoInstrumentor().instrument(request_hook=self.request_hook,
                                        response_hook=self.response_hook)
        _install_blocking_middleware()

    def uninstrument(self):
        """need this to match wrapper interface for specs"""
        return

    def request_hook(self, span: Span, request):
        """django request hook before request is processed by app"""
        try:
            body = request.body
            self.generic_request_handler(request.headers, body, span)
            _apply_missing_wsgi_request_attributes(span, request)
            full_url = request.build_absolute_uri()
            control_result = get_control_registry().evaluate(span,
                                                    full_url,
                                                    request.headers,
                                                    body,
                                                    False)
            if control_result.block:
                self._apply_block(request, span, control_result)
        except Exception as err:  # pylint:disable=W0703
            logger.debug(constants.INST_RUNTIME_EXCEPTION_MSSG,
                         'django request hook',
                         err,
                         traceback.format_exc())

    @staticmethod
    def _apply_block(request, span: Span, control_result: ControlResult) -> None:
        logger.debug('should block evaluated to true, aborting')
        span.set_attribute('http.status_code', control_result.response_status_code)
        span.end()
        setattr(request, _IS_BLOCKED_ATTR, control_result)

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
