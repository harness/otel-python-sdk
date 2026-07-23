'''Base class of all Hypertrace Instrumentation Wrapper classes'''
import sys
import traceback
from opentelemetry.trace.span import Span

from harness_sdk.config.config import Config
from harness_sdk.custom_logger import get_custom_logger
logger = get_custom_logger(__name__)


# This is a base class for all Hypertrace Instrumentation wrapper classes
class BaseInstrumentorWrapper:
    '''This is a base class for all Hypertrace Instrumentation wrapper classes'''
    # Standard extended span attribute names / prefixes
    HTTP_REQUEST_HEADER_PREFIX = 'http.request.header.'
    HTTP_RESPONSE_HEADER_PREFIX = 'http.response.header.'
    HTTP_REQUEST_BODY_PREFIX = 'http.request.body'
    HTTP_RESPONSE_BODY_PREFIX = 'http.response.body'
    RPC_REQUEST_METADATA_PREFIX = 'rpc.request.metadata.'
    RPC_RESPONSE_METADATA_PREFIX = 'rpc.response.metadata.'
    RPC_REQUEST_BODY_PREFIX = 'rpc.request.body'
    RPC_RESPONSE_BODY_PREFIX = 'rpc.response.body'

    # Headers/metadata whose values are credentials or session secrets and must
    # never be exported verbatim in telemetry. Compared case-insensitively.
    SENSITIVE_HEADERS = frozenset({
        'authorization',
        'proxy-authorization',
        'x-api-key',
        'api-key',
        'apikey',
        'x-auth-token',
        'x-amz-security-token',
        'x-goog-api-key',
        'cookie',
        'set-cookie',
        'x-harness-service-token',
        'x-harness-token',
    })
    REDACTED_VALUE = '[REDACTED]'

    # Constructor
    def __init__(self):
        '''constructor'''
        logger.debug('Entering BaseInstrumentorWrapper constructor.')
        self._process_request_headers = Config()._instance.config.data_capture.http_headers.request.value
        self._process_response_headers = Config()._instance.config.data_capture.http_headers.response.value
        self._process_request_body = Config()._instance.config.data_capture.http_body.request.value
        self._process_response_body = Config()._instance.config.data_capture.http_body.response.value
        self._max_body_size = Config()._instance.config.data_capture.body_max_size_bytes.value

        proto_allowed_content_types = Config()._instance.config.data_capture.allowed_content_types
        self._allowed_content_types = [item.value for item in proto_allowed_content_types]
        super().__init__()

    # we need the headers lowercased multiple times
    # just do it once upfront
    def lowercase_headers(self, headers):
        '''convert all headers to lowercase'''
        return {k.lower(): v for k, v in headers.items()}

    def _redact_if_sensitive(self, header_key: str, header_value):
        '''Return the header value, redacted if the header is credential-bearing.'''
        if header_key.lower() in self.SENSITIVE_HEADERS:
            return self.REDACTED_VALUE
        return header_value

    def add_headers_to_span(self, prefix: str, span: Span, headers: dict):
        '''set header attributes on the span, redacting sensitive header values'''
        for header_key, header_value in headers.items():
            span.set_attribute(
                f"{prefix}{header_key}",
                self._redact_if_sensitive(header_key, header_value),
            )

    def eligible_based_on_content_type(self, headers: dict):
        '''find content-type in headers'''
        content_type = headers.get("content-type")
        if content_type is None:
            return False

        for configured_content_type in self._allowed_content_types:
            if configured_content_type in content_type:
                return True
        return False

    def _capture_headers(self, record_headers: bool, header_prefix: str,  # pylint:disable=R0913,R0917
                         span, headers: dict, record_body):
        try:  # pylint: disable=R1702
            if not span.is_recording():
                return False

            logger.debug('Span is Recording!')
            lowercased_headers = self.lowercase_headers(headers)
            if record_headers:
                self.add_headers_to_span(header_prefix, span, lowercased_headers)

            if record_body:
                content_type_recordable = self.eligible_based_on_content_type(lowercased_headers)
                return content_type_recordable
            return False
        except:  # pylint: disable=W0702
            logger.debug('An error occurred in _capture_headers: exception=%s, stacktrace=%s',
                         sys.exc_info()[0],
                         traceback.format_exc())
            return False

    def _generic_handler(self, record_headers: bool, header_prefix: str,  # pylint:disable=R0913,R0917
                         record_body: bool, body_prefix: str,
                         span: Span, headers: dict, body):
        logger.debug('Entering BaseInstrumentationWrapper.generic_handler().')
        try:  # pylint: disable=R1702
            if not span.is_recording():
                return span

            logger.debug('Span is Recording!')
            lowercased_headers = self.lowercase_headers(headers)
            if record_headers:
                self.add_headers_to_span(header_prefix, span, lowercased_headers)

            if record_body:
                content_type = self.eligible_based_on_content_type(lowercased_headers)

                if content_type is False:
                    return span

                body_str = None
                if isinstance(body, bytes):
                    body_str = body.decode('UTF8', 'backslashreplace')
                else:
                    body_str = body

                request_body_str = self.grab_first_n_bytes(body_str)
                span.set_attribute(body_prefix, request_body_str)

        except:  # pylint: disable=W0702
            logger.debug('An error occurred in genericRequestHandler: exception=%s, stacktrace=%s',
                         sys.exc_info()[0],
                         traceback.format_exc())
        return span

    # Generic HTTP Request Handler
    def generic_request_handler(self,  # pylint: disable=R0912
                                request_headers: dict,
                                request_body,
                                span: Span) -> Span:
        '''Add extended request data to the span'''
        logger.debug('Entering BaseInstrumentationWrapper.genericRequestHandler().')
        return self._generic_handler(self._process_request_headers, self.HTTP_REQUEST_HEADER_PREFIX,
                                     self._process_request_body, self.HTTP_REQUEST_BODY_PREFIX,
                                     span, request_headers, request_body)

    # Generic HTTP Response Handler
    def generic_response_handler(self,  # pylint: disable=R0912
                                 response_headers: dict,
                                 response_body,
                                 span: Span) -> Span:  # pylint: disable=R0912
        '''generic response handler'''
        logger.debug(
            'Entering BaseInstrumentationWrapper.genericResponseHandler().')
        return self._generic_handler(self._process_response_headers, self.HTTP_RESPONSE_HEADER_PREFIX,
                                     self._process_response_body, self.HTTP_RESPONSE_BODY_PREFIX,
                                     span, response_headers, response_body)

    # Generic RPC Request Handler
    def generic_rpc_request_handler(self,
                                    request_headers: dict,
                                    request_body,
                                    span: Span) -> Span:
        '''Add extended request rpc data to span.'''
        logger.debug(
            'Entering BaseInstrumentationWrapper.genericRpcRequestHandler().')
        try:
            # Is the span currently recording?
            if not span.is_recording():
                return span

            logger.debug('Span is Recording!')
            lowercased_headers = self.lowercase_headers(request_headers)

            # Log rpc metatdata if requested
            if self._process_request_headers:
                self.add_headers_to_span(self.RPC_REQUEST_METADATA_PREFIX, span, lowercased_headers)
            # Log rpc body if requested
            if self._process_response_body:
                request_body_str = str(request_body)
                request_body_str = self.grab_first_n_bytes(request_body_str)
                span.set_attribute(self.RPC_REQUEST_BODY_PREFIX,
                                   request_body_str)
        except:  # pylint: disable=W0702
            logger.debug('An error occurred in genericRequestHandler: exception=%s, stacktrace=%s',
                         sys.exc_info()[0],
                         traceback.format_exc())
            # Not rethrowing to avoid causing runtime errors
        return span

    # Generic RPC Response Handler
    def generic_rpc_response_handler(self,
                                     response_headers: dict,
                                     response_body,
                                     span: Span) -> Span:
        '''Add extended response rpc data to span'''
        logger.debug(
            'Entering BaseInstrumentationWrapper.genericRpcResponseHandler().')
        try:
            # is the span currently recording?
            if not span.is_recording():
                return span

            logger.debug('Span is Recording!')
            lowercased_headers = self.lowercase_headers(response_headers)
            # Log rpc metadata if requested?
            if self._process_response_headers:
                logger.debug('Dumping Response Headers:')
                self.add_headers_to_span(self.RPC_RESPONSE_METADATA_PREFIX, span, lowercased_headers)
            # Log rpc body if requested
            if self._process_response_body:
                response_body_str = str(response_body)
                logger.debug('Processing response body')
                response_body_str = self.grab_first_n_bytes(response_body_str)
                span.set_attribute(
                    self.RPC_RESPONSE_BODY_PREFIX, response_body_str)
        except:  # pylint: disable=W0702
            logger.debug('An error occurred in genericResponseHandler: exception=%s, stacktrace=%s',
                         sys.exc_info()[0],
                         traceback.format_exc())
            # Not rethrowing to avoid causing runtime errors
        return span

    # Check body size
    def check_body_size(self, body: str) -> bool:
        '''Is the size of this message body larger than the configured max?'''
        if body in (None, ''):
            return False
        body_len = len(body)
        max_body_size = self._max_body_size
        if max_body_size and body_len > max_body_size:
            logger.debug('message body size is greater than max size.')
            return True
        return False

    # grab first N bytes
    def grab_first_n_bytes(self, body: str) -> str:
        '''Return the first N (max_body_size) bytes of a request'''
        if body in (None, ''):
            return ''
        if self.check_body_size(body):  # pylint: disable=R1705
            return body[0 : self._max_body_size]
        return body
