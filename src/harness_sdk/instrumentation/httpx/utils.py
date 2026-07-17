'''Helpers for extracting httpx request/response data for tracing.'''

from harness_sdk.custom_logger import get_custom_logger

logger = get_custom_logger(__name__)


def headers_from_httpx(headers):
    '''Convert httpx.Headers to a plain dict.'''
    if headers is None:
        return {}
    return dict(headers)


def url_from_request_info(request_info):
    '''Return the request URL as a string.'''
    return str(request_info.url)


def _body_from_byte_stream(stream):
    internal = getattr(stream, "_stream", None)
    if isinstance(internal, (bytes, bytearray)):
        return bytes(internal)
    return None


def read_request_body(stream):
    '''Read request body bytes without breaking replay.'''
    if stream is None:
        return None

    body = _body_from_byte_stream(stream)
    if body is not None:
        return body

    try:
        return b"".join(stream)
    except Exception:  # pylint: disable=broad-except
        logger.debug("Unable to read httpx request stream body", exc_info=True)
        return None
