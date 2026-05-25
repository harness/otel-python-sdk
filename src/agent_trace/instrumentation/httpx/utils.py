'''Helpers for extracting httpx request/response data for tracing.'''
import gzip
import zlib

from agent_trace.custom_logger import get_custom_logger

logger = get_custom_logger(__name__)


def headers_from_httpx(headers):
    '''Convert httpx.Headers to a plain dict.'''
    if headers is None:
        return {}
    return dict(headers)


def url_from_request_info(request_info):
    '''Return the request URL as a string.'''
    return str(request_info.url)


def _content_encoding_chain(headers):
    '''Return Content-Encoding tokens in wire order (outer coding first).'''
    if not headers:
        return []
    lower = {str(k).lower(): v for k, v in headers.items()}
    raw = lower.get('content-encoding') or ''
    return [t.strip().lower() for t in raw.split(',') if t.strip()]


def _apply_content_encoding_decodings(original, chain):
    '''Apply decodings in reverse wire order; return ``original`` on failure.'''
    out = original
    for encoding in reversed(chain):
        try:
            if encoding in ('gzip', 'x-gzip'):
                out = gzip.decompress(out)
            elif encoding == 'deflate':
                try:
                    out = zlib.decompress(out, -zlib.MAX_WBITS)
                except zlib.error:
                    out = zlib.decompress(out)
            elif encoding == 'br':
                try:
                    import brotli  # pylint: disable=import-outside-toplevel
                except ImportError:
                    logger.debug('brotli not installed; skipping br decompression for capture')
                    return original
                out = brotli.decompress(out)
            elif encoding in ('identity', 'compress'):
                continue
            else:
                logger.debug('Unsupported content-encoding for capture: %s', encoding)
                return original
        except Exception:  # pylint: disable=broad-except
            logger.debug('Response body decompression failed (%s)', encoding, exc_info=True)
            return original
    return out


def decode_response_body_for_capture(headers, body):
    '''Decode compressed response bytes for span capture (gzip/deflate/br).

    Transport hooks often see the on-the-wire body while ``Content-Encoding``
    still names the compression. httpx may also surface compressed bytes when
    decoding is deferred. If decoding fails or an encoding is unsupported,
    the original ``body`` is returned.
    '''
    if body in (None, b''):
        return body
    if not isinstance(body, (bytes, bytearray)):
        return body

    chain = _content_encoding_chain(headers)
    if not chain:
        return body

    return _apply_content_encoding_decodings(bytes(body), chain)


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


def read_response_body(stream):
    '''Read response body bytes and restore the stream for downstream consumers.'''
    if stream is None:
        return None

    body = _body_from_byte_stream(stream)
    if body is not None:
        return body

    httpcore_stream = getattr(stream, "_httpcore_stream", None)
    if httpcore_stream is not None:
        try:
            chunks = list(httpcore_stream)
            body = b"".join(chunks)
            stream._httpcore_stream = chunks  # pylint: disable=protected-access
            return body
        except Exception:  # pylint: disable=broad-except
            logger.debug("Unable to read httpx response stream body", exc_info=True)
            return None

    return None


async def read_response_body_async(stream):
    '''Read async response body bytes and restore the stream for downstream consumers.'''
    if stream is None:
        return None

    body = _body_from_byte_stream(stream)
    if body is not None:
        return body

    httpcore_stream = getattr(stream, "_httpcore_stream", None)
    if httpcore_stream is not None:
        try:
            chunks = []
            async for part in httpcore_stream:
                chunks.append(part)
            body = b"".join(chunks)
            stream._httpcore_stream = _ReplayAsyncStream(chunks)  # pylint: disable=protected-access
            return body
        except Exception:  # pylint: disable=broad-except
            logger.debug("Unable to read httpx async response stream body", exc_info=True)
            return None

    return None


class _ReplayAsyncStream:  # pylint: disable=too-few-public-methods
    '''Minimal async iterable used to replay captured response chunks.'''

    def __init__(self, chunks):
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
