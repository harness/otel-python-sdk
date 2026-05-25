'''traceable instrumentation logic for aiohttp-client'''
import types
import typing
from collections import deque
import asyncio
import aiohttp
import wrapt
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.aiohttp_client.version import __version__
from opentelemetry.trace import TracerProvider, get_tracer
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor

from agent_trace.plugins.control import get_control_registry
from agent_trace.instrumentation import BaseInstrumentorWrapper

from agent_trace.custom_logger import get_custom_logger

logger = get_custom_logger(__name__)

# Max time to wait for response data to be read.
MAX_WAIT_TIME = 0.1  # seconds


# aiohttp-client instrumentation module wrapper class
class AioHttpClientInstrumentorWrapper(AioHttpClientInstrumentor, BaseInstrumentorWrapper):
    """traceable wrapper class around OpenTelemetry AioHttpClient Instrumentor class"""

    # Constructor
    def __init__(self):
        '''Constructor'''
        logger.debug('Entering AioHttpClientInstrumentor.__init__().')
        super().__init__()

    def _instrument(self, **kwargs) -> None:
        '''Enable instrumentation.'''
        logger.debug(
            'Entering AioHttpClientInstrumentorWrapper._instrument().')
        # Initialize OTel instrumentor
        super()._instrument(
            tracer_provider=kwargs.get("tracer_provider")
        )
        # Initialize traceable instrumentor
        _instrument(
            tracer_provider=kwargs.get("tracer_provider"),
            url_filter=kwargs.get("url_filter"),
            aiohttp_client_wrapper=self
        )


# aliases for type definitions
_UrlFilterT = typing.Optional[typing.Callable[[  # pylint: disable=unsubscriptable-object
    str], str]]
_SpanNameT = typing.Optional[  # pylint: disable=unsubscriptable-object
    typing.Union[typing.Callable[[aiohttp.TraceRequestStartParams],  # pylint: disable=unsubscriptable-object
    str], str]
]


# build an aiohttp trace config


def create_trace_config(  # pylint:disable=R0915
        url_filter: _UrlFilterT = None,
        tracer_provider: TracerProvider = None,
        aiohttp_client_wrapper: AioHttpClientInstrumentorWrapper = None
) -> aiohttp.TraceConfig:
    '''Build an aiohttp-client trace config for use with traceable'''
    tracer = get_tracer(__name__, __version__, tracer_provider)

    # This runs at the start of a request
    async def on_request_start(
            unused_session: aiohttp.ClientSession,
            trace_config_ctx: types.SimpleNamespace,
            params: aiohttp.TraceRequestStartParams,
    ):
        logger.debug('Entering traceable on_request_start().')
        # Get the current span
        span = trace.get_current_span()

        # Extract request details
        url = str(params.url)
        headers = params.headers if params.headers else {}

        # Initialize request body collection
        trace_config_ctx.request_body = bytearray()
        trace_config_ctx.request_body_complete = False

        # We don't have the body at this point, but we can still apply filtering based on URL and headers
        # The body will be captured in on_request_chunk_sent and processed in on_request_end
        filter_result = get_control_registry().evaluate(span, url, headers, None, False)

        # Store the filter result in the trace context for later use
        trace_config_ctx.filter_result = filter_result

    # This runs after each chunk of request data is sent
    async def on_request_chunk_sent(
            unused_session: aiohttp.ClientSession,
            trace_config_ctx: types.SimpleNamespace,
            params: aiohttp.TraceRequestChunkSentParams
    ):
        logger.debug('Entering traceable on_request_chunk_sent().')
        if hasattr(params, 'chunk') and params.chunk is not None:
            try:
                # Append raw bytes to our buffer - we'll decode once at the end
                trace_config_ctx.request_body.extend(params.chunk)
                logger.debug('Collected %s bytes of request data', str(len(params.chunk)))
            except Exception as e:  # pylint:disable=W0718
                logger.error('Error collecting request chunk: %s', str(e))

    # This runs after an exception occurs
    async def on_request_exception(  # pylint: disable=W0613
            unused_session: aiohttp.ClientSession,
            trace_config_ctx: types.SimpleNamespace,
            params: aiohttp.TraceRequestExceptionParams,
    ):
        logger.debug('Entering on_request_exception().')

    # This runs after the request
    async def on_request_end(
            unused_session: aiohttp.ClientSession,
            trace_config_ctx: types.SimpleNamespace,
            params: aiohttp.TraceRequestEndParams,
    ) -> None:
        logger.debug('Entering traceable on_request_end().')
        response_body = b''
        if hasattr(params.response, 'content') and params.response.content is not None:
            content_stream = params.response.content
            # A temporary dual end queue to copy data into and use to reset the stream
            tmp_deque = deque()

            try:
                # Read response data with timeout protection
                while not content_stream.at_eof():
                    try:
                        response_chunk = await asyncio.wait_for(
                            content_stream.read(),
                            MAX_WAIT_TIME
                        )
                        tmp_deque.append(response_chunk)
                        response_body += response_chunk
                    except asyncio.TimeoutError:
                        logger.debug('Timeout while reading response data')
                        break
            except Exception as err:  # pylint:disable=W0718
                logger.error('Error reading response data: %s', str(err))
            finally:
                # Reset response content stream for other consumers
                content_stream._cursor = 0  # pylint: disable=W0212
                content_stream._buffer = tmp_deque  # pylint: disable=W0212

        # Convert collected request body bytes to string
        request_body = ''
        if hasattr(trace_config_ctx, 'request_body') and trace_config_ctx.request_body:
            try:
                request_body = trace_config_ctx.request_body.decode('utf-8', errors='replace')
            except Exception as e:  # pylint:disable=W0718
                logger.error('Error decoding request body: %s', str(e))

        span = trace.get_current_span()

        # Add headers & body to span
        if span.is_recording():
            aiohttp_client_wrapper.generic_request_handler(
                params.headers, request_body, span)
            aiohttp_client_wrapper.generic_response_handler(
                params.response.headers, response_body, span)

        trace_config_ctx.end_callback_called = True
        trace_config_ctx.span = span

    def _trace_config_ctx_factory(**kwargs):
        kwargs.setdefault("trace_request_ctx", {})
        return types.SimpleNamespace(
            tracer=tracer,
            url_filter=url_filter,
            **kwargs,
            request_body='',
        )

    trace_config = aiohttp.TraceConfig(
        trace_config_ctx_factory=_trace_config_ctx_factory
    )

    trace_config.on_request_start.append(on_request_start)
    trace_config.on_request_chunk_sent.append(on_request_chunk_sent)
    trace_config.on_request_end.append(on_request_end)
    trace_config.on_request_exception.append(on_request_exception)

    return trace_config


def _instrument(
        tracer_provider: TracerProvider = None,
        url_filter: _UrlFilterT = None,
        aiohttp_client_wrapper: AioHttpClientInstrumentorWrapper = None
) -> None:
    '''Setup details of trace config context'''

    def instrumented_init(wrapped, instance, args, kwargs) -> None:  # pylint: disable=W0613
        if context_api.get_value("suppress_instrumentation"):
            return wrapped(*args, **kwargs)

        trace_configs = list(kwargs.get("trace_configs") or ())

        trace_config = create_trace_config(
            url_filter=url_filter,
            tracer_provider=tracer_provider,
            aiohttp_client_wrapper=aiohttp_client_wrapper
        )
        trace_configs.append(trace_config)

        kwargs["trace_configs"] = trace_configs
        return wrapped(*args, **kwargs)

    wrapt.wrap_function_wrapper(
        aiohttp.ClientSession, "__init__", instrumented_init
    )
