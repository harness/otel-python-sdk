'''Initialize all the components using configuration from AgentConfig'''
# pylint: disable=C0303
import os
import traceback
import logging

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPGrpcSpanExporter
except ImportError:
    pass

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHttpSpanExporter
from opentelemetry.trace import ProxyTracerProvider
from opentelemetry.sdk.resources import Resource
from agent_trace import constants
from agent_trace.config import config_pb2
from agent_trace.otlp_reporting import (
    compression_type_to_otlp_grpc,
    compression_type_to_otlp_http,
    is_supported_encoding,
    normalize_otlp_http_traces_endpoint,
)

logger = logging.getLogger(__name__)  # pylint: disable=C0103


class AgentInit:  # pylint: disable=R0902,R0903
    '''Initialize all the OTel components using configuration from AgentConfig'''

    def __init__(self, agent_config):
        logger.debug('Initializing AgentInit object.')
        self._config = agent_config

        if hasattr(os, 'register_at_fork'):
            logger.info('Registering after_in_child handler.')
            os.register_at_fork(after_in_child=self.post_fork)  # pylint:disable=E1101

    def post_fork(self):
        self.apply_config(self._config)  # pylint:disable=W0212

    def apply_config(self, agent_config):
        if agent_config:
            self._config = agent_config
        self.init_trace_provider()
        self.init_propagation()
        if (
            "HA_ENABLE_CONSOLE_SPAN_EXPORTER" in os.environ
            or "AT_ENABLE_CONSOLE_SPAN_EXPORTER" in os.environ
            or "TA_ENABLE_CONSOLE_SPAN_EXPORTER" in os.environ
        ):
            self.set_console_span_processor()

    def init_trace_provider(self) -> None:
        if isinstance(trace.get_tracer_provider(), ProxyTracerProvider):
            logger.debug("no configured trace provider detected, adding one")
            resource_attributes = {
                "service.name": self._config.config.service_name.value,
                "service.instance.id": os.getpid(),
                "telemetry.sdk.version": constants.TELEMETRY_SDK_VERSION,
                "telemetry.sdk.name": constants.TELEMETRY_SDK_NAME,
                "telemetry.sdk.language": constants.TELEMETRY_SDK_LANGUAGE
            }
            if self._config.config.resource_attributes:
                resource_attributes.update(self._config.config.resource_attributes)
            tracer_provider = TracerProvider(
                resource=Resource.create(resource_attributes)
            )
            trace.set_tracer_provider(tracer_provider)
        else:
            logger.debug("tracer provider already configured, skipping trace provider configuration")

    def init_propagation(self) -> None:
        propagator_list = []
        for prop_format in self._config.config.propagation_formats:
            if prop_format == config_pb2.PropagationFormat.TRACECONTEXT:
                from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator  # pylint: disable=C0415
                propagator_list += [TraceContextTextMapPropagator()]
            if prop_format == config_pb2.PropagationFormat.B3:
                from opentelemetry.propagators.b3 import B3MultiFormat  # pylint: disable=C0415
                propagator_list += [B3MultiFormat()]

        from opentelemetry.propagate import set_global_textmap  # pylint: disable=C0415
        from opentelemetry.propagators.composite import CompositePropagator  # pylint: disable=C0415
        set_global_textmap(CompositePropagator(propagator_list))

    def init_library_instrumentation(self, instrumentation_name, wrapper_instance):
        logger.debug("Attempting to initialize %s instrumentation", instrumentation_name)
        try:
            wrapper_instance.instrument()
        except Exception as err:  # pylint: disable=W0703
            logger.debug(
                constants.INST_WRAP_EXCEPTION_MSSG,
                instrumentation_name,
                err,
                traceback.format_exc(),
            )

    def register_processor(self, processor) -> None:
        trace.get_tracer_provider().add_span_processor(processor)

    def set_console_span_processor(self) -> None:
        console_span_exporter = ConsoleSpanExporter(
            service_name=self._config.config.service_name)
        simple_export_span_processor = SimpleSpanProcessor(console_span_exporter)
        trace.get_tracer_provider().add_span_processor(simple_export_span_processor)

    def _init_exporter(self, trace_reporter_type):
        exporter_type = ''
        exporter = None
        exporter_endpoint = None
        reporting = self._config.config.reporting
        reporting_encoding = getattr(self._config, 'reporting_encoding', None)
        if reporting_encoding and not is_supported_encoding(reporting_encoding):
            logger.warning(
                'Unsupported reporting encoding `%s`; OTLP protobuf encoding will be used',
                reporting_encoding,
            )
        try:
            _token = reporting.token.value
            headers = {"agent-trace-token": _token} if _token else {}
            if trace_reporter_type == config_pb2.TraceReporterType.OTLP:
                exporter_type = 'otlp'
                exporter_endpoint = reporting.endpoint.value
                exporter = OTLPGrpcSpanExporter(
                    endpoint=exporter_endpoint,
                    insecure=not reporting.secure.value,
                    headers=headers,
                    compression=compression_type_to_otlp_grpc(reporting.compression_type),
                )
            elif trace_reporter_type == config_pb2.TraceReporterType.OTLP_HTTP:
                exporter_type = 'otlp_http'
                exporter_endpoint = normalize_otlp_http_traces_endpoint(
                    reporting.endpoint.value
                )
                exporter = OTLPHttpSpanExporter(
                    endpoint=exporter_endpoint,
                    headers=headers,
                    compression=compression_type_to_otlp_http(reporting.compression_type),
                )

            if exporter:
                logger.info('Initialized %s exporter reporting to `%s`',
                            exporter_type, exporter_endpoint)
            return exporter
        except Exception as err:  # pylint: disable=W0703
            logger.error('Failed to initialize %s exporter: exception=%s, stacktrace=%s',
                         exporter_type, err, traceback.format_exc())
            return None
