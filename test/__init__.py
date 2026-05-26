import logging
import os
import sys
import traceback

os.environ['HA_ENABLE_CONSOLE_SPAN_EXPORTER'] = 'true'


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 0))
    portnum = sock.getsockname()[1]
    sock.close()

    return portnum


def configure_default_environment(agent, config_file_path, service_name):
    os.environ.setdefault('HA_CONFIG_FILE', '')
    os.environ.setdefault('HA_SERVICE_NAME', '')
    os.environ.setdefault('HA_LOG_LEVEL', 'ERROR')
    os.environ.setdefault('HA_ENABLE_CONSOLE_SPAN_EXPORTER', True)


def configure_inmemory_span_exporter(agent):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    memory_exporter = InMemorySpanExporter()
    simple_export_span_processor = SimpleSpanProcessor(memory_exporter)
    agent.register_processor(simple_export_span_processor)
    return memory_exporter


def setup_custom_logger(name):
    try:
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler = logging.FileHandler('agent.log', mode='a')
        handler.setFormatter(formatter)
        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.addHandler(screen_handler)
        return logger
    except:
        print('Failed to customize logger: exception=%s, stacktrace=%s',
              sys.exc_info()[0],
              traceback.format_exc())
