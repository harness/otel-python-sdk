import os
import sys
import threading
import traceback

import distro

from harness_sdk import constants
from harness_sdk.agent_init import AgentInit
from harness_sdk.config.config import Config
from harness_sdk.custom_logger import get_custom_logger
from harness_sdk.env import get_env_value
from harness_sdk.instrumentation.instrumentation_definitions import (
    SUPPORTED_LIBRARIES,
    get_instrumentation_wrapper,
    FLASK_KEY,
    DJANGO_KEY,
    FAST_API_KEY,
    instrument_supported_contrib_without_wrapper,
)
from harness_sdk.instrumentation.genai_env import maybe_set_genai_payload_capture_env_vars
from harness_sdk.plugins.control import ControlPlugin, get_control_registry
from harness_sdk.plugins.loader import load_control_plugins, load_observability_plugins
from harness_sdk.version import __version__  # pylint:disable=C0413

logger = get_custom_logger(__name__)


class Agent:
    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._singleton_lock:
                logger.debug('Creating Agent')
                logger.debug('Python version: %s', sys.version)
                logger.debug('Harness SDK version: %s', __version__)
                cls._instance = super(Agent, cls).__new__(cls)
        else:
            logger.debug('Using existing Agent.')
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        logger.debug('Initializing Agent.')
        if not self.is_enabled():
            return
        self.is_lambda = False
        try:
            self._config = Config()
            self._init = AgentInit(self._config)
            load_control_plugins(self._config)
            self._init.init_trace_provider()
            self._init.init_propagation()
            load_observability_plugins(self._config)
            self._initialized = True
            logger.debug("Platform: %s", distro.id())
            logger.debug("Platform version: %s", distro.version())
            logger.debug('Harness SDK version: %s', __version__)
            logger.debug("successfully initialized harness sdk")
            if hasattr(os, 'register_at_fork'):
                logger.info('Registering after_in_child handler.')
                os.register_at_fork(after_in_child=self.post_fork)  # pylint:disable=E1101
        except Exception as err:  # pylint: disable=W0703
            logger.error(
                'Failed to initialize Agent: exception=%s, stacktrace=%s',
                err,
                traceback.format_exc(),
            )

    def post_fork(self):
        logger.info("In post fork hook")
        self._init.post_fork()
        load_control_plugins(self._config)

    def instrument(self, app=None, skip_libraries=None, auto_instrument=False):
        logger.debug("Beginning instrumentation")
        self._init.apply_config(self._config)

        if skip_libraries is None:
            skip_libraries = []
        if not self.is_initialized():
            logger.debug('agent is not initialized, not instrumenting')
            return

        maybe_set_genai_payload_capture_env_vars()

        for library_key in SUPPORTED_LIBRARIES:
            if library_key in skip_libraries:
                logger.debug('not attempting to instrument %s', library_key)
                continue
            logger.debug("attempting to instrument %s", library_key)
            self._instrument(library_key, app, auto_instrument)

        if self._should_instrument_generic_contrib():
            instrument_supported_contrib_without_wrapper(skip_libraries)
        else:
            logger.debug("Skipping generic contrib fallback instrumentation")
        logger.debug("Complete instrumentation")

    def _should_instrument_generic_contrib(self):
        """
        Fallback instrumentation runs when no control plugin provides blocking.
        """
        if get_control_registry().has_blocking_capability():
            logger.info("Blocking control plugin active, skipping fallback instrumentation")
            return False
        logger.info("No blocking control plugin, enabling fallback instrumentation")
        return True

    def _instrument(self, library_key, app=None, auto_instrument=False):
        wrapper_instance = get_instrumentation_wrapper(library_key)
        if wrapper_instance is None:
            logger.debug("no instrumentation wrapper instance available for %s", library_key)
            return

        if library_key == FLASK_KEY and app is not None:
            wrapper_instance.with_app(app)

        if library_key == DJANGO_KEY and auto_instrument is True:
            from harness_sdk.instrumentation.django.django_auto_instrumentation_compat import (  # pylint: disable=C0415
                add_django_auto_instr_wrappers,
            )
            add_django_auto_instr_wrappers(self, wrapper_instance)
            return

        if library_key == FAST_API_KEY:
            from harness_sdk.instrumentation.fast_api.fast_api_auto_instrumentation_compat import (  # pylint: disable=C0415
                add_fast_api_auto_instr_wrappers,
            )
            add_fast_api_auto_instr_wrappers(self, wrapper_instance)
            return

        logger.debug("registering library %s with wrapper instance", library_key)
        self.register_library(library_key, wrapper_instance)

    def register_library(self, library_name, wrapper_instance):
        logger.debug('attempting to register library instrumentation: %s', library_name)
        try:
            self._init.init_library_instrumentation(library_name, wrapper_instance)
        except Exception as err:  # pylint: disable=W0703
            logger.debug(constants.EXCEPTION_MESSAGE, library_name, err, traceback.format_exc())

    def register_control_plugin(self, plugin: ControlPlugin) -> None:
        logger.debug('Registering control plugin %s', plugin.name)
        plugin.on_init(self._config)
        get_control_registry().register(plugin)

    def register_processor(self, processor) -> None:  # pylint: disable=R1710
        logger.debug('Entering Agent.register_processor().')
        if not self.is_initialized():
            return None
        return self._init.register_processor(processor)

    def is_enabled(self) -> bool:
        enabled = get_env_value('ENABLED')
        if enabled and enabled.lower() == 'false':
            logger.debug("ENABLED is disabled.")
            return False
        return True

    def is_initialized(self) -> bool:
        if not self.is_enabled():
            return False
        if not self._initialized:
            return False
        return True
