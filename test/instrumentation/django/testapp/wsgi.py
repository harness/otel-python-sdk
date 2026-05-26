"""
WSGI config for testapp project.

Agent initialization is deferred to pytest fixtures (see ``test_django_1.py``).
"""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("HA_ENABLE_CONSOLE_SPAN_EXPORTER", "true")
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "test.instrumentation.django.testapp.settings",
)

application = get_wsgi_application()
