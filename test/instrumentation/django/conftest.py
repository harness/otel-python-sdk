"""Pytest-django configuration for instrumentation tests."""
import os

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "test.instrumentation.django.testapp.settings",
)
