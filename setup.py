# pylint: skip-file
"""Setup for harness-sdk with optional vendored OpenTelemetry gen-ai packages."""
import os
import shutil
from setuptools import setup, find_packages

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_ROOT)

exec(open(os.path.join(_ROOT, "src/harness_sdk/version.py")).read())

with open(os.path.join(_ROOT, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

def _bundle_vendor():
    _vendor_roots = (
        os.path.join(_ROOT, 'temporary-vendor'),
        os.path.join(_PARENT, 'temporary-vendor'),
    )
    _vendor_dst = os.path.join(_ROOT, 'src', 'opentelemetry')
    _packages = (
        ('opentelemetry-instrumentation-anthropic', 'src', 'opentelemetry'),
        ('opentelemetry-instrumentation-openai-v2', 'src', 'opentelemetry'),
        ('opentelemetry-util-genai', 'src', 'opentelemetry'),
    )
    for vendor_root in _vendor_roots:
        if not os.path.isdir(vendor_root):
            continue
        os.makedirs(_vendor_dst, exist_ok=True)
        for pkg_dir, src_subdir, namespace in _packages:
            _src = os.path.join(vendor_root, pkg_dir, src_subdir, namespace)
            if os.path.isdir(_src):
                shutil.copytree(_src, _vendor_dst, dirs_exist_ok=True)
        break


_bundle_vendor()
_VENDOR_DST = os.path.join(_ROOT, 'src', 'opentelemetry')

_VENDOR_PACKAGES = []
if os.path.isdir(_VENDOR_DST):
    for _dirpath, _dirnames, _filenames in os.walk(_VENDOR_DST):
        if '__init__.py' in _filenames:
            _rel = os.path.relpath(_dirpath, os.path.join(_ROOT, 'src'))
            _VENDOR_PACKAGES.append(_rel.replace(os.sep, '.'))

setup(
    name="harness-sdk",
    version=__version__,
    description="Generic Python agent SDK with instrumentation and plugin architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src") + _VENDOR_PACKAGES,
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "opentelemetry-api==1.41.1",
        "opentelemetry-exporter-otlp==1.41.1",
        "opentelemetry-instrumentation==0.62b1",
        "opentelemetry-instrumentation-aiohttp-client==0.62b1",
        "opentelemetry-instrumentation-botocore==0.62b1",
        "opentelemetry-instrumentation-wsgi==0.62b1",
        "opentelemetry-instrumentation-fastapi==0.62b1",
        "opentelemetry-instrumentation-flask==0.62b1",
        "opentelemetry-instrumentation-mysql==0.62b1",
        "opentelemetry-instrumentation-psycopg2==0.62b1",
        "opentelemetry-instrumentation-requests==0.62b1",
        "opentelemetry-instrumentation-httpx==0.62b1",
        "opentelemetry-instrumentation-grpc==0.62b1",
        "opentelemetry-instrumentation-django==0.62b1",
        "opentelemetry-instrumentation-mcp==0.60.0",
        "opentelemetry-semantic-conventions-ai>=0.5.1,<0.6.0",
        "opentelemetry-propagator-b3==1.41.1",
        "opentelemetry-proto==1.41.1",
        "opentelemetry-sdk==1.41.1",
        "opentelemetry-util-http==0.62b1",
        "google>=3.0.0",
        "pyyaml",
        "protobuf",
        "psutil",
        "distro",
        "setuptools",
        "jaraco.text",
        "platformdirs",
    ],
    entry_points={
        'console_scripts': [
            'harness-instrument = harness_sdk.autoinstrumentation.wrapper:run',
        ],
        'harness_sdk_observability_plugin': [
            'builtin_pipeline = harness_sdk.plugins.builtin.pipeline:factory',
            'builtin_span_attributes = harness_sdk.plugins.builtin.span_attributes:factory',
        ],
    },
    extras_require={
        'anthropic': ['anthropic>=0.34.0'],
        'openai': ['openai>=1.40.0'],
        'litellm': ['litellm>=1.60.0'],
    },
)
