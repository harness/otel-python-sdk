import io
import json
import logging
import sys
import traceback
import zipfile

import botocore.session
import pytest

pytest.importorskip("docker")

from moto import mock_iam, mock_lambda  # pylint: disable=import-error
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from harness_sdk.agent import Agent
from test import setup_custom_logger


def test_run():
    logger = setup_custom_logger(__name__)
    with mock_iam(), mock_lambda():
        agent = Agent()
        agent.instrument(None)

        logger.info('Agent initialized.')
        logger.info('Adding in-memory span exporter.')
        memory_exporter = InMemorySpanExporter()
        agent.register_processor(SimpleSpanProcessor(memory_exporter))
        logger.info('Added in-memory span exporter')

        logger.info('Running test calls.')
        try:
            session = botocore.session.get_session()
            session.set_credentials(
                access_key="access-key", secret_key="secret-key"
            )
            region = "us-west-2"
            client = session.create_client("lambda", region_name=region)
            iam_client = session.create_client("iam", region_name=region)
            arn = _create_role_and_get_arn(iam_client)
            result = _create_lambda_function(
                'some_function', return_headers_lambda_str(), client, arn
            )
            memory_exporter.clear()
            response = client.invoke(
                Payload=json.dumps({}),
                FunctionName=result['FunctionArn'],
                InvocationType="RequestResponse",
            )

            spans = memory_exporter.get_finished_spans()
            invoke_span = spans[-1]

            assert invoke_span.attributes['faas.invoked_name'] == 'some_function'
            assert invoke_span.attributes['http.status_code'] == 200
            assert invoke_span.attributes['rpc.service'] == 'Lambda'
            memory_exporter.clear()
        except Exception:
            logger.error(
                'Failed to test boto instrumentation wrapper: exception=%s, stacktrace=%s',
                sys.exc_info()[0],
                traceback.format_exc(),
            )
            raise


def get_as_zip_file(file_name, content):
    zip_output = io.BytesIO()
    with zipfile.ZipFile(zip_output, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(file_name, content)
    zip_output.seek(0)
    return zip_output.read()


def return_headers_lambda_str():
    pfunc = """
def lambda_handler(event, context):
    print("custom log event")
    headers = event.get('headers', event.get('attributes', {}))
    return headers
"""
    return pfunc


def _create_role_and_get_arn(iam_client) -> str:
    return iam_client.create_role(
        RoleName="my-role",
        AssumeRolePolicyDocument="some policy",
        Path="/my-path/",
    )["Role"]["Arn"]


def _create_lambda_function(function_name: str, function_code: str, client, role_arn):
    return client.create_function(
        FunctionName=function_name,
        Runtime="python3.8",
        Role=role_arn,
        Handler="lambda_function.lambda_handler",
        Code={
            "ZipFile": get_as_zip_file("lambda_function.py", function_code)
        },
        Description="test lambda function",
        Timeout=3,
        MemorySize=128,
        Publish=True,
    )
