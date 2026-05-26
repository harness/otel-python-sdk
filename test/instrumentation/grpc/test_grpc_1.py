# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import re
from concurrent import futures

import grpc

from test import setup_custom_logger
from test.instrumentation.grpc import helloworld_pb2, helloworld_pb2_grpc

logger = setup_custom_logger(__name__)


def _assert_grpc_round_trip(exporter, channel, name, *, expect_metadata):
    stub = helloworld_pb2_grpc.GreeterStub(channel)
    response = stub.SayHello(helloworld_pb2.HelloRequest(name=name))
    assert response.message == f'Hello, {name}!'
    logger.info("Greeter client received: %s", response.message)

    span_list = exporter.get_finished_spans()
    assert span_list
    logger.debug('len(span_list): %s', len(span_list))
    assert len(span_list) == 2
    span_object = json.loads(span_list[0].to_json())

    assert span_object['attributes']['rpc.system'] == 'grpc'
    assert span_object['attributes']['rpc.method'] == 'SayHello'
    user_agent_re = re.compile(r'grpc-python/.* grpc-c/.* (.*; chttp2)')
    assert re.match(
        user_agent_re, span_object['attributes']['rpc.request.metadata.user-agent'])
    assert span_object['attributes']['rpc.request.body'] == f'{{"name": "{name}"}}'
    assert span_object['attributes']['rpc.grpc.status_code'] == 0
    assert span_object['attributes']['rpc.response.body'] == f'{{"message": "Hello, {name}!"}}'
    if expect_metadata:
        assert span_object['attributes']['rpc.response.metadata.tester2'] == 'tester2'
        assert span_object['attributes']['rpc.response.metadata.tester'] == 'tester'
    exporter.clear()


def test_grpc(agent, exporter):
    agent.instrument()

    class Greeter(helloworld_pb2_grpc.GreeterServicer):
        def SayHello(self, request, context):
            logger.debug('Received request.')
            if request.name != 'no-metadata':
                metadata = (('tester', 'tester'), ('tester2', 'tester2'))
                logger.debug('Setting custom headers.')
                context.set_trailing_metadata(metadata)
            logger.debug('Returning response.')
            return helloworld_pb2.HelloReply(message='Hello, %s!' % request.name)

    executor = futures.ThreadPoolExecutor(max_workers=10)
    server = grpc.server(executor)
    try:
        helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
        server.add_insecure_port('[::]:50051')
        server.start()

        with grpc.insecure_channel('0.0.0.0:50051') as channel:
            _assert_grpc_round_trip(exporter, channel, 'you', expect_metadata=True)
            _assert_grpc_round_trip(exporter, channel, 'no-metadata', expect_metadata=False)
    finally:
        server.stop(grace=0)
        executor.shutdown(wait=False, cancel_futures=True)
