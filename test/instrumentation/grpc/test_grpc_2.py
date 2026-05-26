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


def test_run(agent_with_filter, exporter):
    agent_with_filter.instrument()

    class Greeter(helloworld_pb2_grpc.GreeterServicer):
        def SayHello(self, request, context):
            logger.debug('Received request.')
            logger.debug('Returning response.')
            return helloworld_pb2.HelloReply(message='Hello, %s!' % request.name)

    executor = futures.ThreadPoolExecutor(max_workers=10)
    server = grpc.server(executor)
    try:
        helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
        server.add_insecure_port('[::]:50052')
        server.start()

        with grpc.insecure_channel('0.0.0.0:50052') as channel:
            stub = helloworld_pb2_grpc.GreeterStub(channel)
            permission_denied_exception = False
            try:
                stub.SayHello(helloworld_pb2.HelloRequest(name='you'))
            except grpc.RpcError as exc:
                permission_denied_exception = exc.code() == grpc.StatusCode.PERMISSION_DENIED

            assert permission_denied_exception

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
            assert span_object['attributes']['rpc.request.body'] == '{"name": "you"}'
            assert span_object['attributes']['rpc.grpc.status_code'] == 7
            exporter.clear()
    finally:
        server.stop(grace=0)
        executor.shutdown(wait=False, cancel_futures=True)
