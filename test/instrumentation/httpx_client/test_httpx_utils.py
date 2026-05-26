import gzip
import zlib

import pytest

from harness_sdk.instrumentation.httpx.utils import decode_response_body_for_capture


def test_decode_gzip_response_body():
    payload = b'{"hello":"world"}'
    compressed = gzip.compress(payload)
    headers = {'Content-Encoding': 'gzip', 'content-type': 'application/json'}
    assert decode_response_body_for_capture(headers, compressed) == payload


def test_decode_deflate_response_body():
    payload = b'plain text body'
    compressed = zlib.compress(payload)
    headers = {'Content-Encoding': 'deflate'}
    assert decode_response_body_for_capture(headers, compressed) == payload


def test_no_content_encoding_returns_raw():
    payload = b'not compressed'
    assert decode_response_body_for_capture({'content-type': 'text/plain'}, payload) == payload


def test_bad_gzip_returns_original():
    payload = b'not gzip at all'
    headers = {'Content-Encoding': 'gzip'}
    assert decode_response_body_for_capture(headers, payload) == payload


@pytest.mark.parametrize('empty', [None, b''])
def test_decode_empty_body(empty):
    assert decode_response_body_for_capture({'Content-Encoding': 'gzip'}, empty) == empty
