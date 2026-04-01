"""Tests for VLLMBackend."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from dlrouter.backends.factory import create_backend
from dlrouter.backends.vllm_backend import VLLMBackend
from dlrouter.config import BackendConfig
from dlrouter.constants import BackendType


NODE_URL = 'http://10.0.0.1:8000'


class _AsyncLines:
    """Async iterable over lines of bytes, for mocking resp.content."""

    def __init__(self, body: bytes) -> None:
        self._lines = body.splitlines()

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for line in self._lines:
            yield line


def _make_session_mock(status: int = 200, body: bytes = b'', exception=None):
    """Build a mock aiohttp.ClientSession context manager chain.

    Returns (session_mock, response_mock) so callers can inspect them.
    """
    resp = AsyncMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body.decode())
    resp.content = _AsyncLines(body)

    req_ctx = AsyncMock()
    if exception:
        req_ctx.__aenter__ = AsyncMock(side_effect=exception)
    else:
        req_ctx.__aenter__ = AsyncMock(return_value=resp)
    req_ctx.__aexit__ = AsyncMock(return_value=False)

    session = AsyncMock()
    session.post = MagicMock(return_value=req_ctx)
    session.get = MagicMock(return_value=req_ctx)

    sess_ctx = AsyncMock()
    sess_ctx.__aenter__ = AsyncMock(return_value=session)
    sess_ctx.__aexit__ = AsyncMock(return_value=False)

    return sess_ctx, resp


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_creates_vllm_backend(self):
        cfg = BackendConfig(type=BackendType.VLLM)
        backend = create_backend(cfg)
        assert isinstance(backend, VLLMBackend)

    def test_supports_pd_disagg(self):
        cfg = BackendConfig(type=BackendType.VLLM)
        backend = create_backend(cfg)
        assert backend.supports_pd_disagg() is True


# ---------------------------------------------------------------------------
# forward_request
# ---------------------------------------------------------------------------


class TestForwardRequest:
    async def test_success(self):
        sess_ctx, _ = _make_session_mock(status=200, body=b'{"choices":[]}')
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            result = await backend.forward_request(NODE_URL, '/v1/chat/completions', {'model': 'x'})
        assert result == '{"choices":[]}'

    async def test_raises_on_connection_error(self):
        sess_ctx, _ = _make_session_mock(exception=aiohttp.ClientConnectionError('refused'))
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            with pytest.raises(aiohttp.ClientConnectionError):
                await backend.forward_request(NODE_URL, '/v1/chat/completions', {})


# ---------------------------------------------------------------------------
# stream_forward
# ---------------------------------------------------------------------------


class TestStreamForward:
    async def test_yields_non_empty_lines(self):
        body = b'data: {"id":1}\ndata: [DONE]'
        sess_ctx, _ = _make_session_mock(status=200, body=body)
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            chunks = [chunk async for chunk in backend.stream_forward(NODE_URL, '/v1/chat/completions', {})]
        assert len(chunks) > 0
        combined = b''.join(chunks)
        assert b'data: {"id":1}' in combined

    async def test_raises_on_error(self):
        sess_ctx, _ = _make_session_mock(exception=aiohttp.ServerConnectionError('server error'))
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            with pytest.raises(aiohttp.ServerConnectionError):
                async for _ in backend.stream_forward(NODE_URL, '/v1/chat/completions', {}):
                    pass


# ---------------------------------------------------------------------------
# fetch_models
# ---------------------------------------------------------------------------


class TestFetchModels:
    def test_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'data': [
                {'id': '/models/Qwen3-32B'},
                {'id': '/models/Llama-3'},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch('dlrouter.backends.vllm_backend.requests.get', return_value=mock_resp):
            backend = VLLMBackend()
            models = backend.fetch_models(NODE_URL)

        assert models == ['/models/Qwen3-32B', '/models/Llama-3']

    def test_empty_data(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {'data': []}
        mock_resp.raise_for_status = MagicMock()

        with patch('dlrouter.backends.vllm_backend.requests.get', return_value=mock_resp):
            backend = VLLMBackend()
            models = backend.fetch_models(NODE_URL)

        assert models == []

    def test_connection_error_returns_empty(self):
        with patch(
            'dlrouter.backends.vllm_backend.requests.get',
            side_effect=Exception('connection refused'),
        ):
            backend = VLLMBackend()
            models = backend.fetch_models(NODE_URL)

        assert models == []


# ---------------------------------------------------------------------------
# check_health
# ---------------------------------------------------------------------------


class TestCheckHealth:
    async def test_healthy_200(self):
        sess_ctx, _ = _make_session_mock(status=200)
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            assert await backend.check_health(NODE_URL) is True

    async def test_unhealthy_non_200(self):
        sess_ctx, _ = _make_session_mock(status=503)
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            assert await backend.check_health(NODE_URL) is False

    async def test_connection_error_returns_false(self):
        sess_ctx, _ = _make_session_mock(exception=aiohttp.ClientConnectionError('refused'))
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            assert await backend.check_health(NODE_URL) is False


# ---------------------------------------------------------------------------
# deregister_node
# ---------------------------------------------------------------------------


class TestDeregisterNode:
    def test_is_noop(self):
        backend = VLLMBackend()
        # Should not raise
        backend.deregister_node(NODE_URL)


# ---------------------------------------------------------------------------
# PD disaggregation
# ---------------------------------------------------------------------------


class TestPDDisagg:
    def test_supports_pd_disagg(self):
        assert VLLMBackend().supports_pd_disagg() is True

    def test_is_connected_pd_always_true(self):
        backend = VLLMBackend()
        assert backend.is_connected_pd('http://p:8001', 'http://d:8002') is True

    async def test_connect_pd_is_noop(self):
        backend = VLLMBackend()
        # Should not raise
        await backend.connect_pd('http://p:8001', 'http://d:8002')

    async def test_prefill_request_uses_max_tokens_1(self):
        body = b'{"id":"cmpl-1","choices":[{"message":{"content":"Hello"},"finish_reason":"length"}]}'
        sess_ctx, _ = _make_session_mock(status=200, body=body)
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            result = await backend.prefill_request(
                NODE_URL,
                '/v1/chat/completions',
                {'model': 'x', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 100},
            )
        assert result is not None
        assert result['id'] == 'cmpl-1'
        # Verify that the actual call used max_tokens=1
        session = sess_ctx.__aenter__.return_value
        call_kwargs = session.post.call_args
        assert call_kwargs.kwargs['json']['max_tokens'] == 1

    async def test_decode_request_appends_first_token(self):
        """Test decode_request sends original request (not modified messages)."""
        body = b'{"id":"cmpl-2","choices":[{"message":{"content":"World"}}]}'
        sess_ctx, _ = _make_session_mock(status=200, body=body)
        prefill_info = {'id': 'cmpl-1', 'choices': [{'message': {'content': 'Hello'}}]}
        original_messages = [{'role': 'user', 'content': 'hi'}]
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            await backend.decode_request(
                NODE_URL,
                '/v1/chat/completions',
                {'model': 'x', 'messages': list(original_messages)},
                prefill_info,
                stream=False,
            )
        session = sess_ctx.__aenter__.return_value
        sent_messages = session.post.call_args.kwargs['json']['messages']
        # vLLM PD: decode sends ORIGINAL request, not modified with first token
        assert sent_messages == original_messages

    async def test_decode_request_no_first_token_sends_original(self):
        body = b'{"id":"cmpl-3","choices":[{"message":{"content":"resp"}}]}'
        sess_ctx, _ = _make_session_mock(status=200, body=body)
        original_messages = [{'role': 'user', 'content': 'hi'}]
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            await backend.decode_request(
                NODE_URL,
                '/v1/chat/completions',
                {'model': 'x', 'messages': list(original_messages)},
                {},  # empty prefill_info
                stream=False,
            )
        session = sess_ctx.__aenter__.return_value
        sent_messages = session.post.call_args.kwargs['json']['messages']
        assert sent_messages == original_messages


# ---------------------------------------------------------------------------
# Request ID generation (vLLM official format)
# ---------------------------------------------------------------------------


class TestRequestIdGeneration:
    """Test request_id generation in official vLLM proxy format."""

    def test_generate_request_id_format(self):
        """Test request_id follows official format."""
        backend = VLLMBackend()
        p_url = 'http://10.0.0.1:8000'
        d_url = 'http://10.0.0.2:8000'

        request_id = backend._generate_request_id(p_url, d_url)

        # Format: ___prefill_addr_{p_zmq}___decode_addr_{d_zmq}_{uuid}
        assert request_id.startswith('___prefill_addr_')
        assert '___decode_addr_' in request_id
        assert '10.0.0.1:8000' in request_id
        assert '10.0.0.2:8000' in request_id

    def test_get_zmq_address_without_discovery(self):
        """Test _get_zmq_address without ZMQ discovery falls back to HTTP address."""
        backend = VLLMBackend()
        assert backend._get_zmq_address('http://10.0.0.1:8000') == '10.0.0.1:8000'
        assert backend._get_zmq_address('https://10.0.0.2:8000') == '10.0.0.2:8000'

    def test_get_zmq_address_with_discovery(self):
        """Test _get_zmq_address with ZMQ discovery returns ZMQ address."""
        from unittest.mock import MagicMock

        backend = VLLMBackend()
        mock_discovery = MagicMock()
        mock_discovery.get_zmq_address.return_value = '10.0.0.1:30001'
        backend.set_zmq_discovery(mock_discovery)

        result = backend._get_zmq_address('http://10.0.0.1:8000')
        assert result == '10.0.0.1:30001'

    def test_set_zmq_discovery(self):
        """Test set_zmq_discovery method."""
        from unittest.mock import MagicMock

        backend = VLLMBackend()
        assert backend._zmq_discovery is None

        mock_discovery = MagicMock()
        backend.set_zmq_discovery(mock_discovery)
        assert backend._zmq_discovery is mock_discovery


class TestPrefillWithRequestId:
    """Test prefill_request with request_id generation."""

    async def test_prefill_request_with_d_url_generates_request_id(self):
        """Test that prefill_request with d_url generates request_id."""
        body = b'{"id":"cmpl-1","choices":[{"message":{"content":"Hi"}}]}'
        sess_ctx, _ = _make_session_mock(status=200, body=body)

        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            result = await backend.prefill_request(
                'http://10.0.0.1:8000',
                '/v1/chat/completions',
                {'model': 'x', 'messages': [{'role': 'user', 'content': 'hi'}]},
                d_url='http://10.0.0.2:8000',
            )

        assert result is not None
        assert '_dlrouter_request_id' in result
        assert result['_dlrouter_request_id'].startswith('___prefill_addr_')

        # Check that X-Request-Id header was set
        session = sess_ctx.__aenter__.return_value
        call_kwargs = session.post.call_args
        headers = call_kwargs.kwargs.get('headers', {})
        assert 'X-Request-Id' in headers

    async def test_prefill_request_without_d_url_no_request_id(self):
        """Test that prefill_request without d_url doesn't generate request_id."""
        body = b'{"id":"cmpl-1","choices":[{"message":{"content":"Hi"}}]}'
        sess_ctx, _ = _make_session_mock(status=200, body=body)

        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            result = await backend.prefill_request(
                'http://10.0.0.1:8000',
                '/v1/chat/completions',
                {'model': 'x', 'messages': [{'role': 'user', 'content': 'hi'}]},
            )

        assert result is not None
        assert '_dlrouter_request_id' not in result


class TestDecodeWithRequestId:
    """Test decode_request with request_id from prefill."""

    async def test_decode_request_uses_prefill_request_id(self):
        """Test that decode_request uses request_id from prefill_info."""
        body = b'{"id":"cmpl-2","choices":[{"message":{"content":"World"}}]}'
        sess_ctx, _ = _make_session_mock(status=200, body=body)

        prefill_info = {
            'id': 'cmpl-1',
            'choices': [{'message': {'content': 'Hello'}}],
            '_dlrouter_request_id': '___prefill_addr_10.0.0.1:8000___decode_addr_10.0.0.2:8000_abc123',
        }

        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            await backend.decode_request(
                'http://10.0.0.2:8000',
                '/v1/chat/completions',
                {'model': 'x', 'messages': [{'role': 'user', 'content': 'hi'}]},
                prefill_info,
                stream=False,
            )

        session = sess_ctx.__aenter__.return_value
        call_kwargs = session.post.call_args
        headers = call_kwargs.kwargs.get('headers', {})
        assert headers.get('X-Request-Id') == '___prefill_addr_10.0.0.1:8000___decode_addr_10.0.0.2:8000_abc123'

    async def test_decode_request_stream_uses_request_id(self):
        """Test that streaming decode_request uses request_id."""
        body = b'data: {"id":"cmpl-2"}\ndata: [DONE]'
        sess_ctx, _ = _make_session_mock(status=200, body=body)

        prefill_info = {
            '_dlrouter_request_id': '___prefill_addr_p___decode_addr_d_uuid',
        }

        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            gen = await backend.decode_request(
                'http://10.0.0.2:8000',
                '/v1/chat/completions',
                {'model': 'x', 'messages': []},
                prefill_info,
                stream=True,
            )
            # Consume the generator
            chunks = [chunk async for chunk in gen]
            assert len(chunks) > 0

        session = sess_ctx.__aenter__.return_value
        call_kwargs = session.post.call_args
        headers = call_kwargs.kwargs.get('headers', {})
        assert 'X-Request-Id' in headers

