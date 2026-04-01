"""vLLM backend adapter.

Supports standard OpenAI-compatible API forwarding and
PD disaggregation for vLLM inference engine.

For PD disaggregation, vLLM instances must be started with
``--kv-transfer-config`` specifying the KV connector and role
(``kv_producer`` for P nodes, ``kv_consumer`` for D nodes).
The connector (e.g. NixlConnector, MooncakeConnector) handles
the actual KV cache transfer between P and D instances.
DLRouter's role is to orchestrate the two-phase request flow.
"""

import copy
import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional

import aiohttp
import requests

from dlrouter.backends.base import BaseBackend
from dlrouter.constants import AIOHTTP_TIMEOUT, HEALTH_CHECK_TIMEOUT
from dlrouter.logger import get_logger


logger = get_logger('dlrouter.backends.vllm')


def _random_uuid() -> str:
    """Generate a random UUID hex string."""
    return uuid.uuid4().hex


class VLLMBackend(BaseBackend):
    """Backend adapter for vLLM inference engine.

    Handles standard OpenAI-compatible API forwarding and
    PD disaggregation via vLLM's kv-transfer-config mechanism.
    """

    def __init__(self, zmq_discovery=None) -> None:
        timeout_val = AIOHTTP_TIMEOUT
        self._timeout = aiohttp.ClientTimeout(total=timeout_val)
        # Optional ZMQ discovery for getting ZMQ addresses
        self._zmq_discovery = zmq_discovery

    # -- Core forwarding --

    async def forward_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        stream: bool = False,
        headers: Optional[dict[str, str]] = None,
    ) -> Any:
        """Forward request to vLLM node."""
        try:
            async with aiohttp.ClientSession() as sess:
                url = node_url + endpoint
                req_headers = headers or {}
                # Add authorization if available
                api_key = os.environ.get('OPENAI_API_KEY')
                if api_key and 'Authorization' not in req_headers:
                    req_headers['Authorization'] = f'Bearer {api_key}'
                async with sess.post(
                    url,
                    json=request_data,
                    headers=req_headers if req_headers else None,
                    timeout=self._timeout,
                ) as resp:
                    return await resp.text()
        except Exception as e:
            logger.error(f'Forward error: {e}')
            raise

    async def stream_forward(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
    ) -> AsyncIterator[bytes]:
        """Stream-forward request to vLLM node."""
        try:
            async with aiohttp.ClientSession() as sess:
                url = node_url + endpoint
                req_headers = headers or {}
                api_key = os.environ.get('OPENAI_API_KEY')
                if api_key and 'Authorization' not in req_headers:
                    req_headers['Authorization'] = f'Bearer {api_key}'
                async with sess.post(
                    url,
                    json=request_data,
                    headers=req_headers if req_headers else None,
                    timeout=self._timeout,
                ) as resp:
                    async for line in resp.content:
                        if line.strip():
                            yield line + b'\n\n'
        except Exception as e:
            logger.error(f'Stream error: {e}')
            raise

    def fetch_models(self, node_url: str) -> list[str]:
        """Fetch available models from vLLM node."""
        try:
            url = f'{node_url}/v1/models'
            headers = {'accept': 'application/json'}
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            models = [m['id'] for m in data.get('data', [])]
            return models
        except Exception as e:
            logger.error(f'Failed to fetch models from {node_url}: {e}')
            return []

    async def check_health(self, node_url: str) -> bool:
        """Check vLLM node health via async request."""
        try:
            url = f'{node_url}/health'
            timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)
            async with aiohttp.ClientSession() as sess, sess.get(url, timeout=timeout) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f'Failed to check health from {node_url}: {e}')
            return False

    def deregister_node(self, node_url: str) -> None:
        """No-op for vLLM (no explicit PD connection pool)."""

    # -- PD Disaggregation support --

    def supports_pd_disagg(self) -> bool:
        """vLLM supports PD disaggregation via kv-transfer-config."""
        return True

    def is_connected_pd(self, p_url: str, d_url: str) -> bool:
        """Always True: vLLM connector is configured at instance startup."""
        return True

    async def connect_pd(self, p_url: str, d_url: str) -> None:
        """No-op: vLLM P/D nodes connect via --kv-transfer-config at startup."""

    def set_zmq_discovery(self, zmq_discovery) -> None:
        """Set the ZMQ discovery instance for address lookup."""
        self._zmq_discovery = zmq_discovery

    def _get_zmq_address(self, http_address: str) -> str:
        """Get ZMQ address for an HTTP address, or return the HTTP address."""
        if self._zmq_discovery:
            zmq_addr = self._zmq_discovery.get_zmq_address(http_address)
            if zmq_addr:
                return zmq_addr
        # Fallback: extract host:port from http address
        return http_address.replace('http://', '').replace('https://', '')

    def _generate_request_id(self, p_url: str, d_url: str) -> str:
        """Generate request ID in official vLLM proxy format.

        Format: ___prefill_addr_{p_zmq}___decode_addr_{d_zmq}_{uuid}
        """
        p_zmq = self._get_zmq_address(p_url)
        d_zmq = self._get_zmq_address(d_url)
        return f'___prefill_addr_{p_zmq}___decode_addr_{d_zmq}_{_random_uuid()}'

    async def prefill_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        d_url: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Send prefill-only request to P node.

        Sends request with max_tokens=1 so vLLM completes the prefill
        phase and stores KV cache via the configured connector.  The
        connector then transfers the KV to the D node automatically.

        Args:
            node_url: P node URL
            endpoint: API endpoint
            request_data: Request payload
            d_url: D node URL (for request_id generation)

        Returns:
            Prefill response info dict with request_id, or None on error.
        """
        prefill_data = copy.deepcopy(request_data)
        prefill_data['max_tokens'] = 1
        prefill_data['max_completion_tokens'] = 1
        prefill_data['stream'] = False

        # Generate request_id in official format if d_url is provided
        headers = {}
        request_id = None
        if d_url:
            request_id = self._generate_request_id(node_url, d_url)
            headers['X-Request-Id'] = request_id

        try:
            text = await self.forward_request(
                node_url, endpoint, prefill_data, headers=headers if headers else None
            )
            result = json.loads(text)
            # Store the request_id for use in decode phase
            if request_id:
                result['_dlrouter_request_id'] = request_id
            return result
        except Exception as e:
            logger.error(f'Prefill request failed on {node_url}: {e}')
            return None

    async def decode_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        prefill_info: dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """Send decode request to D node.

        IMPORTANT: For vLLM PD disaggregation, we send the ORIGINAL request
        to the D node (not modified). The D node uses the same request_id
        from prefill phase to locate the pre-transferred KV cache via the
        configured connector (e.g., NixlConnector, MooncakeConnector).

        The KV cache transfer happens automatically through the connector,
        so we don't need to append any tokens or modify the messages.
        """
        # Use original request data (deep copy to avoid mutation)
        decode_data = copy.deepcopy(request_data)

        # Remove DLRouter internal fields that vLLM doesn't recognize
        # These fields are used by LMDeploy backend, not vLLM
        decode_data.pop('_prefill_url', None)

        # Use the same request_id from prefill phase - THIS IS THE KEY
        # vLLM uses this to match the prefilled KV cache on the D node
        headers = {}
        request_id = prefill_info.get('_dlrouter_request_id') if prefill_info else None
        if request_id:
            headers['X-Request-Id'] = request_id

        if stream:
            return self.stream_forward(node_url, endpoint, decode_data, headers=headers if headers else None)
        return await self.forward_request(node_url, endpoint, decode_data, headers=headers if headers else None)
