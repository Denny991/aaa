"""ZMQ-based service discovery for vLLM PD disaggregation.

This module provides automatic P/D node registration via ZMQ heartbeats,
compatible with the official vLLM disaggregated inference proxy.

vLLM P/D nodes send periodic heartbeat messages containing:
    {"type": "P"/"D", "http_address": "ip:port", "zmq_address": "ip:port"}

The ZMQServiceDiscovery listens for these messages and automatically
registers/deregisters nodes with the NodeManager.
"""

import threading
import time
from typing import Any, Optional

from dlrouter.constants import (
    EngineRole,
    ZMQ_DEFAULT_PING_SECONDS,
    ZMQ_DEFAULT_PORT,
)
from dlrouter.logger import get_logger
from dlrouter.models.node import NodeStatus


logger = get_logger('dlrouter.zmq_discovery')


class ZMQServiceDiscovery:
    """ZMQ-based service discovery for vLLM P/D nodes.

    Listens for heartbeat messages from P/D nodes and automatically
    registers them with the NodeManager. Nodes that stop sending
    heartbeats are automatically removed after expiration.

    Args:
        node_manager: The NodeManager instance to sync nodes to.
        hostname: ZMQ bind hostname. Default "0.0.0.0".
        port: ZMQ bind port. Default 30001.
        ping_seconds: Heartbeat expiration in seconds. Default 5.
    """

    def __init__(
        self,
        node_manager,
        hostname: str = '0.0.0.0',
        port: int = ZMQ_DEFAULT_PORT,
        ping_seconds: int = ZMQ_DEFAULT_PING_SECONDS,
    ) -> None:
        self._manager = node_manager
        self._hostname = hostname
        self._port = port
        self._ping_seconds = ping_seconds

        # Internal node tracking: http_address -> (zmq_address, expire_time)
        self._prefill_instances: dict[str, tuple[str, float]] = {}
        self._decode_instances: dict[str, tuple[str, float]] = {}

        # Lock for thread-safe access
        self._lock = threading.Lock()

        # Thread handles
        self._listener_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._running = False

        # Track which nodes we've registered (to avoid duplicate adds)
        self._registered_nodes: set[str] = set()

    def start(self) -> None:
        """Start the ZMQ service discovery.

        Starts two daemon threads:
        1. Listener thread: receives ZMQ heartbeat messages
        2. Sync thread: periodically syncs nodes to NodeManager
        """
        if self._listener_thread is not None:
            return

        self._running = True

        # Import ZMQ lazily to avoid hard dependency
        try:
            import msgpack  # noqa: F401
            import zmq  # noqa: F401
        except ImportError as e:
            logger.error(
                f'ZMQ service discovery requires pyzmq and msgpack. '
                f'Install with: pip install pyzmq msgpack. Error: {e}'
            )
            return

        # Start listener thread
        self._listener_thread = threading.Thread(
            target=self._run_listener,
            daemon=True,
            name='dlrouter-zmq-listener',
        )
        self._listener_thread.start()

        # Start sync thread
        self._sync_thread = threading.Thread(
            target=self._run_sync,
            daemon=True,
            name='dlrouter-zmq-sync',
        )
        self._sync_thread.start()

        logger.info(
            f'ZMQ service discovery started on tcp://{self._hostname}:{self._port} '
            f'(ping_seconds={self._ping_seconds})'
        )

    def stop(self) -> None:
        """Stop the ZMQ service discovery."""
        self._running = False
        self._listener_thread = None
        self._sync_thread = None
        logger.info('ZMQ service discovery stopped.')

    def _run_listener(self) -> None:
        """Run the ZMQ listener loop."""
        import msgpack
        import zmq

        context = zmq.Context()
        router_socket = context.socket(zmq.ROUTER)
        router_socket.bind(f'tcp://{self._hostname}:{self._port}')

        poller = zmq.Poller()
        poller.register(router_socket, zmq.POLLIN)

        try:
            self._listen_for_register(poller, router_socket)
        finally:
            router_socket.close()
            context.term()

    def _listen_for_register(self, poller, router_socket) -> None:
        """Listen for registration messages from P/D nodes.

        Message format (msgpack):
            {"type": "P" or "D", "http_address": "ip:port", "zmq_address": "ip:port"}
        """
        import msgpack

        while self._running:
            try:
                socks = dict(poller.poll(timeout=1000))  # 1 second timeout
                if router_socket not in socks:
                    continue

                remote_address, message = router_socket.recv_multipart()
                data = msgpack.loads(message)

                node_type = data.get('type')
                http_address = data.get('http_address')
                zmq_address = data.get('zmq_address')

                if not all([node_type, http_address, zmq_address]):
                    logger.warning(f'Invalid registration message: {data}')
                    continue

                expire_time = time.time() + self._ping_seconds

                with self._lock:
                    if node_type == 'P':
                        is_new = http_address not in self._prefill_instances
                        self._prefill_instances[http_address] = (zmq_address, expire_time)
                        self._remove_oldest_instances(self._prefill_instances)
                        if is_new:
                            logger.info(f'[PREFILL] Add [HTTP:{http_address}, ZMQ:{zmq_address}]')
                    elif node_type == 'D':
                        is_new = http_address not in self._decode_instances
                        self._decode_instances[http_address] = (zmq_address, expire_time)
                        self._remove_oldest_instances(self._decode_instances)
                        if is_new:
                            logger.info(f'[DECODE] Add [HTTP:{http_address}, ZMQ:{zmq_address}]')
                    else:
                        logger.warning(f'Unknown node type: {node_type} from {remote_address}')

            except Exception as e:
                logger.error(f'Error in ZMQ listener: {e}')

    def _remove_oldest_instances(self, instances: dict[str, Any]) -> None:
        """Remove expired instances from the dict."""
        now = time.time()
        expired = [k for k, (_, expire) in instances.items() if expire <= now]
        for key in expired:
            value = instances.pop(key, None)
            if value:
                logger.info(f'Remove expired [HTTP:{key}, ZMQ:{value[0]}]')

    def _run_sync(self) -> None:
        """Run the sync loop to update NodeManager."""
        while self._running:
            try:
                time.sleep(1)  # Sync every second
                self._sync_to_manager()
            except Exception as e:
                logger.error(f'Error in ZMQ sync: {e}')

    def _sync_to_manager(self) -> None:
        """Sync discovered nodes to NodeManager."""
        now = time.time()

        with self._lock:
            # Get current active nodes
            active_prefill = {
                f'http://{addr}': zmq_addr
                for addr, (zmq_addr, expire) in self._prefill_instances.items()
                if expire > now
            }
            active_decode = {
                f'http://{addr}': zmq_addr
                for addr, (zmq_addr, expire) in self._decode_instances.items()
                if expire > now
            }

        # Combine all active nodes
        all_active = set(active_prefill.keys()) | set(active_decode.keys())

        # Find nodes to add
        to_add = all_active - self._registered_nodes

        # Find nodes to remove
        to_remove = self._registered_nodes - all_active

        # Add new nodes
        for url in to_add:
            if url in active_prefill:
                role = EngineRole.PREFILL
                zmq_addr = active_prefill[url]
            else:
                role = EngineRole.DECODE
                zmq_addr = active_decode[url]

            status = NodeStatus(role=role)
            # Store zmq_address for potential use in request_id generation
            # This is stored but models will be discovered by backend.register_node()
            try:
                self._manager.add(url, status)
                self._registered_nodes.add(url)
                logger.info(f'Registered {role.name} node: {url} (ZMQ: {zmq_addr})')
            except Exception as e:
                logger.error(f'Failed to register node {url}: {e}')

        # Remove expired nodes
        for url in to_remove:
            try:
                self._manager.remove(url)
                self._registered_nodes.discard(url)
                logger.info(f'Deregistered node: {url}')
            except Exception as e:
                logger.error(f'Failed to deregister node {url}: {e}')

    @property
    def prefill_instances(self) -> dict[str, tuple[str, float]]:
        """Get current prefill instances (thread-safe copy)."""
        with self._lock:
            return dict(self._prefill_instances)

    @property
    def decode_instances(self) -> dict[str, tuple[str, float]]:
        """Get current decode instances (thread-safe copy)."""
        with self._lock:
            return dict(self._decode_instances)

    def get_zmq_address(self, http_address: str) -> Optional[str]:
        """Get the ZMQ address for a given HTTP address.

        Args:
            http_address: The HTTP address (e.g., "http://ip:port")

        Returns:
            The ZMQ address if found, None otherwise.
        """
        # Normalize: remove http:// prefix if present
        addr = http_address.replace('http://', '').replace('https://', '')

        with self._lock:
            if addr in self._prefill_instances:
                return self._prefill_instances[addr][0]
            if addr in self._decode_instances:
                return self._decode_instances[addr][0]
        return None
