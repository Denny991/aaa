"""Base backend interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Optional

from dlrouter.models.node import NodeStatus


class BaseBackend(ABC):
    """Abstract base class for inference backends.

    Each backend adapter knows how to communicate with
    a specific inference engine (lmdeploy, vllm, etc.).
    """

    @abstractmethod
    async def forward_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """Forward a request to the backend node.

        Args:
            node_url: The URL of the backend node.
            endpoint: API endpoint path.
            request_data: The request payload dict.
            stream: Whether to stream the response.

        Returns:
            Response text or async generator for stream.
        """

    @abstractmethod
    async def stream_forward(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Stream-forward request to backend node.

        Yields:
            Response chunks as bytes.
        """

    @abstractmethod
    def fetch_models(self, node_url: str) -> list[str]:
        """Fetch available model names from a node.

        Args:
            node_url: The URL of the backend node.

        Returns:
            List of available model names.
        """

    @abstractmethod
    async def check_health(self, node_url: str) -> bool:
        """Check health of a backend node.

        Args:
            node_url: The URL of the backend node.

        Returns:
            True if healthy, False otherwise.
        """

    def supports_pd_disagg(self) -> bool:
        """Whether this backend supports PD disagg."""
        return False

    def is_connected_pd(self, p_url: str, d_url: str) -> bool:
        """Check if a PD connection exists between two nodes."""
        return False

    async def connect_pd(self, p_url: str, d_url: str) -> None:
        """Establish a PD connection between two nodes."""
        raise NotImplementedError('This backend does not support PD disagg.')

    def shelf_prefill_session(self, p_url: str, d_url: str, session_id: str) -> None:
        """Shelf a prefill session (backend-specific bookkeeping)."""
        return

    def unshelf_prefill_session(self, p_url: str, d_url: str, session_id: str) -> None:
        """Unshelf a prefill session (backend-specific bookkeeping)."""
        return

    async def prefill_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        d_url: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Send a prefill-only request (PD disagg).

        Args:
            node_url: P node URL.
            endpoint: API endpoint path.
            request_data: Request payload.
            d_url: D node URL (optional, for request_id generation).

        Returns:
            Prefill result info dict, or None.
        """
        raise NotImplementedError('This backend does not support PD disagg.')

    async def decode_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        prefill_info: dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """Send a decode request with prefill info.

        Returns:
            Response text or async generator.
        """
        raise NotImplementedError('This backend does not support PD disagg.')

    @abstractmethod
    def deregister_node(self, node_url: str) -> None:
        """Cleanup when a node is removed.

        Called by NodeManager.remove() to let the backend
        release any resources tied to the node (e.g. PD
        connection pool entries).

        Args:
            node_url: URL of the node being removed.
        """

    def register_node(
        self,
        node_url: str,
        status: Optional[NodeStatus] = None,
    ) -> NodeStatus:
        """Register/discover a new node.

        Default implementation fetches models.

        Returns:
            NodeStatus with discovered info.
        """
        if status is None:
            status = NodeStatus()
        if not status.models:
            status.models = self.fetch_models(node_url)
        return status
