"""DLRouter CLI entry point.

Usage:
    python -m dlrouter [OPTIONS]
    dlrouter [OPTIONS]

Examples:
    # Start with defaults
    python -m dlrouter

    # Custom port and strategy
    python -m dlrouter --server_port 9000 \
        --routing_strategy round_robin

    # With PD disaggregation
    python -m dlrouter --serving_strategy distserve

    # With vLLM PD disaggregation and ZMQ service discovery
    python -m dlrouter \
        --backend vllm \
        --serving_strategy distserve \
        --zmq_discovery_enabled \
        --zmq_discovery_port 30001 \
        --server_port 10001
"""

import os
from typing import Literal, Optional, Union

import fire
import uvicorn

from dlrouter.api.app import create_app
from dlrouter.config import (
    BackendConfig,
    LMDeployPDConfig,
    RouterConfig,
    SSLConfig,
    ZMQDiscoveryConfig,
)
from dlrouter.constants import (
    BackendType,
    RoutingStrategy,
    ServingStrategy,
    ZMQ_DEFAULT_PING_SECONDS,
    ZMQ_DEFAULT_PORT,
)
from dlrouter.logger import get_logger


logger = get_logger('dlrouter')


def serve(
    server_name: str = '0.0.0.0',
    server_port: int = 8000,
    backend: Literal['lmdeploy', 'vllm'] = 'lmdeploy',
    routing_strategy: Literal[
        'round_robin',
        'random',
        'consistent_hash',
        'min_expected_latency',
        'min_observed_latency',
    ] = 'min_expected_latency',
    serving_strategy: Literal['hybrid', 'distserve'] = 'hybrid',
    api_keys: Optional[Union[list[str], str]] = None,
    ssl: bool = False,
    log_level: str = 'INFO',
    disable_cache_status: bool = False,
    config_path: Optional[str] = None,
    migration_protocol: str = 'RDMA',
    link_type: Literal['RoCE', 'IB'] = 'RoCE',
    with_gdr: bool = True,
    dummy_prefill: bool = False,
    # ZMQ service discovery options (for vLLM PD disaggregation)
    zmq_discovery_enabled: bool = False,
    zmq_discovery_hostname: str = '0.0.0.0',
    zmq_discovery_port: int = ZMQ_DEFAULT_PORT,
    zmq_discovery_ping_seconds: int = ZMQ_DEFAULT_PING_SECONDS,
):
    """Launch the DLRouter proxy server.

    Args:
        server_name: Bind address. Default 0.0.0.0.
        server_port: Listen port. Default 8000.
        backend: Inference backend type.
        routing_strategy: Request routing strategy.
        serving_strategy: Serving mode.
        api_keys: Optional API keys (comma-separated
            string or list).
        ssl: Enable SSL (requires SSL_KEYFILE and
            SSL_CERTFILE env vars).
        log_level: Logging level.
        disable_cache_status: Disable config caching.
        config_path: Path to config persistence file.
        migration_protocol: PD migration protocol.
        link_type: RDMA link type.
        with_gdr: Enable GPU Direct RDMA.
        dummy_prefill: Use dummy prefill for testing.
        zmq_discovery_enabled: Enable ZMQ service discovery
            for vLLM P/D nodes.
        zmq_discovery_hostname: ZMQ bind hostname.
        zmq_discovery_port: ZMQ bind port. Default 30001.
        zmq_discovery_ping_seconds: Heartbeat expiration seconds.
    """
    # Parse api_keys
    if isinstance(api_keys, str):
        api_keys = api_keys.split(',')

    # Build config
    config = RouterConfig(
        server_name=server_name,
        server_port=server_port,
        routing_strategy=RoutingStrategy(routing_strategy),
        serving_strategy=ServingStrategy(serving_strategy),
        backend=BackendConfig(type=BackendType(backend)),
        pd_config=LMDeployPDConfig(
            migration_protocol=migration_protocol,
            link_type=link_type,
            with_gdr=with_gdr,
            dummy_prefill=dummy_prefill,
        ),
        ssl=SSLConfig(enabled=ssl),
        zmq_discovery=ZMQDiscoveryConfig(
            enabled=zmq_discovery_enabled,
            hostname=zmq_discovery_hostname,
            port=zmq_discovery_port,
            ping_seconds=zmq_discovery_ping_seconds,
        ),
        api_keys=api_keys,
        log_level=log_level,
        cache_status=not disable_cache_status,
        config_path=config_path,
    )

    # Set log level
    logger.setLevel(log_level.upper())

    # Create app
    app = create_app(config)

    # SSL
    ssl_keyfile = None
    ssl_certfile = None
    if ssl:
        ssl_keyfile = os.environ.get('SSL_KEYFILE')
        ssl_certfile = os.environ.get('SSL_CERTFILE')

    # Launch
    uv_log = os.getenv('UVICORN_LOG_LEVEL', 'info').lower()
    uvicorn.run(
        app=app,
        host=server_name,
        port=server_port,
        log_level=uv_log,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )


def main():
    """CLI entry point."""
    fire.Fire(serve)


if __name__ == '__main__':
    main()
