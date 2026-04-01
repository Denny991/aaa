"""Configuration models for DLRouter."""

from typing import Optional

from pydantic import BaseModel, Field

from dlrouter.constants import (
    BackendType,
    RoutingStrategy,
    ServingStrategy,
    ZMQ_DEFAULT_PING_SECONDS,
    ZMQ_DEFAULT_PORT,
)


class BackendConfig(BaseModel):
    """Configuration for an inference backend."""

    type: BackendType = BackendType.LMDEPLOY
    extra: dict = Field(default_factory=dict)


class LMDeployPDConfig(BaseModel):
    """LMDeploy PD disaggregation config."""

    migration_protocol: str = 'RDMA'
    link_type: str = 'RoCE'
    with_gdr: bool = True
    dummy_prefill: bool = False


class SSLConfig(BaseModel):
    """SSL configuration."""

    enabled: bool = False
    keyfile: Optional[str] = None
    certfile: Optional[str] = None


class ZMQDiscoveryConfig(BaseModel):
    """ZMQ service discovery config for vLLM PD disaggregation.

    When enabled, DLRouter listens for ZMQ heartbeat messages
    from vLLM P/D nodes and automatically registers them.
    """

    enabled: bool = False
    hostname: str = '0.0.0.0'
    port: int = ZMQ_DEFAULT_PORT
    ping_seconds: int = ZMQ_DEFAULT_PING_SECONDS


class RouterConfig(BaseModel):
    """Top-level router configuration."""

    server_name: str = '0.0.0.0'
    server_port: int = 8000
    routing_strategy: RoutingStrategy = RoutingStrategy.MIN_EXPECTED_LATENCY
    serving_strategy: ServingStrategy = ServingStrategy.HYBRID
    backend: BackendConfig = Field(default_factory=BackendConfig)
    pd_config: LMDeployPDConfig = Field(default_factory=LMDeployPDConfig)
    ssl: SSLConfig = Field(default_factory=SSLConfig)
    zmq_discovery: ZMQDiscoveryConfig = Field(default_factory=ZMQDiscoveryConfig)
    api_keys: Optional[list[str]] = None
    log_level: str = 'INFO'
    cache_status: bool = True
    config_path: Optional[str] = None
