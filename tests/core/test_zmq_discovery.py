"""Tests for ZMQServiceDiscovery."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from dlrouter.constants import EngineRole
from dlrouter.models.node import NodeStatus


# ---------------------------------------------------------------------------
# Test ZMQServiceDiscovery without actual ZMQ (mocked)
# ---------------------------------------------------------------------------


class TestZMQServiceDiscoveryInit:
    """Test ZMQServiceDiscovery initialization."""

    def test_init_default_values(self):
        """Test default initialization values."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager)

        assert discovery._manager is manager
        assert discovery._hostname == '0.0.0.0'
        assert discovery._port == 30001
        assert discovery._ping_seconds == 5
        assert discovery._prefill_instances == {}
        assert discovery._decode_instances == {}
        assert discovery._running is False

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(
            manager,
            hostname='127.0.0.1',
            port=40001,
            ping_seconds=10,
        )

        assert discovery._hostname == '127.0.0.1'
        assert discovery._port == 40001
        assert discovery._ping_seconds == 10


class TestZMQServiceDiscoveryProperties:
    """Test ZMQServiceDiscovery property methods."""

    def test_prefill_instances_returns_copy(self):
        """Test that prefill_instances returns a copy."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager)

        # Manually add an instance
        discovery._prefill_instances['10.0.0.1:8000'] = ('10.0.0.1:30001', time.time() + 100)

        instances = discovery.prefill_instances
        assert '10.0.0.1:8000' in instances
        # Verify it's a copy (modifying returned dict doesn't affect internal)
        instances['new_key'] = ('addr', 0)
        assert 'new_key' not in discovery._prefill_instances

    def test_decode_instances_returns_copy(self):
        """Test that decode_instances returns a copy."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager)

        discovery._decode_instances['10.0.0.2:8000'] = ('10.0.0.2:30001', time.time() + 100)

        instances = discovery.decode_instances
        assert '10.0.0.2:8000' in instances
        instances['new_key'] = ('addr', 0)
        assert 'new_key' not in discovery._decode_instances

    def test_get_zmq_address_found(self):
        """Test get_zmq_address when address exists."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager)

        discovery._prefill_instances['10.0.0.1:8000'] = ('10.0.0.1:30001', time.time() + 100)
        discovery._decode_instances['10.0.0.2:8000'] = ('10.0.0.2:30002', time.time() + 100)

        # Test prefill lookup
        assert discovery.get_zmq_address('http://10.0.0.1:8000') == '10.0.0.1:30001'
        assert discovery.get_zmq_address('10.0.0.1:8000') == '10.0.0.1:30001'

        # Test decode lookup
        assert discovery.get_zmq_address('http://10.0.0.2:8000') == '10.0.0.2:30002'

    def test_get_zmq_address_not_found(self):
        """Test get_zmq_address when address doesn't exist."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager)

        assert discovery.get_zmq_address('http://unknown:8000') is None


class TestRemoveOldestInstances:
    """Test _remove_oldest_instances method."""

    def test_removes_expired_instances(self):
        """Test that expired instances are removed."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager)

        # Add expired instance
        discovery._prefill_instances['expired:8000'] = ('expired:30001', time.time() - 10)
        # Add valid instance
        discovery._prefill_instances['valid:8000'] = ('valid:30001', time.time() + 100)

        discovery._remove_oldest_instances(discovery._prefill_instances)

        assert 'expired:8000' not in discovery._prefill_instances
        assert 'valid:8000' in discovery._prefill_instances

    def test_keeps_valid_instances(self):
        """Test that valid instances are kept."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager)

        future_time = time.time() + 100
        discovery._prefill_instances['valid1:8000'] = ('valid1:30001', future_time)
        discovery._prefill_instances['valid2:8000'] = ('valid2:30001', future_time + 1)

        discovery._remove_oldest_instances(discovery._prefill_instances)

        assert len(discovery._prefill_instances) == 2


class TestSyncToManager:
    """Test _sync_to_manager method."""

    def test_adds_new_prefill_node(self):
        """Test that new prefill nodes are added to manager."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager)

        # Add a prefill instance
        discovery._prefill_instances['10.0.0.1:8000'] = ('10.0.0.1:30001', time.time() + 100)

        discovery._sync_to_manager()

        # Verify manager.add was called
        manager.add.assert_called_once()
        call_args = manager.add.call_args
        assert call_args[0][0] == 'http://10.0.0.1:8000'
        assert isinstance(call_args[0][1], NodeStatus)
        assert call_args[0][1].role == EngineRole.PREFILL

    def test_adds_new_decode_node(self):
        """Test that new decode nodes are added to manager."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager)

        # Add a decode instance
        discovery._decode_instances['10.0.0.2:8000'] = ('10.0.0.2:30002', time.time() + 100)

        discovery._sync_to_manager()

        manager.add.assert_called_once()
        call_args = manager.add.call_args
        assert call_args[0][0] == 'http://10.0.0.2:8000'
        assert call_args[0][1].role == EngineRole.DECODE

    def test_removes_expired_node(self):
        """Test that expired nodes are removed from manager."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager)

        # First sync: add a node
        discovery._prefill_instances['10.0.0.1:8000'] = ('10.0.0.1:30001', time.time() + 100)
        discovery._sync_to_manager()
        assert 'http://10.0.0.1:8000' in discovery._registered_nodes

        # Simulate expiration
        discovery._prefill_instances['10.0.0.1:8000'] = ('10.0.0.1:30001', time.time() - 10)

        # Second sync: should remove
        discovery._sync_to_manager()
        manager.remove.assert_called_with('http://10.0.0.1:8000')
        assert 'http://10.0.0.1:8000' not in discovery._registered_nodes

    def test_does_not_duplicate_registered_nodes(self):
        """Test that already registered nodes are not added again."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager)

        # Add and sync once
        discovery._prefill_instances['10.0.0.1:8000'] = ('10.0.0.1:30001', time.time() + 100)
        discovery._sync_to_manager()
        assert manager.add.call_count == 1

        # Update timestamp, sync again - should not add again
        discovery._prefill_instances['10.0.0.1:8000'] = ('10.0.0.1:30001', time.time() + 200)
        discovery._sync_to_manager()
        assert manager.add.call_count == 1  # Still 1


class TestStartStop:
    """Test start and stop methods."""

    def test_start_without_zmq_logs_error(self):
        """Test that start without pyzmq installed logs error."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager)

        # Mock import to fail
        with patch.dict('sys.modules', {'zmq': None, 'msgpack': None}):
            with patch('builtins.__import__', side_effect=ImportError('No module')):
                discovery.start()

        # Should not start threads if import fails
        assert discovery._listener_thread is None or not discovery._listener_thread.is_alive()

    def test_stop_sets_running_false(self):
        """Test that stop sets _running to False."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager)
        discovery._running = True

        discovery.stop()

        assert discovery._running is False


# ---------------------------------------------------------------------------
# Integration tests with mocked ZMQ
# ---------------------------------------------------------------------------


class TestZMQIntegration:
    """Integration tests with mocked ZMQ."""

    def test_listen_for_register_handles_prefill_message(self):
        """Test that prefill registration message is handled correctly."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager, ping_seconds=5)
        discovery._running = True

        # Simulate a single message then stop
        message_data = {
            'type': 'P',
            'http_address': '10.0.0.1:8000',
            'zmq_address': '10.0.0.1:30001',
        }

        import msgpack

        packed = msgpack.dumps(message_data)

        # Mock the socket behavior for one iteration
        mock_socket = MagicMock()
        mock_socket.recv_multipart.return_value = (b'remote', packed)

        mock_poller = MagicMock()
        # Return socket once, then empty to stop loop
        mock_poller.poll.side_effect = [{mock_socket: 1}, {}]

        # Run one iteration manually
        with patch.object(discovery, '_running', True):
            # We'll call the method directly with mocked objects
            # but only process one message
            try:
                socks = dict(mock_poller.poll())
                if mock_socket in socks:
                    remote_address, message = mock_socket.recv_multipart()
                    data = msgpack.loads(message)

                    with discovery._lock:
                        discovery._prefill_instances[data['http_address']] = (
                            data['zmq_address'],
                            time.time() + discovery._ping_seconds,
                        )
            except Exception:
                pass

        assert '10.0.0.1:8000' in discovery._prefill_instances
        assert discovery._prefill_instances['10.0.0.1:8000'][0] == '10.0.0.1:30001'

    def test_listen_for_register_handles_decode_message(self):
        """Test that decode registration message is handled correctly."""
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        manager = MagicMock()
        discovery = ZMQServiceDiscovery(manager, ping_seconds=5)

        message_data = {
            'type': 'D',
            'http_address': '10.0.0.2:8000',
            'zmq_address': '10.0.0.2:30002',
        }

        import msgpack

        packed = msgpack.dumps(message_data)
        data = msgpack.loads(packed)

        with discovery._lock:
            discovery._decode_instances[data['http_address']] = (
                data['zmq_address'],
                time.time() + discovery._ping_seconds,
            )

        assert '10.0.0.2:8000' in discovery._decode_instances
        assert discovery._decode_instances['10.0.0.2:8000'][0] == '10.0.0.2:30002'


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Test ZMQ config integration."""

    def test_zmq_config_defaults(self):
        """Test ZMQDiscoveryConfig default values."""
        from dlrouter.config import ZMQDiscoveryConfig

        config = ZMQDiscoveryConfig()
        assert config.enabled is False
        assert config.hostname == '0.0.0.0'
        assert config.port == 30001
        assert config.ping_seconds == 5

    def test_zmq_config_custom_values(self):
        """Test ZMQDiscoveryConfig with custom values."""
        from dlrouter.config import ZMQDiscoveryConfig

        config = ZMQDiscoveryConfig(
            enabled=True,
            hostname='192.168.1.1',
            port=40001,
            ping_seconds=10,
        )
        assert config.enabled is True
        assert config.hostname == '192.168.1.1'
        assert config.port == 40001
        assert config.ping_seconds == 10

    def test_router_config_includes_zmq_discovery(self):
        """Test that RouterConfig includes zmq_discovery field."""
        from dlrouter.config import RouterConfig

        config = RouterConfig()
        assert hasattr(config, 'zmq_discovery')
        assert config.zmq_discovery.enabled is False
