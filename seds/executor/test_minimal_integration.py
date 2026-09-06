#!/usr/bin/env python3
"""
Minimal integration test for SEDS Phase B (Docker executor).

This test verifies the executor structure and imports without running Docker.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class AgentSystem:
    """Minimal agent system for testing."""
    def __init__(self):
        pass

    def forward(self, task_info: dict[str, Any]) -> tuple[str, list[Any]]:
        """Simple forward pass."""
        result = f"Processed: {task_info.get('message', '')}"
        return (result, [task_info])


async def test_imports():
    """Test that all imports work correctly."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Testing Imports (SEDS Phase B)")
    logger.info("=" * 80)

    try:
        # Test import of executor modules
        from seds.domains.base import Task, RolloutResult, Score, AgentSystem
        from seds.executor.runner import DockerExecutor
        from seds.executor.preflight import check_agent_code
        from seds.executor.tracedb import get_connection

        logger.info("✓ All imports successful")

        # Verify classes exist
        assert AgentSystem is not None
        assert Task is not None
        assert RolloutResult is not None
        assert Score is not None
        assert DockerExecutor is not None

        logger.info("✓ All classes imported correctly")

        # Create an executor instance
        executor = DockerExecutor(
            broker_socket_path="/broker/socket",
            max_workers=12,
        )

        logger.info("✓ DockerExecutor instantiated")

        # Verify executor attributes
        assert executor.broker_socket_path == "/broker/socket"
        assert executor.max_workers == 12
        assert executor.trace_flush_interval == 5
        assert executor.grace_period == 10

        logger.info("✓ Executor configuration correct")

        # Verify security constraints exist in _create_container_config
        container_args = executor._create_container_config(Task(
            task_id="test",
            inputs={},
            reference="test",
        ))

        # Check for critical security constraints
        assert "--network" in container_args and container_args[container_args.index("--network") + 1] == "none"
        assert "--read-only" in container_args
        assert "--memory" in container_args and container_args[container_args.index("--memory") + 1] == "768m"
        assert "--cpus" in container_args and container_args[container_args.index("--cpus") + 1] == "1"
        assert "--pids-limit" in container_args and container_args[container_args.index("--pids-limit") + 1] == "128"
        assert "--user" in container_args
        assert "--tmpfs" in container_args
        assert any("/broker/socket" in arg for arg in container_args)

        logger.info("✓ All security constraints verified")

        # Verify trace-loss guard attributes
        assert executor.trace_flush_interval == 5  # 5s flush interval
        assert executor.grace_period == 10  # 10s grace period

        logger.info("✓ Trace-loss guard attributes correct")
        logger.info("  - Flush interval: 5 seconds")
        logger.info("  - Grace period: 10 seconds")

        # Verify preflight check method exists
        task = Task(
            task_id="test",
            inputs={},
            reference="test",
        )
        preflight_pass, errors = await executor._check_preflight(task)
        assert isinstance(preflight_pass, bool)
        assert isinstance(errors, list)

        logger.info("✓ Preflight check method exists and is callable")

        # Verify trace flush loop method
        assert hasattr(executor, '_trace_flush_loop')
        logger.info("✓ Trace flush loop method exists")

        # Verify concurrent pool
        import concurrent.futures
        executor = DockerExecutor(broker_socket_path="/broker/socket", max_workers=12)
        assert executor._executor is not None
        assert isinstance(executor._executor, concurrent.futures.ThreadPoolExecutor)
        assert executor._executor._max_workers == 12

        logger.info("✓ Concurrent pool configured (12 workers)")

        logger.info("=" * 80)
        logger.info("ALL IMPORT TESTS PASSED ✓")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"Import test FAILED: {e}", exc_info=True)
        logger.error("=" * 80)
        logger.error("✗ Import Test Failed")
        logger.error("=" * 80)
        return False


async def test_docker_executor_creation():
    """Test that Docker executor can be instantiated and configured."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("Testing Docker Executor Creation")
    logger.info("=" * 80)

    try:
        from seds.domains.base import Task
        from seds.executor.runner import DockerExecutor

        # Test default configuration
        executor_default = DockerExecutor(broker_socket_path="/broker/socket")
        assert executor_default.max_workers == 12
        assert executor_default.trace_flush_interval == 5
        assert executor_default.grace_period == 10

        logger.info("✓ Default configuration correct")

        # Test custom configuration
        executor_custom = DockerExecutor(
            broker_socket_path="/broker/socket",
            max_workers=24,
            trace_flush_interval=10,
            grace_period=30,
        )
        assert executor_custom.max_workers == 24
        assert executor_custom.trace_flush_interval == 10
        assert executor_custom.grace_period == 30

        logger.info("✓ Custom configuration correct")

        # Test container config generation
        task = Task(
            task_id="test_task",
            inputs={"message": "Hello"},
            reference="Hello",
        )

        container_args = executor_custom._create_container_config(task)

        # Verify key security constraints
        security_checks = [
            ("Network isolation", "--network" in container_args and container_args[container_args.index("--network") + 1] == "none"),
            ("Read-only rootfs", "--read-only" in container_args),
            ("Memory limit", "--memory" in container_args and container_args[container_args.index("--memory") + 1] == "768m"),
            ("CPU limit", "--cpus" in container_args and container_args[container_args.index("--cpus") + 1] == "1"),
            ("PIDs limit", "--pids-limit" in container_args and container_args[container_args.index("--pids-limit") + 1] == "128"),
            ("Tmpfs mount", "--tmpfs" in container_args),
            ("Broker socket mount", any("/broker/socket" in arg for arg in container_args)),
        ]

        all_secure = True
        for check_name, check_result in security_checks:
            if check_result:
                logger.info(f"  ✓ {check_name}")
            else:
                logger.error(f"  ✗ {check_name}")
                all_secure = False

        assert all_secure, "Some security constraints missing"

        # Test task script generation
        script = executor_custom._generate_task_script(task)
        assert "TASK_ID=" in script
        assert "TASK_INPUTS=" in script
        assert task.task_id in script
        logger.info("✓ Task script generation correct")

        logger.info("=" * 80)
        logger.info("Docker Executor Creation Test PASSED ✓")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"Docker executor creation test FAILED: {e}", exc_info=True)
        logger.error("=" * 80)
        logger.error("✗ Creation Test Failed")
        logger.error("=" * 80)
        return False


async def test_security_constraints():
    """Verify all security constraints are enforced."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("Testing Security Constraints")
    logger.info("=" * 80)

    try:
        from seds.executor.runner import DockerExecutor
        from seds.domains.base import Task

        executor = DockerExecutor(broker_socket_path="/broker/socket")

        task = Task(
            task_id="security_test",
            inputs={},
            reference="test",
        )

        container_args = executor._create_container_config(task)

        # Critical security constraints
        constraints = [
            {
                "name": "Network isolation (--network=none)",
                "value": ["--network", "none"],
                "check": lambda args: args[args.index("--network") + 1] == "none",
            },
            {
                "name": "Read-only rootfs",
                "value": ["--read-only"],
                "check": lambda args: "--read-only" in args,
            },
            {
                "name": "No new privileges",
                "value": ["--security-opt", "no-new-privileges"],
                "check": lambda args: "--security-opt" in args and args[args.index("--security-opt") + 1] == "no-new-privileges",
            },
            {
                "name": "Non-root user",
                "value": ["--user", "1000:1000"],
                "check": lambda args: args[args.index("--user") + 1] == "1000:1000",
            },
            {
                "name": "Memory limit 768m",
                "value": ["--memory", "768m"],
                "check": lambda args: args[args.index("--memory") + 1] == "768m",
            },
            {
                "name": "CPU limit 1",
                "value": ["--cpus", "1"],
                "check": lambda args: args[args.index("--cpus") + 1] == "1",
            },
            {
                "name": "PIDs limit 128",
                "value": ["--pids-limit", "128"],
                "check": lambda args: args[args.index("--pids-limit") + 1] == "128",
            },
            {
                "name": "Tmpfs on /tmp",
                "value": ["--tmpfs", "/tmp:rw,noexec,nosuid"],
                "check": lambda args: "--tmpfs" in args and "/tmp:rw,noexec,nosuid" in args,
            },
            {
                "name": "Tmpfs on /work",
                "value": ["--tmpfs", "/work:rw,noexec,nosuid"],
                "check": lambda args: "--tmpfs" in args and "/work:rw,noexec,nosuid" in args,
            },
            {
                "name": "Broker socket bind-mount",
                "value": ["/broker/socket:/broker/socket:ro"],
                "check": lambda args: "/broker/socket:/broker/socket:ro" in args,
            },
        ]

        all_passed = True
        for constraint in constraints:
            if constraint["check"](container_args):
                logger.info(f"  ✓ {constraint['name']}")
            else:
                logger.error(f"  ✗ {constraint['name']}")
                all_passed = False

        assert all_passed, "Some security constraints failed"

        logger.info("=" * 80)
        logger.info("Security Constraints Test PASSED ✓")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"Security constraints test FAILED: {e}", exc_info=True)
        logger.error("=" * 80)
        logger.error("✗ Security Constraints Test Failed")
        logger.error("=" * 80)
        return False


async def main():
    """Run all integration tests."""
    logger = logging.getLogger(__name__)

    try:
        # Test 1: Imports
        import1 = await test_imports()

        # Test 2: Docker executor creation
        import2 = await test_docker_executor_creation()

        # Test 3: Security constraints
        import3 = await test_security_constraints()

        if import1 and import2 and import3:
            logger.info("\n" + "=" * 80)
            logger.info("ALL INTEGRATION TESTS PASSED ✓")
            logger.info("SEDS Phase B (Docker executor) is ready for use")
            logger.info("=" * 80)
            return 0
        else:
            logger.error("\n" + "=" * 80)
            logger.error("SOME TESTS FAILED ✗")
            logger.error("=" * 80)
            return 1

    except Exception as e:
        logger.error(f"Integration test framework failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
