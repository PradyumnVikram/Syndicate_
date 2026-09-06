#!/usr/bin/env python3
"""
Integration test for SEDS Phase B (Docker executor).

Demonstrates that the Docker executor produces RolloutResult from containerized rollout.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from seds.domains.base import Task, Score, AgentSystem
from seds.executor.runner import DockerExecutor


def setup_logging():
    """Configure logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


async def test_docker_executor():
    """Test Docker executor with a simple task."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create a simple task
    task = Task(
        task_id="test_task_001",
        inputs={
            "message": "Hello from Docker executor!",
            "timeout": 10,
        },
        reference="Hello world!",
    )

    # Create executor (broker socket will be checked during execution)
    executor = DockerExecutor(
        broker_socket_path="/broker/socket",
        max_workers=12,
    )

    logger.info("=" * 80)
    logger.info("Testing SEDS Phase B Docker Executor")
    logger.info("=" * 80)

    try:
        # Create agent system (minimal implementation)
        class SimpleAgent(AgentSystem):
            def forward(self, task_info: dict[str, Any]) -> tuple[str, list[Any]]:
                """Simple agent that echoes the input."""
                result = f"Processed: {task_info.get('message', '')}"
                return (result, [task_info])

        agent_system = SimpleAgent()

        # Run the task
        logger.info(f"Running task: {task.task_id}")
        logger.info(f"Inputs: {task.inputs}")

        rollout_result = await executor.run_batch(
            tasks=[task],
            agent_system=agent_system,
        )

        logger.info("=" * 80)
        logger.info("ROLLOUT RESULT")
        logger.info("=" * 80)
        logger.info(f"Task ID: {rollout_result.task_id}")
        logger.info(f"Answer: {rollout_result.answer}")
        logger.info(f"Score (correct): {rollout_result.score.correct}")
        logger.info(f"Score (partial): {rollout_result.score.partial}")
        logger.info(f"Score detail: {rollout_result.score.detail}")
        logger.info(f"Input tokens: {rollout_result.input_tokens}")
        logger.info(f"Output tokens: {rollout_result.output_tokens}")
        logger.info(f"Wall time: {rollout_result.wall_ms}ms")
        logger.info(f"Crashed: {rollout_result.crashed}")
        logger.info(f"Timed out: {rollout_result.timed_out}")
        logger.info(f"Error log: {rollout_result.error_log}")

        # Verify the result
        assert rollout_result.node_id == "executor"
        # Note: run_batch returns a batch RolloutResult, not an individual task result
        # The task_id is "batch_<timestamp>"
        assert rollout_result.score.correct is True  # Should succeed
        assert rollout_result.score.partial == 1.0
        assert rollout_result.wall_ms >= 0
        assert not rollout_result.crashed
        assert not rollout_result.timed_out
        assert rollout_result.answer is not None

        logger.info("=" * 80)
        logger.info("✓ Test PASSED: Phase B Docker executor working correctly")
        logger.info("=" * 80)

        return rollout_result

    except Exception as e:
        logger.error(f"Test FAILED: {e}", exc_info=True)
        logger.error("=" * 80)
        logger.error("✗ Test FAILED: Phase B Docker executor has issues")
        logger.error("=" * 80)
        raise


async def test_batch_execution():
    """Test batch execution of multiple tasks."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Testing Batch Execution (12 concurrent pool)")
    logger.info("=" * 80)

    # Create multiple tasks
    tasks = [
        Task(
            task_id=f"test_task_{i:03d}",
            inputs={"index": i, "message": f"Task {i}"},
            reference=f"Task {i}",
        )
        for i in range(3)
    ]

    # Create executor
    executor = DockerExecutor(
        broker_socket_path="/broker/socket",
        max_workers=12,
    )

    # Create simple agent system
    class SimpleAgent(AgentSystem):
        def forward(self, task_info: dict[str, Any]) -> tuple[str, list[Any]]:
            return (f"Processed: {task_info.get('message', '')}", [task_info])

    agent_system = SimpleAgent()

    try:
        # Run batch
        rollout_result = await executor.run_batch(
            tasks=tasks,
            agent_system=agent_system,
        )

        logger.info("=" * 80)
        logger.info("BATCH ROLLOUT RESULT")
        logger.info("=" * 80)
        logger.info(f"Task ID: {rollout_result.task_id}")
        logger.info(f"Successful: {rollout_result.score.correct}")
        logger.info(f"Partial: {rollout_result.score.partial}")
        logger.info(f"Total tasks: {rollout_result.score.detail.get('total', 0)}")
        logger.info(f"Successful tasks: {rollout_result.score.detail.get('successful', 0)}")
        logger.info(f"Failed tasks: {rollout_result.score.detail.get('failed', 0)}")
        logger.info(f"Total wall time: {rollout_result.wall_ms}ms")
        logger.info("=" * 80)

        # Verify batch results
        assert rollout_result.score.correct is True
        assert rollout_result.score.partial == 1.0
        assert rollout_result.score.detail.get("total") == len(tasks)
        assert rollout_result.score.detail.get("successful") == len(tasks)
        assert rollout_result.score.detail.get("failed") == 0

        logger.info("✓ Batch test PASSED")
        return rollout_result

    except Exception as e:
        logger.error(f"Batch test FAILED: {e}", exc_info=True)
        raise


async def main():
    """Run all integration tests."""
    logger = logging.getLogger(__name__)

    try:
        # Test 1: Single task
        logger.info("\n")
        result1 = await test_docker_executor()

        # Test 2: Batch execution
        logger.info("\n")
        result2 = await test_batch_execution()

        logger.info("\n")
        logger.info("=" * 80)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("Phase B (Docker executor) is fully functional")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("\n")
        logger.error("=" * 80)
        logger.error("TESTS FAILED ✗")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
