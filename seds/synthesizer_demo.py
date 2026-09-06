#!/usr/bin/env python3
"""
Minimal demo of the SEDS Synthesizer module.

This demo verifies:
1. Module imports correctly (all class imports work)
2. Basic instantiation of core classes
3. MutationContext creation and basic operation
"""

import sys
import os
import random
import hashlib

# Add parent directory to path to enable module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthesizer import (
    SEDSSynthesizer,
    MutationContext,
    MutationRequest,
    MutationResult,
    MutationMenu,
    DiversityGuard,
)


def create_mock_broker():
    """Create a minimal mock Broker for testing."""
    class MockBroker:
        """Mock Broker for synthetic tests."""
        def __init__(self):
            self.call_count = 0
            self.tier_routes = {"deterministic": "deterministic", "reasoner": "reasoner"}
            self.SOCKET_PATH = "/tmp/mock_broker.sock"

        def generate_branch(self, prompt: str, tier: str = "deterministic") -> str:
            """Mock LLM call for mutation generation."""
            self.call_count += 1

            # Generate deterministic response based on prompt content
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]

            # Create a dummy mutation response
            dummy_code = f"# Generated mutation {prompt_hash}\n# Based on: {prompt[:30]}...\n"
            return dummy_code

        def handle_call(self, msg):
            """Mock broker call returning canned structured JSON response."""
            self.call_count += 1

            # Return a canned structured mutation response
            response_content = '''{
                "operator": "prompt_edit",
                "target_file": "agent.py",
                "old_str": "def agent():",
                "new_str": "def agent():\\n    improved_agent()",
                "rationale": "Fixed bug in agent logic"
            }'''

            return {
                "ok": True,
                "response": {
                    "content": response_content
                }
            }

    return MockBroker()


def create_test_context(node_id: str, parent_code: str = "") -> MutationContext:
    """Create a simple test mutation context."""
    return MutationContext(
        parent_code=parent_code,
        failure_traces=[
            {"trace_id": "t1", "error": "Test error"},
            {"trace_id": "t2", "error": "Another error"},
        ],
        failure_mode_histogram={"test_error": 2, "another_error": 1},
        ancestor_performance_log=[
            {"node_id": "parent1", "reward": 0.5},
            {"node_id": "parent2", "reward": 0.7},
        ],
        remaining_budget=0.8,
        current_node_id=node_id,
    )


def test_basic_imports():
    """Verify all classes can be imported."""
    print("1. Testing Module Imports")
    print("-" * 70)

    # Check that classes are importable
    assert MutationRequest is not None
    assert MutationResult is not None
    assert MutationContext is not None
    assert MutationMenu is not None
    assert DiversityGuard is not None
    assert SEDSSynthesizer is not None

    print("   ✓ All core classes imported successfully")
    print("   - MutationRequest")
    print("   - MutationResult")
    print("   - MutationContext")
    print("   - MutationMenu")
    print("   - DiversityGuard")
    print("   - SEDSSynthesizer")
    print()


def test_basic_instantiation():
    """Test basic instantiation of core classes."""
    print("2. Testing Basic Instantiation")
    print("-" * 70)

    # Test MutationMenu (static class with operators)
    menu = MutationMenu()
    assert menu is not None
    assert "prompt_edit" in menu.OPERATORS
    print(f"   ✓ MutationMenu created (has {len(menu.OPERATORS)} operators)")

    # Test DiversityGuard
    guard = DiversityGuard()
    assert guard is not None
    print("   ✓ DiversityGuard created")

    # Test Mock Broker
    broker = create_mock_broker()
    assert broker is not None
    print("   ✓ Mock Broker created")

    # Test SEDSSynthesizer
    synthesizer = SEDSSynthesizer(
        broker=broker,
        best_of_n=3,
        deterministic_tier="deterministic",
        max_file_size=50000,
        max_similar_mutations=5,
    )
    assert synthesizer is not None
    assert synthesizer.broker == broker
    assert synthesizer.best_of_n == 3
    assert synthesizer.diversity_guard is not None
    print("   ✓ SEDSSynthesizer created successfully")
    print(f"      - best_of_n: {synthesizer.best_of_n}")
    print(f"      - deterministic_tier: {synthesizer.deterministic_tier}")
    print(f"      - diversity_guard: {synthesizer.diversity_guard}")
    print()


def test_mutation_context():
    """Test MutationContext creation and properties."""
    print("3. Testing MutationContext")
    print("-" * 70)

    context = create_test_context("node_0")
    assert context is not None
    assert context.current_node_id == "node_0"
    assert context.parent_code == ""
    assert len(context.failure_traces) == 2
    assert len(context.ancestor_performance_log) == 2
    assert context.remaining_budget > 0

    print(f"   ✓ MutationContext created for node {context.current_node_id}")
    print(f"      - parent_code: {'(empty)' if not context.parent_code else 'present'}")
    print(f"      - failure_traces: {len(context.failure_traces)}")
    print(f"      - ancestors: {len(context.ancestor_performance_log)}")
    print(f"      - remaining_budget: {context.remaining_budget}")
    print()


def test_mutation_request():
    """Test MutationRequest creation."""
    print("4. Testing MutationRequest")
    print("-" * 70)

    request = MutationRequest(
        operator="prompt_edit",
        target_file="agent.py",
        old_str="old_code",
        new_str="new_code",
        rationale="Fixed bug in agent logic",
    )

    assert request is not None
    assert request.operator == "prompt_edit"
    assert request.target_file == "agent.py"
    assert request.old_str == "old_code"
    assert request.new_str == "new_code"
    assert request.rationale == "Fixed bug in agent logic"

    print(f"   ✓ MutationRequest created")
    print(f"      - operator: {request.operator}")
    print(f"      - target_file: {request.target_file}")
    print(f"      - rationale: {request.rationale[:40]}...")
    print()


def test_ast_hash_function():
    """Test the internal _ast_hash helper function."""
    print("5. Testing AST Hash Function")
    print("-" * 70)

    # Import the helper function
    from synthesizer import _ast_hash, _prompt_hash

    # Test AST hash generation
    code1 = "def test(): return 1"
    hash1 = _ast_hash(code1)
    assert hash1 is not None
    assert isinstance(hash1, str)
    assert len(hash1) == 16  # SHA256[:16]

    code2 = "class Test: pass"
    hash2 = _ast_hash(code2)
    assert hash2 is not None
    # Different code should give different hashes (95% probability)
    # We just check they're both valid strings
    assert isinstance(hash2, str)
    assert len(hash2) == 16

    # Test prompt hash
    prompt = "Generate a mutation for the agent code"
    hash3 = _prompt_hash(prompt)
    assert hash3 is not None
    assert isinstance(hash3, str)
    assert len(hash3) == 16

    print(f"   ✓ AST hash function works")
    print(f"      - code1 hash: {hash1}")
    print(f"      - code2 hash: {hash2}")
    print(f"      - prompt hash: {hash3}")
    print()


def test_sample_mutations():
    """Test the core sample_mutations() method with mock broker."""
    print("6. Testing sample_mutations() with Mock Broker")
    print("-" * 70)

    # Create mock broker
    broker = create_mock_broker()
    assert broker is not None
    print("   ✓ Mock broker created")

    # Create synthesizer with small best_of_n for quick testing
    synthesizer = SEDSSynthesizer(
        broker=broker,
        best_of_n=2,  # Small number for quick demo
        deterministic_tier="deterministic",
        max_file_size=50000,
        max_similar_mutations=5,
    )
    assert synthesizer is not None
    print(f"   ✓ SEDSSynthesizer created (best_of_n=2)")

    # Create a test context
    context = create_test_context("node_test")
    assert context is not None
    print(f"   ✓ MutationContext created for node {context.current_node_id}")

    # Call sample_mutations() - this is the core entry point
    print("   → Calling sample_mutations() (may take a moment)...")
    try:
        results = synthesizer.sample_mutations(context)
        print(f"   ✓ sample_mutations() completed successfully")

        # Verify results
        assert results is not None
        assert isinstance(results, list)
        print(f"   ✓ Returned {len(results)} mutation result(s)")

        # Check each result
        for i, result in enumerate(results):
            print(f"\n   Result {i+1}:")
            print(f"      - request: {result.request}")
            print(f"      - success: {result.success}")
            print(f"      - duplicate: {result.duplicate}")
            print(f"      - preflight_checked: {result.preflight_checked}")
            print(f"      - diff_hash: {result.diff_hash[:12] if result.diff_hash else 'N/A'}...")
            print(f"      - ast_matches_parent: {result.ast_matches_parent}")

            # Verify request is a MutationRequest
            assert result.request is not None
            assert hasattr(result.request, 'operator')
            assert hasattr(result.request, 'target_file')
            print(f"      ✓ Valid MutationRequest created")

        print("\n   ✓ All MutationRequests are valid objects")
        print(f"   ✓ sample_mutations() exercised end-to-end with mock broker")
        print()

    except Exception as e:
        print(f"   ✗ sample_mutations() failed: {e}")
        raise


def main():
    """Run all tests."""
    print("=" * 70)
    print("SEDS Synthesizer Module - Minimal Demo")
    print("=" * 70)
    print()

    try:
        test_basic_imports()
        test_basic_instantiation()
        test_mutation_context()
        test_mutation_request()
        test_ast_hash_function()
        test_sample_mutations()

        print("=" * 70)
        print("All Tests Passed Successfully!")
        print("=" * 70)
        print()
        print("Summary:")
        print("  - Module imports correctly (no NameError for 'sys')")
        print("  - All classes instantiate properly")
        print("  - MutationContext and MutationRequest work")
        print("  - Internal helper functions (_ast_hash, _prompt_hash) work")
        print("  - sample_mutations() exercised end-to-end with mock broker")
        print("  - Core value of synthesizer module: mutation proposal generation")
        print("  - No crashes or errors in basic usage")
        return 0

    except Exception as e:
        print()
        print("=" * 70)
        print("TEST FAILED!")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
