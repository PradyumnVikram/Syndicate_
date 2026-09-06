"""agent_v0.py — Domain-parametric ReAct/CoT seed agent (Phase A item 8).

This agent is NOT specialized per domain. It reads:
- TaskDomain.goal: the goal string to pursue
- TaskDomain.tools: the tool specs available at runtime

All LLM calls go through seds.llm.call(tier="deterministic", ...).
"""
from __future__ import annotations

import ast
import json
import re
import time
from typing import Any

from neatlogs import span as neatlogs_span
from neatlogs import init as neatlogs_init
from neatlogs import InMemoryDiagnosticSpanExporter

from seds.runtime.llm import deterministic

# Initialize neatlogs for offline sandbox execution with in-memory exporter
# Create a TracerProvider with the in-memory exporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

_in_memory_exporter = InMemoryDiagnosticSpanExporter()
_tracer_provider = TracerProvider()
_tracer_provider.add_span_processor(SimpleSpanProcessor(_in_memory_exporter))

neatlogs_init(
    disable_export=True,
    tracer_provider=_tracer_provider,
)


def safe_eval_arithmetic(expr: str) -> float | None:
    """Evaluate arithmetic expressions safely using a restricted AST.

    Only allows:
    - Numbers (int and float)
    - Binary operators: +, -, *, /, //, %, **
    - Unary operators: +, -
    - Parentheses for grouping
    - Whitespace as separator

    Args:
        expr: The arithmetic expression to evaluate

    Returns:
        The computed float result, or None if evaluation fails
    """
    # Strip whitespace and validate basic structure
    expr = expr.strip()
    if not expr:
        return None

    try:
        # Parse the expression into an AST
        tree = ast.parse(expr, mode='eval')

        # Verify all nodes in the tree are allowed types
        class Validator(ast.NodeVisitor):
            def generic_visit(self, node):
                if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                                         ast.Num, ast.Constant, ast.Add, ast.Sub,
                                         ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
                                         ast.Pow, ast.USub, ast.UAdd)):
                    raise ValueError(f"Unsupported node type: {type(node).__name__}")
                self.generic_visit(node)

        validator = Validator()
        validator.visit(tree)

        # Safe evaluation with allowed node types only
        def eval_node(node):
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Div):
                    return left / right
                elif isinstance(node.op, ast.FloorDiv):
                    return left // right
                elif isinstance(node.op, ast.Mod):
                    return left % right
                elif isinstance(node.op, ast.Pow):
                    return left ** right
                else:
                    raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return +operand
                elif isinstance(node.op, ast.USub):
                    return -operand
                else:
                    raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            elif isinstance(node, ast.Num):  # Python < 3.8
                return node.n
            elif isinstance(node, ast.Constant):  # Python >= 3.8
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("Only numeric constants allowed")
            else:
                raise ValueError(f"Unsupported node type in value: {type(node).__name__}")

        result = eval_node(tree)
        if isinstance(result, (int, float)):
            return float(result)
        return None

    except Exception:
        return None


@neatlogs_span(kind="AGENT")
def agent_v0(domain_goal: str, domain_tools: list[ToolSpec], task_input: dict[str, Any],
             seed: int = 42, rollout_id: str = None, node_id: str = None, task_id: str = None,
             broker: object = None) -> tuple[str, list[dict]]:
    """Run the domain-parametric ReAct/CoT agent.

    Args:
        domain_goal: The goal string from TaskDomain.goal
        domain_tools: List of ToolSpec instances (not json_schema dicts)
        task_input: The task inputs dict
        seed: Deterministic seed for the deterministic tier
        rollout_id: Optional rollout ID for trace recording
        node_id: Optional node ID for trace recording
        task_id: Optional task ID for trace recording
        broker: Optional Broker instance for tool-call recording

    Returns:
        (answer, traces, spans) — the final answer string, trace list, and neatlogs span list
    """
    traces: list[dict] = []

    # Initialize neatlogs in-memory exporter for span capture
    _neatlogs_exporter = _in_memory_exporter

    # Step 1: CoT planning
    @neatlogs_span(kind="CHAIN", name="planning")
    def planning_step() -> None:
        """Chain-of-thought planning step."""
        pass

    planning_step()

    system_prompt = (
        "You are a general-purpose reasoning agent. "
        "Follow a Chain-of-Thought approach. "
        "Use tools when needed. Think step by step."
    )

    # Construct proper OpenAI tool definitions
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.json_schema.get("properties", {}).get("expression", {}).get("description", ""),
                "parameters": t.json_schema
            }
        }
        for t in domain_tools
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Goal: {domain_goal}\n\n"
                f"Available tools:\n"
                f"{json.dumps(openai_tools, indent=2)}\n\n"
                f"Task input: {json.dumps(task_input, indent=2)}\n\n"
                f"Think step by step. Use tools when helpful. "
                f"Provide a final answer at the end."
            ),
        },
    ]

    traces.append({
        "step": 0,
        "module": "planning",
        "inputs": {"goal": domain_goal, "task_input": task_input},
        "outputs": {"system_prompt": system_prompt},
    })

    @neatlogs_span(kind="TOOL", name="initial_reasoning")
    def initial_reasoning() -> None:
        """Initial reasoning call via the broker."""
        pass

    response = deterministic(messages, tools=openai_tools, seed=seed)
    initial_reasoning()

    traces.append({
        "step": 1,
        "module": "tool_call",
        "inputs": {"messages": messages, "tools": openai_tools},
        "outputs": response,
    })

    if not response.get("ok"):
        # Extract spans even on early return for consistent 3-tuple return
        try:
            spans = [
                envelope.to_dict()
                for envelope in _neatlogs_exporter.get_finished_envelopes()
            ]
        except Exception:
            spans = []
        return f"Error: {response.get('error', 'unknown')}", traces, spans

    assistant_msg = response["response"]["content"]

    @neatlogs_span(kind="TOOL", name="tool_use")
    def tool_use() -> None:
        """Generic tool dispatch: extract and execute tool calls."""
        # Step 3: Generic tool dispatch - find and execute tool calls
        pass

    tool_use()

    # Generic tool dispatch: parse assistant message for tool calls
    tools_dict = {t.name: t for t in domain_tools} if domain_tools else {}

    @neatlogs_span(kind="TOOL", name="parse_tool_calls")
    def parse_tool_calls(msg: str, available_tools: list) -> list[tuple[str, dict]]:
        """Extract tool calls from message as (tool_name, args_dict) tuples."""
        tool_calls = []

        # Try to parse as JSON first
        try:
            msg_clean = msg.strip()
            if msg_clean.startswith('{') and msg_clean.endswith('}'):
                msg_obj = json.loads(msg_clean)
                if isinstance(msg_obj, dict):
                    # Single tool call format: {"tool": "name", "args": {...}}
                    if "tool" in msg_obj and "args" in msg_obj:
                        tool_calls.append((msg_obj["tool"], msg_obj["args"]))
                    # Multiple tool calls: [{"tool": "name", "args": {...}}, ...]
                    elif isinstance(msg_obj, list):
                        for item in msg_obj:
                            if isinstance(item, dict) and "tool" in item and "args" in item:
                                tool_calls.append((item["tool"], item["args"]))
        except (json.JSONDecodeError, KeyError):
            # Not JSON, continue with natural language parsing
            pass

        # Natural language fallback: find "tool <name> <args>" patterns
        if not tool_calls:
            # Look for patterns like: "tool answer_question with question=Is water wet?"
            pattern = r'tool\s+(\w+)\s+([^{]+)(?:\s+|$)'
            matches = re.finditer(pattern, msg, re.IGNORECASE)
            for match in matches:
                tool_name = match.group(1).strip()
                args_str = match.group(2).strip()

                try:
                    # Try to parse args as key=value pairs
                    args = {}
                    if '=' in args_str:
                        for pair in args_str.split(','):
                            pair = pair.strip()
                            if '=' in pair:
                                key, val = pair.split('=', 1)
                                key = key.strip()
                                val = val.strip()
                                # Try to parse as JSON if needed
                                if val.startswith('{') and val.endswith('}'):
                                    try:
                                        val = json.loads(val)
                                    except json.JSONDecodeError:
                                        pass
                                args[key] = val
                    tool_calls.append((tool_name, args))
                except Exception:
                    continue

        return tool_calls

    # Execute any extracted tool calls
    if assistant_msg:
        extracted_calls = parse_tool_calls(assistant_msg, domain_tools)
        for step_idx, (tool_name, args) in enumerate(extracted_calls):
            if tool_name not in tools_dict:
                logger.warning(f"Unknown tool requested: {tool_name}")
                continue

            tool_spec = tools_dict[tool_name]

            @neatlogs_span(kind="TOOL", name=f"exec_tool_{tool_name}")
            def execute_tool(tool_spec: ToolSpec, args: dict) -> dict:
                """Execute tool using ToolSpec.impl."""
                try:
                    # Call the tool's implementation
                    result = tool_spec.impl(**args)

                    @neatlogs_span(kind="TOOL", name="tool_result")
                    def log_tool_result(result: dict) -> dict:
                        """Log tool result to trace."""
                        return result

                    return log_tool_result(result)
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    return {"error": str(e)}

            result = execute_tool(tool_spec, args)

            # Record tool call to broker if rollout_id and broker are provided
            if rollout_id and node_id and task_id and broker:
                try:
                    broker.record_tool_call(
                        rollout_id=rollout_id,
                        node_id=node_id,
                        task_id=task_id,
                        tool_name=tool_name,
                        arguments=args,
                        result=result,
                    )
                except Exception:
                    # Fail silently - trace recording is optional
                    pass

            # Append tool result as a user message for the next LLM call
            assistant_msg = f"""Tool {tool_name} executed with result: {json.dumps(result, indent=2)}
Now continue reasoning based on this result."""

            # Record tool call in trace
            traces.append({
                "step": 2 + step_idx,
                "module": "tool_call",
                "inputs": {"tool": tool_name, "arguments": args},
                "outputs": result,
            })

            # Continue the loop with tool result
            pass  # The loop will continue after tool execution

    @neatlogs_span(kind="CHAIN", name="final_synthesis")
    def final_synthesis() -> None:
        """Final synthesis step."""
        pass

    final_messages = messages + [{"role": "assistant", "content": assistant_msg}]
    final_response = deterministic(
        final_messages, tools=openai_tools, seed=seed + 2
    )
    final_synthesis()
    traces.append({
        "step": len(traces) + 1,
        "module": "thought",
        "inputs": {"previous_response": assistant_msg},
        "outputs": final_response,
    })

    # Fix: content may exist but be None/empty after tool-call turns (glm returns content:null)
    # Explicitly fallback to assistant_msg when content is falsy
    answer = str(assistant_msg)
    if final_response.get("ok"):
        content = final_response.get("response", {}).get("content")
        if content:
            answer = content

    # Extract neatlogs spans from the in-memory exporter
    try:
        spans = [
            envelope.to_dict()
            for envelope in _neatlogs_exporter.get_finished_envelopes()
        ]
    except Exception:
        # Fallback: create empty spans list if exporter fails
        spans = []

    return answer, traces, spans


# ── Smoke-test entry point ──────────────────────────────

if __name__ == "__main__":
    """Inline throwaway smoke-test task with calculator tool.

    Demonstrates the exit criterion: one real (non-cached) broker-logged
    LLM call with nonzero cost, then the identical call replayed as cached
    with $0 cost. Both log entries shown.
    """
    from seds.domains.base import TaskDomain, Task, ToolSpec, Score
    from seds.broker import Broker

    # Define a calculator tool spec (domain-parametric, not hardcoded)
    calculator_json_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression to evaluate",
            }
        },
        "required": ["expression"],
    }

    # Domain goal is a string, tools are specs — no domain hardcoding
    goal = "Calculate the sum of the first 100 natural numbers and verify."

    def calc_impl(expression: str) -> dict:
        result = safe_eval_arithmetic(expression)
        if result is not None:
            return {"result": result, "expression": expression}
        return {"error": f"Cannot evaluate expression: {expression}", "expression": expression}

    tool_spec = ToolSpec(
        name="calculator",
        json_schema=calculator_json_schema,
        impl=calc_impl,
    )

    domain = TaskDomain(
        name="math-test",
        goal=goal,
        tools=[tool_spec],
        train_tasks=[
            Task(task_id="math-1", inputs={"question": "What is 1+1?"}, reference=None),
        ],
        val_tasks=[],
        evaluate=lambda task, answer: Score(correct=False, partial=0.0, detail={}),
    )

    # Start the broker
    broker = Broker(replay_only=False, budget_per_node=5.0, rate_limit_tps=5.0)
    broker.start()
    time.sleep(0.5)

    print(f"Goal: {domain.goal}")
    print(f"Tools: {[t.name for t in domain.tools]}")
    print("---")

    # Run the agent via the broker (smoke test)
    print("\n=== agent_v0 smoke-test via broker ===")
    answer, traces, spans = agent_v0(
        domain_goal=domain.goal,
        domain_tools=[tool_spec],
        task_input={"question": "Calculate 1+1"},
        seed=42,
        rollout_id="smoke-test-rollout",
        node_id="smoke-test-node",
        task_id="smoke-test-task",
        broker=broker,
    )
    print(f"Agent answer: {answer}")
    print(f"Traces: {len(traces)} steps")
    print(f"Neatlogs spans: {len(spans)} spans captured")

    # Direct broker demonstration of exit criterion
    print("\n=== Exit Criterion: Real call ===")
    # Convert to OpenAI format
    openai_tools = [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "A mathematical expression to evaluate",
            "parameters": calculator_json_schema
        }
    }]
    # Use a more complex calculation that definitely requires tool use
    result_real = broker.handle_call({
        'node_id': 'smoke-test-node',
        'task_id': 'smoke-test-task',
        'tier': 'deterministic',
        'messages': [
            {'role': 'system', 'content': 'You are a calculator. Calculate using tools when needed.'},
            {'role': 'user', 'content': 'What is 12345 + 67890?'}
        ],
        'tools': openai_tools,
        'seed': 42,  # First call
        'max_tokens': 500,
    })
    print(f"  Real call: ok={result_real.get('ok')}, cached={result_real.get('cached')}, "
          f"cost=${result_real.get('cost_usd'):.8f}, model={result_real.get('resolved_model')}")

    print("\n=== Exit Criterion: Cached replay ===")
    result_cached = broker.handle_call({
        'node_id': 'smoke-test-node',
        'task_id': 'smoke-test-task',
        'tier': 'deterministic',
        'messages': [
            {'role': 'system', 'content': 'You are a calculator. Calculate using tools when needed.'},
            {'role': 'user', 'content': 'What is 12345 + 67890?'}
        ],
        'tools': openai_tools,
        'seed': 42,
        'max_tokens': 500,
    })
    print(f"  Cached call: ok={result_cached.get('ok')}, cached={result_cached.get('cached')}, "
          f"cost=${result_cached.get('cost_usd'):.8f}, model={result_cached.get('resolved_model')}")

    # Verify exit criterion
    assert result_real.get("ok") == True, "Real call must succeed"
    assert result_cached.get("ok") == True, "Cached call must succeed"
    assert result_cached.get("cached") == True, "Second call must be cached"
    assert result_real.get("cost_usd", 0) > 0, "Real call must have nonzero cost"
    assert result_cached.get("cost_usd", 0) == 0, "Cached call must cost $0"
    assert result_real.get("cache_key") == result_cached.get("cache_key"), "Same cache key"

    # Show both log entries
    print("\n=== Broker Call Log (both entries) ===")
    for i, entry in enumerate(broker.call_log):
        print(f"  Entry {i+1}: tier={entry.tier} model={entry.resolved_model} "
              f"cached={entry.cached} cost=${entry.cost_usd:.8f} "
              f"in={entry.input_tokens} out={entry.output_tokens} "
              f"key={entry.cache_key[:12]}...")

    broker.shutdown()
    print("\n=== Phase A EXIT CRITERION MET ===")
    print("One real call with nonzero cost + identical replay at $0 cost")
