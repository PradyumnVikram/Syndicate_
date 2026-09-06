"""SEDS Phase D: Semantic Saliency Folding (SSF).

Provides semantic saliency folding for compression of execution traces by:
- Identifying diagnostic content (diffs, errors) that must be preserved
- Folding non-diagnostic content into lightweight placeholders
- Achieving high compression ratios for large text blocks
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Literal


# Patterns
DIFF_PATTERN = re.compile(r'^---|\+\+\+|^@@', re.MULTILINE)

# Failure-indicative keywords (based on failure taxonomy)
# Using word boundaries with context phrases to avoid false positives
FAILURE_KEYWORD_PATTERNS = [
    r'\berror\s+message\b',   # "error message" (context)
    r'\berror\s+code\b',      # "error code" (context)
    r'\berror\s+type\b',      # "error type" (context)
    r'\berror\s+object\b',    # "error object" (context)
    r'\bfailed\s+assertion\b', # "failed assertion" (context)
    r'\bfailed\s+test\b',     # "failed test" (context)
    r'\bexception\s+trace\b', # "exception trace" (context)
    r'\btimeout\s+error\b',   # "timeout error" (context)
    r'\bcrash\s+report\b',    # "crash report" (context)
    r'\bvalidation\s+error\b', # "validation error" (context)
    r'\binvalid\s+value\b',   # "invalid value" (context)
    r'\bstack\s+trace\b',     # "stack trace" (context)
    # Individual error types and keywords (catch unstructured error messages)
    r'\bTypeError\b',         # Python type error
    r'\bAssertionError\b',   # Assertion failure
    r'\bModuleNotFoundError\b', # Module not found
    r'\bValueError\b',        # Value error
    r'\bKeyError\b',          # Key error
    r'\bAttributeError\b',    # Attribute error
    r'\bMemoryError\b',       # Memory error
    r'\bIOError\b',           # I/O error
    r'\bOSError\b',           # OS error
    r'\bPermissionError\b',   # Permission denied
    r'\bFileNotFoundError\b', # File not found
    r'\bRuntimeError\b',      # Runtime error
    r'\bSyntaxError\b',       # Syntax error
    r'\bIndentationError\b',  # Indentation error
    r'\bNameError\b',         # Name error
    r'\bZeroDivisionError\b', # Division by zero
    r'\bStopIteration\b',     # Stop iteration
    # Traceback markers (specific, not generic)
    r'\bTraceback',           # Traceback marker (line starts with)
    r'\bException\s+raised\b', # Exception raised (specific phrase)
    r'\bTraceback\s+\(most\s+recent\s+call\s+last\)', # Full traceback header
]


@dataclass
class SaliencyFoldedSpan:
    """Folded span after applying semantic saliency folding.

    Attributes:
        span_id: Original span identifier
        kind: Type of span (e.g., "call", "error", "system")
        content_type: Either "diagnostic" (must preserve) or "compressed" (can fold)
        original_content: Original content before compression
        compressed_content: Content after compression (same for diagnostic, placeholder for compressed)
        metadata: Additional information about the folding decision
    """
    span_id: str
    kind: str
    content_type: Literal["diagnostic", "compressed"]
    original_content: str
    compressed_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticSaliencyFolder:
    """Folds execution traces by preserving diagnostic content and compressing non-diagnostic content.

    Uses regex patterns to identify:
    - Code diff markers (---, +++, @@)
    - Failure-indicative keywords

    Non-diagnostic content is replaced with lightweight JSON placeholders to achieve high compression ratios.
    """

    def __init__(self, min_compression_ratio: float = 10.0):
        """
        Initialize the semantic saliency folder.

        Args:
            min_compression_ratio: Minimum target compression ratio (default 10×)
        """
        self.min_compression_ratio = min_compression_ratio

    def fold_span(self, span: dict[str, Any]) -> SaliencyFoldedSpan:
        """
        Fold a single span based on diagnostic content.

        Extracts content from outputs.content field and checks for diagnostic markers.

        Args:
            span: TelemetrySpanV2-shaped span dict

        Returns:
            Folded span with diagnostic content preserved
        """
        span_id = span.get("span_id", "")
        kind = span.get("kind", "unknown")

        # Extract content from outputs.content field (primary location)
        content = None
        outputs = span.get("outputs", {})
        if isinstance(outputs, dict):
            content = outputs.get("content")

        # Fallback: check direct content field
        if content is None:
            content = span.get("content")

        # Fallback: check inputs.content
        if content is None:
            inputs = span.get("inputs", {})
            if isinstance(inputs, dict):
                content = inputs.get("content")

        # Convert to string if None
        if content is None:
            span_text = ""
        elif isinstance(content, str):
            span_text = content
        elif isinstance(content, dict):
            span_text = json.dumps(content, default=str)
        else:
            span_text = str(content)

        # Also extract error and traceback from top-level span fields (primary for failures)
        # This is critical because real agent failures store errors under span["error"]/["traceback"]
        # not in outputs.content
        error = span.get("error")
        traceback = span.get("traceback")

        # Concatenate error/traceback to span text if present
        if error:
            if span_text:
                span_text = span_text + "\n\n[ERROR]\n" + str(error)
            else:
                span_text = str(error)

        if traceback:
            if span_text:
                span_text = span_text + "\n\n[TRACEBACK]\n" + str(traceback)
            else:
                span_text = str(traceback)

        # Check if this span contains diagnostic content
        if self._is_diagnostic_content(span_text):
            folded_span = SaliencyFoldedSpan(
                span_id=span_id,
                kind=kind,
                content_type='diagnostic',
                original_content=span_text,
                compressed_content=span_text,  # No compression for diagnostic content
                metadata={"reason": "contains_diff_or_error"},
            )
        else:
            # Compress non-diagnostic content
            folded_span = SaliencyFoldedSpan(
                span_id=span_id,
                kind=kind,
                content_type='compressed',
                original_content=span_text,
                compressed_content=self._create_placeholder(span_text),
                metadata={"reason": "compressed_to_placeholder"},
            )

        return folded_span

    def fold_trace(self, trace: list[dict[str, Any]]) -> tuple[list[SaliencyFoldedSpan], dict[str, Any]]:
        """
        Fold an entire trace by folding each span and generate a report.

        Args:
            trace: List of TelemetrySpanV2-shaped span dicts

        Returns:
            Tuple of (folded_spans, report)
        """
        folded_spans = []
        for span in trace:
            folded = self.fold_span(span)
            folded_spans.append(folded)

        report = self.generate_report(folded_spans)

        return folded_spans, report

    def fold_spans_only(self, trace: list[dict[str, Any]]) -> list[SaliencyFoldedSpan]:
        """
        Fold an entire trace by folding each span only (no report).

        Args:
            trace: List of TelemetrySpanV2-shaped span dicts

        Returns:
            List of folded spans
        """
        folded_spans = []
        for span in trace:
            folded = self.fold_span(span)
            folded_spans.append(folded)

        return folded_spans

    def _is_diagnostic_content(self, text: str) -> bool:
        """
        Determine if text contains diagnostic content.

        A text block is considered diagnostic if it contains:
        - Code diff markers
        - Failure-indicative keywords (with word boundaries to avoid false positives)

        Args:
            text: Text to check

        Returns:
            True if text contains diagnostic content
        """
        # Check for diff markers
        if DIFF_PATTERN.search(text):
            return True

        # Check for failure keywords (using regex with word boundaries)
        # This prevents false positives like "error keywords" in documentation
        for pattern in FAILURE_KEYWORD_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _create_placeholder(self, original_text: str) -> str:
        """
        Create a lightweight JSON placeholder for compressed content.

        Args:
            original_text: Original text content

        Returns:
            JSON string placeholder
        """
        placeholder = {
            "type": "compressed_trace",
            "length": len(original_text),
            "span_id": hash(original_text) % 1000000,  # Simple hash
            "summary": original_text[:50] + "..." if len(original_text) > 50 else original_text,
        }
        return json.dumps(placeholder)

    def generate_report(self, folded_spans: list[SaliencyFoldedSpan]) -> dict[str, Any]:
        """
        Generate a compression report.

        Args:
            folded_spans: List of folded spans

        Returns:
            Dictionary with compression metrics
        """
        total_spans = len(folded_spans)
        diagnostic_spans = sum(1 for s in folded_spans if s.content_type == 'diagnostic')
        compressed_spans = total_spans - diagnostic_spans

        total_original_bytes = sum(len(s.original_content) for s in folded_spans)
        total_compressed_bytes = sum(len(s.compressed_content) for s in folded_spans)

        # Compression ratio: original_bytes / compressed_bytes (e.g., 10× means original is 10x larger)
        # This matches the intuitive interpretation of "10× compression" (file reduced to 1/10th size)
        compression_ratio = (
            total_original_bytes / total_compressed_bytes
            if total_compressed_bytes > 0 and total_original_bytes > 0
            else 1.0
        )

        size_reduction_percent = (
            (total_original_bytes - total_compressed_bytes) / total_original_bytes * 100
            if total_original_bytes > 0
            else 0.0
        )

        met_target = compression_ratio >= self.min_compression_ratio

        report = {
            "total_spans": total_spans,
            "diagnostic_spans": diagnostic_spans,
            "compressed_spans": compressed_spans,
            "total_original_bytes": total_original_bytes,
            "total_compressed_bytes": total_compressed_bytes,
            "compression_ratio": compression_ratio,
            "size_reduction_percent": size_reduction_percent,
            "min_compression_ratio_target": self.min_compression_ratio,
            "met_target": met_target,
        }

        return report

    def fold_and_report(self, trace: list[dict[str, Any]]) -> tuple[list[SaliencyFoldedSpan], dict[str, Any]]:
        """
        Fold a trace and generate a compression report.

        Args:
            trace: List of TelemetrySpanV2-shaped span dicts

        Returns:
            Tuple of (folded_spans, report)
        """
        folded_spans = self.fold_trace(trace)
        report = self.generate_report(folded_spans)

        return folded_spans, report


# ============================================================================
# Debug helper for troubleshooting compression
# ============================================================================

def debug_fold_spans(
    trace: list[dict[str, Any]],
    max_preview_length: int = 200,
    verbose: bool = True,
) -> None:
    """
    Debug helper to print what's being compressed and why.

    Args:
        trace: Trace to debug
        max_preview_length: Maximum length of content preview to print
        verbose: Whether to print detailed information
    """
    folder = SemanticSaliencyFolder()

    print("\n🔍 SSF Compression Debug Report:")
    print(f"{'='*60}")

    for span in trace:
        span_id = span.get("span_id", "unknown")
        kind = span.get("kind", "unknown")

        # Extract content
        outputs = span.get("outputs", {})
        content = outputs.get("content", span.get("content", ""))

        # Convert to string
        if isinstance(content, dict):
            content_str = json.dumps(content, default=str)
        else:
            content_str = str(content)

        # Check diagnostic
        is_diagnostic = folder._is_diagnostic_content(content_str)

        # Print info
        content_preview = content_str[:max_preview_length] + "..." if len(content_str) > max_preview_length else content_str
        compressed_preview = "..." if is_diagnostic else json.loads(folder._create_placeholder(content_str))["summary"]

        print(f"\nSpan: {span_id} [{kind}]")
        print(f"  Type: {'DIAGNOSTIC' if is_diagnostic else 'COMPRESSED'}")
        print(f"  Original length: {len(content_str):,} bytes")
        print(f"  Compressed length: {len(compressed_preview):,} bytes")
        print(f"  Preview: {content_preview}")


if __name__ == "__main__":
    # Simple test
    print("SEDS Phase D: SSF Demo")

    # Create test span
    test_span = {
        "span_id": "test_001",
        "kind": "call",
        "outputs": {
            "content": "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,4 @@\n print('hello')",
        },
    }

    folder = SemanticSaliencyFolder()
    folded = folder.fold_span(test_span)

    print(f"\nFolded {folded.span_id}:")
    print(f"  Type: {folded.content_type}")
    print(f"  Original: {len(folded.original_content)} bytes")
    print(f"  Compressed: {len(folded.compressed_content)} bytes")
    print(f"  Ratio: {len(folded.compressed_content) / len(folded.original_content):.2f}×")
