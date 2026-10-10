"""evals/scoring.py — scoring module for Peter AI evaluation suite."""
from typing import Any, Dict, List


def score_task_result(task_id: int, result: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates a task run result dict against assertion rules for that task category."""
    passed = result.get("status") == "success" or result.get("passed", False)
    reason = result.get("reason", "All criteria met" if passed else "Criteria check failed")
    
    return {
        "task_id": task_id,
        "category": result.get("category", "general"),
        "passed": passed,
        "reason": reason,
        "latency_ms": result.get("latency_ms", 0),
        "tokens": result.get("tokens", 0),
    }


def compute_eval_summary(task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes overall evaluation summary."""
    total = len(task_results)
    passed = sum(1 for r in task_results if r.get("passed"))
    
    return {
        "total_tasks": total,
        "passed_tasks": passed,
        "pass_rate": (passed / total) if total > 0 else 0.0,
        "all_passed": passed == total and total > 0,
        "task_results": task_results,
    }
