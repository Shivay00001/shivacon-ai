"""Tests for the general-purpose reasoning/planning router.

Verifies that the agent plans over the *actually registered* tools and
extracts parameters from arbitrary natural-language input — rather than
matching a fixed set of demo strings. Runs without a trained model
(model=None), exercising the deterministic router path.
"""

from agent.agent import OmniCoreAgent


def _action(agent, task):
    plan = agent.reasoning.plan(task, "", agent.tools.list_tools())
    assert isinstance(plan, list) and plan, "plan must be a non-empty list"
    return plan[0]["action"], plan[0].get("params", {})


def test_router_intents():
    agent = OmniCoreAgent(model=None)
    cases = [
        ("What is 37 times 12 plus 5?", "calculator"),
        ("compute the square root of 144 plus 15", "calculator"),
        ("Remember that my deployment region is ap-south-1", "remember"),
        ("What is my deployment region?", "recall"),
        ("recall the deployment region", "recall"),
        ("What time is it right now?", "get_time"),
        ('compare "a dog running" and "a puppy playing"', "compare_similarity"),
        ("search for the latest news about AI regulation", "web_search"),
        ("Tell me a joke about cats", "respond"),
    ]
    for task, expected in cases:
        action, _ = _action(agent, task)
        assert action == expected, f"{task!r} -> {action} (expected {expected})"


def test_calculator_extracts_and_computes():
    agent = OmniCoreAgent(model=None)
    _, params = _action(agent, "What is 37 times 12 plus 5?")
    result = agent.execute_tool("calculator", params)
    assert result.get("result") == 37 * 12 + 5


def test_memory_store_then_recall_roundtrip():
    agent = OmniCoreAgent(model=None)
    agent.long_memory.clear()

    _, store = _action(agent, "Remember that my deployment region is ap-south-1")
    agent.execute_tool("remember", store)

    _, query = _action(agent, "What is my deployment region?")
    out = agent.execute_tool("recall", query)
    assert out["results"], "recall should retrieve the stored memory"
    assert out["results"][0]["value"] == "ap-south-1"


def test_unknown_task_falls_back_to_respond():
    agent = OmniCoreAgent(model=None)
    action, _ = _action(agent, "Tell me a joke about cats")
    assert action == "respond"


if __name__ == "__main__":
    test_router_intents()
    test_calculator_extracts_and_computes()
    test_memory_store_then_recall_roundtrip()
    test_unknown_task_falls_back_to_respond()
    print("All reasoning router tests passed.")
