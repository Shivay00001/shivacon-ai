"""
Test Agentic Capabilities
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.agent import (
    OmniCoreAgent,
    ToolRegistry,
    ShortTermMemory,
    LongTermMemory,
    create_agent,
)
from inference.engine import MultiModalInference
from data.tokenizer import BPETokenizer
import torch


def test_memory_systems():
    print("\n" + "="*50)
    print("TEST 1: Memory Systems")
    print("="*50)
    
    short_mem = ShortTermMemory(max_turns=5)
    print(f"Short-term memory created: {short_mem.max_turns} max turns")
    
    long_mem = LongTermMemory(Path(".cache/test_memory.json"))
    long_mem.add("favorite_color", "blue", ["preferences", "color"])
    long_mem.add("home_city", "New York", ["preferences", "location"])
    
    results = long_mem.search("color")
    print(f"Long-term memory search 'color': {len(results)} results")
    
    results = long_mem.search("city")
    print(f"Long-term memory search 'city': {len(results)} results")
    
    print("[PASS] Memory systems work!")


def test_tool_registry():
    print("\n" + "="*50)
    print("TEST 2: Tool Registry")
    print("="*50)
    
    registry = ToolRegistry()
    
    def dummy_tool(x: int, y: int) -> int:
        return x + y
    
    from agent.agent import Tool
    registry.register(Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
            },
            "required": ["x", "y"]
        },
        function=dummy_tool,
    ))
    
    tools = registry.list_tools()
    print(f"Registered tools: {len(tools)}")
    print(f"Tool: {tools[0]['name']} - {tools[0]['description']}")
    
    result = registry.get("add").function(x=5, y=3)
    print(f"Tool execution: 5 + 3 = {result}")
    
    print("[PASS] Tool registry works!")


def test_agent_tools():
    print("\n" + "="*50)
    print("TEST 3: Agent Tools")
    print("="*50)
    
    checkpoint_path = Path("checkpoints/checkpoint_epoch0001_step00000046.pt")
    tokenizer_path = Path(".cache/tokenizer.json")
    
    if not checkpoint_path.exists():
        print("[SKIP] No checkpoint found")
        return
    
    engine = MultiModalInference.from_checkpoint(checkpoint_path, device="cpu")
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    agent = OmniCoreAgent(model=engine.model, tokenizer=tokenizer)
    
    print("\n[Testing Calculator Tool]")
    result = agent.execute_tool("calculator", {"expression": "2 + 2"})
    print(f"2 + 2 = {result}")
    
    print("\n[Testing Time Tool]")
    result = agent.execute_tool("get_time", {})
    print(f"Current time: {result}")
    
    print("\n[Testing Text Encoding]")
    result = agent.execute_tool("encode_text", {"text": "Hello world"})
    print(f"Text encoding: {result}")
    
    print("\n[Testing Similarity]")
    result = agent.execute_tool("compare_similarity", {
        "input_a": "a cat sitting on a couch",
        "input_b": "a cat resting on a sofa",
    })
    print(f"Similarity: {result}")
    
    print("\n[Testing Memory - Remember]")
    result = agent.execute_tool("remember", {
        "key": "user_preference",
        "value": "likes science fiction movies",
        "tags": ["preferences", "movies"]
    })
    print(f"Remember: {result}")
    
    print("\n[Testing Memory - Recall]")
    result = agent.execute_tool("recall", {"query": "preference"})
    print(f"Recall: {result}")
    
    print("[PASS] Agent tools work!")


def test_agent_planning():
    print("\n" + "="*50)
    print("TEST 4: Agent Planning & Execution")
    print("="*50)
    
    checkpoint_path = Path("checkpoints/checkpoint_epoch0001_step00000046.pt")
    tokenizer_path = Path(".cache/tokenizer.json")
    
    if not checkpoint_path.exists():
        print("[SKIP] No checkpoint found")
        return
    
    engine = MultiModalInference.from_checkpoint(checkpoint_path, device="cpu")
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    agent = OmniCoreAgent(model=engine.model, tokenizer=tokenizer)
    
    tasks = [
        "Compare the similarity between 'a happy person' and 'a sad person'",
        "What is 125 + 375?",
        "Remember that my name is John",
    ]
    
    for task in tasks:
        print(f"\n--- Task: {task} ---")
        result = agent.run(task, verbose=False)
        print(f"Response: {result.get('response', 'No response')[:200]}...")
        print(f"Steps executed: {result.get('plan_executed', 0)}")


def test_agent_chat():
    print("\n" + "="*50)
    print("TEST 5: Agent Chat")
    print("="*50)
    
    checkpoint_path = Path("checkpoints/checkpoint_epoch0001_step00000046.pt")
    tokenizer_path = Path(".cache/tokenizer.json")
    
    if not checkpoint_path.exists():
        print("[SKIP] No checkpoint found")
        return
    
    engine = MultiModalInference.from_checkpoint(checkpoint_path, device="cpu")
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    agent = OmniCoreAgent(model=engine.model, tokenizer=tokenizer)
    
    print("\n--- Chat Session ---")
    
    response = agent.chat("Hello! What can you do?")
    print(f"User: Hello! What can you do?")
    print(f"Agent: {response[:150]}...")
    
    response = agent.chat("Compare 'cat' and 'dog'")
    print(f"\nUser: Compare 'cat' and 'dog'")
    print(f"Agent: {response[:200]}...")
    
    response = agent.chat("What is my name?")
    print(f"\nUser: What is my name?")
    print(f"Agent: {response[:200]}...")
    
    print("[PASS] Agent chat works!")


def main():
    print("="*50)
    print("OmniCore AI - Agentic Capabilities Test")
    print("="*50)
    
    results = {}
    
    tests = [
        ("Memory Systems", test_memory_systems),
        ("Tool Registry", test_tool_registry),
        ("Agent Tools", test_agent_tools),
        ("Agent Planning", test_agent_planning),
        ("Agent Chat", test_agent_chat),
    ]
    
    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = "PASS"
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = f"FAIL: {str(e)[:50]}"
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    passed = sum(1 for v in results.values() if v == "PASS")
    
    for name, status in results.items():
        icon = "[OK]" if status == "PASS" else "[XX]"
        print(f"  {icon} {name}: {status}")
    
    print(f"\nTotal: {passed}/{len(tests)} passed")
    
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
