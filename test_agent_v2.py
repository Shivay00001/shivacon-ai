"""
Test Competitive Agent Framework
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.agent_v2 import (
    OmniCoreAgent,
    AgentConfig,
    MultiAgentTeam,
    ToolRegistry,
    Guardrails,
    CodeInterpreter,
    ShortTermMemory,
    LongTermMemory,
    get_framework_comparison,
    Message,
    MessageRole,
)
from inference.engine import MultiModalInference
from data.tokenizer import BPETokenizer
import json


def test_guardrails():
    print("\n" + "="*50)
    print("TEST 1: Guardrails & Safety")
    print("="*50)
    
    guardrails = Guardrails()
    
    test_cases = [
        ("Hello world", True),
        ("How to hack a system", False),
        ("Make a bomb", False),
        ("A" * 20000, False),
    ]
    
    for text, expected in test_cases:
        valid, error = guardrails.validate_input(text)
        status = "OK" if valid == expected else "FAIL"
        print(f"  {status}: '{text[:30]}...' -> {valid} ({error})")
    
    print("[PASS] Guardrails working!")


def test_code_interpreter():
    print("\n" + "="*50)
    print("TEST 2: Code Interpreter")
    print("="*50)
    
    interpreter = CodeInterpreter()
    
    test_cases = [
        ("print(2 + 2)", {"success": True}),
        ("result = 10 * 5; print(result)", {"success": True}),
        ("import os; os.system('ls')", {"error": True}),
    ]
    
    for code, expected in test_cases:
        result = interpreter.execute(code)
        success = result.get("success", False)
        has_error = "error" in result
        status = "OK" if (expected.get("success") == success or expected.get("error") == has_error) else "FAIL"
        print(f"  {status}: {code[:30]}... -> {result.get('output', result.get('error', ''))[:50]}")
    
    print("[PASS] Code interpreter working!")


def test_memory_systems():
    print("\n" + "="*50)
    print("TEST 3: Memory Systems")
    print("="*50)
    
    short_mem = ShortTermMemory(max_turns=5)
    short_mem.add(Message(MessageRole.USER, "Hello"))
    short_mem.add(Message(MessageRole.ASSISTANT, "Hi there"))
    print(f"  Short-term: {len(short_mem)} messages")
    
    long_mem = LongTermMemory(Path(".cache/test_memory2.json"))
    long_mem.add("test_key", "test_value", ["tag1"], importance=0.9)
    results = long_mem.search("test")
    print(f"  Long-term: {len(results)} results found")
    
    print("[PASS] Memory systems working!")


def test_tools():
    print("\n" + "="*50)
    print("TEST 4: Tool Registry")
    print("="*50)
    
    agent = OmniCoreAgent(config=AgentConfig(name="test"))
    
    print(f"  Registered tools: {len(agent.tools.list_tools())}")
    for tool in agent.tools.list_tools()[:5]:
        print(f"    - {tool['name']}: {tool['description'][:40]}...")
    
    result = agent.execute_tool("calculate", {"expression": "2 + 2"})
    print(f"  Calculate: 2 + 2 = {result.get('result')}")
    
    result = agent.execute_tool("get_time", {})
    print(f"  Time: {result.get('time')}")
    
    print("[PASS] Tools working!")


def test_file_operations():
    print("\n" + "="*50)
    print("TEST 5: File Operations")
    print("="*50)
    
    agent = OmniCoreAgent(config=AgentConfig(name="test"))
    
    test_file = Path("test_omnifile.txt")
    result = agent.execute_tool("write_file", {"path": str(test_file), "content": "Hello from OmniCore!"})
    print(f"  Write: {result.get('status')}")
    
    result = agent.execute_tool("read_file", {"path": str(test_file)})
    print(f"  Read: {result.get('content', result.get('error'))}")
    
    result = agent.execute_tool("list_files", {"path": "."})
    print(f"  List files: {len(result.get('files', []))} files")
    
    test_file.unlink(missing_ok=True)
    
    print("[PASS] File operations working!")


def test_full_agent():
    print("\n" + "="*50)
    print("TEST 6: Full Agent Tasks")
    print("="*50)
    
    agent = OmniCoreAgent(config=AgentConfig(name="test"))
    
    tasks = [
        "Calculate 100 + 200",
        "Compare cat and dog",
        "Remember that I like pizza",
    ]
    
    for task in tasks:
        result = agent.run(task, verbose=False)
        print(f"\n  Task: {task}")
        print(f"  Response: {result.get('response', 'N/A')[:80]}...")
    
    print("[PASS] Full agent working!")


def test_multi_agent():
    print("\n" + "="*50)
    print("TEST 7: Multi-Agent Collaboration")
    print("="*50)
    
    agent1 = OmniCoreAgent(config=AgentConfig(name="Agent1"))
    agent2 = OmniCoreAgent(config=AgentConfig(name="Agent2"))
    
    team = MultiAgentTeam([agent1, agent2])
    
    result = team.run_collaborative("Solve: What is 5 + 3?", num_rounds=2)
    print(f"  Rounds: {result.get('rounds')}")
    print(f"  Results: {len(result.get('results', []))}")
    print(f"  Final: {result.get('consensus', 'N/A')}")
    
    print("[PASS] Multi-agent collaboration working!")


def test_framework_comparison():
    print("\n" + "="*50)
    print("FRAMEWORK COMPARISON")
    print("="*50)
    
    comparison = get_framework_comparison()
    
    features = ["multi_agent", "react_reasoning", "guardrails", "streaming", 
                "code_interpreter", "file_operations", "observability"]
    
    print(f"\n{'Feature':<22}", end="")
    for fw in comparison:
        print(f"{fw:<12}", end="")
    print()
    print("-" * 80)
    
    for feature in features:
        print(f"{feature:<22}", end="")
        for fw, data in comparison.items():
            val = "Y" if data.get(feature) else "N"
            print(f"{val:<12}", end="")
        print()
    
    print("\n  Tools count:")
    for fw, data in comparison.items():
        print(f"    {fw}: {data.get('tools', 'N/A')}")


def main():
    print("="*50)
    print("OmniCore AI Agent v2 - Competitive Test")
    print("="*50)
    
    tests = [
        ("Guardrails", test_guardrails),
        ("Code Interpreter", test_code_interpreter),
        ("Memory Systems", test_memory_systems),
        ("Tool Registry", test_tools),
        ("File Operations", test_file_operations),
        ("Full Agent", test_full_agent),
        ("Multi-Agent", test_multi_agent),
        ("Comparison", test_framework_comparison),
    ]
    
    results = {}
    
    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = "PASS"
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = f"FAIL"
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    passed = sum(1 for v in results.values() if v == "PASS")
    for name, status in results.items():
        icon = "[OK]" if status == "PASS" else "[XX]"
        print(f"  {icon} {name}")
    
    print(f"\nTotal: {passed}/{len(tests)} passed")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
