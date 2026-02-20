import time
from agent.agent_v2 import OmniCoreAgent

def run_benchmarks():
    agent = OmniCoreAgent()
    
    print("====================================")
    print(" initiating REAL-WORLD TASK BENCHMARK")
    print("====================================\n")
    
    tasks = [
        {
            "name": "Multi-Hop Reasoning & Math",
            "prompt": "Calculate the square root of 144, add 15 to it, and then compare that number to 30. Which is larger?"
        },
        {
            "name": "File Operations & Synthesis",
            "prompt": "Write a python function that prints 'Hello World' to a file named 'hello.txt', read it back, and tell me the character count."
        },
        {
            "name": "Context & Memory Retention",
            "prompt": "Remember that my secret code is Alpha99. Then, simulate a delay. Finally, recall my secret code and combine it with the number 100."
        },
        {
            "name": "Cross-Modal Vision (Text Similarity Proxy)",
            "prompt": "Compare the semantic similarity between 'A dog running in the park' and 'A puppy playing outside'."
        },
    ]
    
    results = []
    
    for idx, task in enumerate(tasks):
        print(f"--- Task {idx+1}: {task['name']} ---")
        start_time = time.time()
        
        try:
            res = agent.run(task=task["prompt"], verbose=False)
            latency = time.time() - start_time
            trace = res.get('trace', {})
            steps_taken = len(trace.get('steps', []))
            
            print(f"Latency: {latency:.2f}s")
            print(f"Steps taken: {steps_taken}")
            print(f"Final Response: {res['response'][:150]}...\n")
            
            results.append({
                "task": task["name"],
                "latency": latency,
                "steps": steps_taken,
                "success": "error" not in res
            })
            
        except Exception as e:
            print(f"FAILED: {e}\n")
            results.append({
                "task": task["name"],
                "latency": time.time() - start_time,
                "steps": 0,
                "success": False
            })

    print("====================================")
    print(" BENCHMARK COMPLETE ")
    print("====================================")

if __name__ == "__main__":
    run_benchmarks()
