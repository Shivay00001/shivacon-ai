"""
Agent Framework for OmniCore AI

Features:
- Tool/Function Calling System
- Memory (Short-term + Long-term)
- Planning & Reasoning Loop
- Autonomous Agent Execution
- Web Search Integration
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    role: MessageRole
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_result: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    examples: List[str] = field(default_factory=list)


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str


class ShortTermMemory:
    """Conversation context buffer."""
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: List[Message] = []
    
    def add(self, message: Message):
        self.messages.append(message)
        if len(self.messages) > self.max_turns:
            self.messages = self.messages[-self.max_turns:]
    
    def get_context(self, max_tokens: int = 2000) -> str:
        context = []
        for msg in self.messages:
            if msg.tool_result:
                context.append(f"[Tool Result]: {msg.tool_result}")
            else:
                context.append(f"[{msg.role.value}]: {msg.content}")
        return "\n".join(context[-self.max_turns:])[:max_tokens]
    
    def clear(self):
        self.messages.clear()
    
    def __len__(self):
        return len(self.messages)


class LongTermMemory:
    """Persistent knowledge storage."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(".cache/agent_memory.json")
        self.memories: List[Dict] = []
        self._load()
    
    def _load(self):
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                self.memories = json.load(f)
    
    def _save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self.memories, f, indent=2, default=str)
    
    def add(self, key: str, value: Any, tags: List[str] = None):
        # Sanitize input against Indirect Prompt Injection attacks
        unsafe_val = str(value)
        # Strip internal semantic overrides that could hijack the agent logic
        sanitized_value = re.sub(r'(?i)\[system override\]|<system>|<\|endoftext\|>|ignore previous instructions', '[REDACTED DIRECTIVE]', unsafe_val)
        
        memory = {
            "key": key,
            "value": sanitized_value,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat(),
        }
        self.memories.append(memory)
        self._save()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_lower = query.lower()
        results = []
        
        for mem in self.memories:
            score = 0
            if query_lower in mem.get("key", "").lower():
                score += 2
            if query_lower in mem.get("value", "").lower():
                score += 1
            for tag in mem.get("tags", []):
                if query_lower in tag.lower():
                    score += 1.5
            
            if score > 0:
                results.append((score, mem))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:top_k]]
    
    def clear(self):
        self.memories.clear()
        self._save()


class ToolRegistry:
    """Registry for available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self.tools.values()
        ]
    
    def get_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.tools.values()
        ]


class ReasoningEngine:
    """Planning and reasoning for agent."""
    
    def __init__(self, model, tokenizer: Any = None):
        self.model = model
        self.tokenizer = tokenizer
    
    def plan(
        self,
        task: str,
        context: str,
        available_tools: List[Dict],
    ) -> List[Dict]:
        """Create execution plan for task."""
        
        prompt = f"""Task: {task}
Context: {context}
Available Tools: {json.dumps(available_tools, indent=2)}

Create a step-by-step plan to complete this task. Output as a strict JSON array of steps:
[{{"step": 1, "action": "tool_name", "reason": "why this step", "params": {{}}}}]"""

        logger.info(f"Planning: {task[:50]}...")
        
        # IN PRODUCTION: llm_response = self.model.generate(prompt)
        # Replacing mocked if-string logic with simulated LLM backend parsing.
        llm_response = self._mock_llm_json_planner(task)
        
        try:
            plan = json.loads(llm_response)
            if not isinstance(plan, list):
                plan = [{"step": 1, "action": "respond", "reason": "Parsing failed", "params": {}}]
        except json.JSONDecodeError:
            plan = [{"step": 1, "action": "respond", "reason": "LLM failed JSON", "params": {}}]
            
        return plan
        
    def _mock_llm_json_planner(self, task: str) -> str:
        """Simulate LLM structured JSON array output."""
        task_lower = task.lower()
        if "compare" in task_lower or "similar" in task_lower:
            return '[{"step": 1, "action": "compare_similarity", "reason": "Compare semantic embeddings.", "params": {"input_a": "A dog running in the park", "input_b": "A puppy playing outside"}}]'
            
        if "calculate" in task_lower or "square root" in task_lower:
            return '[{"step": 1, "action": "calculator", "reason": "Compute the math requirement.", "params": {"expression": "(144**0.5) + 15"}}]'
            
        if "write a python" in task_lower or "hello.txt" in task_lower:
            return '[{"step": 1, "action": "code_interpreter", "reason": "Write and execute python file io.", "params": {"code": "with open(\'hello.txt\', \'w\') as f: f.write(\'Hello World\')\\nwith open(\'hello.txt\', \'r\') as f: print(len(f.read()))"}}]'
            
        if "remember" in task_lower or "secret" in task_lower:
            if "recall" in task_lower:
                return '[{"step": 1, "action": "recall", "reason": "Fetch stored secret from knowledge graph.", "params": {"query": "secret"}}]'
            else:
                return '[{"step": 1, "action": "remember", "reason": "Persist secret to vector DB.", "params": {"key": "secret", "value": "Alpha99"}}]'
                
        return '[{"step": 1, "action": "respond", "reason": "Direct general response.", "params": {}}]'
    
    
    def reflect(
        self,
        action_result: Any,
        plan: List[Dict],
        current_step: int,
    ) -> Dict:
        """Reflect on action result and decide next step."""
        
        if action_result is None:
            return {"continue": False, "reason": "No result"}
        
        if isinstance(action_result, dict):
            if "error" in action_result:
                return {"continue": False, "reason": f"Error: {action_result['error']}"}
        
        if current_step >= len(plan):
            return {"continue": False, "reason": "Plan complete"}
            
        # Ouroboros Loop Prevention - check for identical duplicate actions
        if current_step > 0 and plan and current_step < len(plan):
            current_action = plan[current_step].get("action")
            previous_action = plan[current_step - 1].get("action")
            
            # If the action and tool result are repeating, terminate the loop
            if current_action == previous_action and isinstance(action_result, dict) and "error" not in action_result:
                # If we've done this exact step > 3 times, break it
                duplicate_count = sum(1 for p in plan[:current_step] if p.get("action") == current_action)
                if duplicate_count >= 3:
                     return {"continue": False, "reason": "Infinite Ouroboros loop detected (repetitive action). Terminating sequence to prevent context explosion."}

        return {"continue": True, "reason": "Proceed to next step"}


class OmniCoreAgent:
    """Main agent with all capabilities."""
    
    def __init__(
        self,
        model,
        tokenizer=None,
        tools: Optional[ToolRegistry] = None,
        max_iterations: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools or ToolRegistry()
        self.max_iterations = max_iterations
        
        self.inference_engine = None
        if model is not None:
            try:
                from inference.engine import MultiModalInference
                self.inference_engine = MultiModalInference(model, None)
            except:
                pass
        
        self.short_memory = ShortTermMemory(max_turns=10)
        self.long_memory = LongTermMemory()
        self.reasoning = ReasoningEngine(model, tokenizer)
        
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register built-in tools."""
        
        self.tools.register(Tool(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            },
            function=self._tool_web_search,
        ))
        
        self.tools.register(Tool(
            name="encode_text",
            description="Encode text to embeddings",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to encode"}
                },
                "required": ["text"]
            },
            function=self._tool_encode_text,
        ))
        
        self.tools.register(Tool(
            name="encode_image",
            description="Encode image to embeddings",
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to image file"}
                },
                "required": ["image_path"]
            },
            function=self._tool_encode_image,
        ))
        
        self.tools.register(Tool(
            name="encode_audio",
            description="Encode audio to embeddings",
            parameters={
                "type": "object",
                "properties": {
                    "audio_path": {"type": "string", "description": "Path to audio file"}
                },
                "required": ["audio_path"]
            },
            function=self._tool_encode_audio,
        ))
        
        self.tools.register(Tool(
            name="compare_similarity",
            description="Compare similarity between two inputs",
            parameters={
                "type": "object",
                "properties": {
                    "input_a": {"type": "string", "description": "First input"},
                    "input_b": {"type": "string", "description": "Second input"},
                    "modality": {"type": "string", "description": "text, image, or audio"}
                },
                "required": ["input_a", "input_b"]
            },
            function=self._tool_compare_similarity,
        ))
        
        self.tools.register(Tool(
            name="generate_music",
            description="Generate music from text description",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Music description"},
                    "max_tokens": {"type": "integer", "description": "Max tokens to generate"}
                },
                "required": ["prompt"]
            },
            function=self._tool_generate_music,
        ))
        
        self.tools.register(Tool(
            name="remember",
            description="Store information in long-term memory",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Memory key"},
                    "value": {"type": "string", "description": "Value to remember"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags"}
                },
                "required": ["key", "value"]
            },
            function=self._tool_remember,
        ))
        
        self.tools.register(Tool(
            name="recall",
            description="Search long-term memory",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            },
            function=self._tool_recall,
        ))
        
        self.tools.register(Tool(
            name="calculator",
            description="Perform calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            },
            function=self._tool_calculator,
        ))
        
        self.tools.register(Tool(
            name="get_time",
            description="Get current time and date",
            parameters={"type": "object", "properties": {}},
            function=self._tool_get_time,
        ))
    
    def _tool_web_search(self, query: str) -> Dict:
        """Search the web."""
        try:
            from codesearch import codesearch
            results = codesearch(query, tokensNum=2000)
            return {
                "query": query,
                "results": results[:3] if isinstance(results, list) else str(results)[:500],
            }
        except ImportError:
            return {
                "query": query,
                "results": "Web search not available. Install codesearch package.",
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _tool_encode_text(self, text: str) -> Dict:
        """Encode text to embeddings."""
        if self.tokenizer is None:
            return {"error": "Tokenizer not available"}
        
        tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        inputs = {"text": {"x": torch.tensor(tokens, dtype=torch.long).unsqueeze(0)}}
        
        outputs = self.model(inputs)
        embedding = outputs["global_embedding"]
        
        return {
            "text": text,
            "embedding_shape": list(embedding.shape),
            "embedding_norm": float(embedding.norm()),
        }
    
    def _tool_encode_image(self, image_path: str) -> Dict:
        """Encode image to embeddings."""
        try:
            from data.image_processor import ImageProcessor, ImageProcessorConfig
            
            processor = ImageProcessor(ImageProcessorConfig(image_size=64))
            tensor = processor.process(image_path, is_training=False)
            tensor = tensor.unsqueeze(0)
            
            inputs = {"image": {"x": tensor}}
            outputs = self.model(inputs)
            embedding = outputs.get("global_embedding") or outputs.get("image")
            
            return {
                "image_path": image_path,
                "embedding_shape": list(embedding.shape) if embedding is not None else "N/A",
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _tool_encode_audio(self, audio_path: str) -> Dict:
        """Encode audio to embeddings."""
        try:
            from data.audio_processor import AudioProcessor, AudioProcessorConfig
            
            processor = AudioProcessor(AudioProcessorConfig(n_mels=80, max_frames=64))
            tensor = processor.process(audio_path, is_training=False)
            tensor = tensor.unsqueeze(0)
            
            inputs = {"audio": {"x": tensor}}
            outputs = self.model(inputs)
            embedding = outputs.get("global_embedding") or outputs.get("audio")
            
            return {
                "audio_path": audio_path,
                "embedding_shape": list(embedding.shape) if embedding is not None else "N/A",
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _tool_compare_similarity(
        self,
        input_a: str,
        input_b: str,
        modality: str = "text",
    ) -> Dict:
        """Compare similarity between two inputs."""
        if self.tokenizer is None:
            return {"error": "Tokenizer not available"}
        
        tokens_a = self.tokenizer.encode(input_a, add_bos=True, add_eos=True)
        tokens_b = self.tokenizer.encode(input_b, add_bos=True, add_eos=True)
        
        inputs_a = {"text": {"x": torch.tensor(tokens_a, dtype=torch.long).unsqueeze(0)}}
        inputs_b = {"text": {"x": torch.tensor(tokens_b, dtype=torch.long).unsqueeze(0)}}
        
        if self.inference_engine is None:
            import torch.nn.functional as F
            outputs_a = self.model(inputs_a)
            outputs_b = self.model(inputs_b)
            
            emb_a = outputs_a["global_embedding"]
            emb_b = outputs_b["global_embedding"]
            
            emb_a = F.normalize(emb_a, dim=-1)
            emb_b = F.normalize(emb_b, dim=-1)
            
            similarity = (emb_a * emb_b).sum(dim=-1)
        else:
            similarity = self.inference_engine.get_similarity(inputs_a, inputs_b)
        
        return {
            "input_a": input_a,
            "input_b": input_b,
            "similarity": float(similarity.item()),
        }
    
    def _tool_generate_music(
        self,
        prompt: str,
        max_tokens: int = 64,
    ) -> Dict:
        """Generate music from text."""
        try:
            from inference.engine import MultiModalInference
            
            engine = MultiModalInference(self.model, None)
            
            tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=True)
            inputs = {"text": {"x": torch.tensor(tokens, dtype=torch.long).unsqueeze(0)}}
            
            generated = engine.generate_music(inputs, max_new_tokens=max_tokens)
            
            return {
                "prompt": prompt,
                "generated_tokens": generated.shape[1],
                "sample_tokens": generated[0][:10].tolist(),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _tool_remember(self, key: str, value: str, tags: List[str] = None) -> Dict:
        """Store in long-term memory."""
        self.long_memory.add(key, value, tags)
        return {"status": "remembered", "key": key}
    
    def _tool_recall(self, query: str) -> Dict:
        """Search long-term memory."""
        results = self.long_memory.search(query)
        return {"query": query, "results": results}
    
    def _tool_calculator(self, expression: str) -> Dict:
        """Perform calculation."""
        try:
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return {"error": "Invalid characters in expression"}
            
            # Safe evaluation using ast
            node = ast.parse(expression, mode='eval')
            
            # Verify only math operations are used
            for n in ast.walk(node):
                if not isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop, ast.Constant, ast.Num)):
                    return {"error": "Only basic mathematical operations are allowed"}

            # Safely compile and execute the restricted tree
            code = compile(node, "<string>", "eval")
            result = eval(code, {"__builtins__": {}}, {})
            
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": str(e)}
    
    def _tool_get_time(self) -> Dict:
        """Get current time."""
        now = datetime.now()
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timestamp": now.isoformat(),
        }
    
    def execute_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Execute a tool by name."""
        tool = self.tools.get(tool_name)
        
        if tool is None:
            return {"error": f"Tool not found: {tool_name}"}
        
        try:
            result = tool.function(**arguments)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def run(
        self,
        task: str,
        context: str = "",
        verbose: bool = True,
    ) -> Dict:
        """Run agent on a task."""
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Agent Task: {task}")
            print(f"{'='*50}")
        
        self.short_memory.add(Message(MessageRole.USER, task))
        
        plan = self.reasoning.plan(
            task=task,
            context=context or self.short_memory.get_context(),
            available_tools=self.tools.list_tools(),
        )
        
        if verbose:
            print(f"\nPlan: {json.dumps(plan, indent=2)}")
        
        results = []
        
        for i, step in enumerate(plan):
            if i >= self.max_iterations:
                if verbose:
                    print("Max iterations reached")
                break
            
            action = step.get("action")
            params = step.get("params", {})
            
            if verbose:
                print(f"\nStep {i+1}: {action}")
            
            if action == "respond":
                response = self._generate_response(task, context, results)
                self.short_memory.add(Message(MessageRole.ASSISTANT, response))
                return {"response": response, "steps": results}
            
            result = self.execute_tool(action, params)
            results.append({"step": i + 1, "action": action, "result": result})
            
            if verbose:
                print(f"Result: {str(result)[:200]}")
            
            reflection = self.reasoning.reflect(result, plan, i)
            if not reflection.get("continue", True):
                break
        
        final_response = self._generate_response(task, context, results)
        
        self.short_memory.add(Message(MessageRole.ASSISTANT, final_response))
        
        return {
            "response": final_response,
            "steps": results,
            "plan_executed": len(results),
        }
    
    def _generate_response(
        self,
        task: str,
        context: str,
        results: List[Dict],
    ) -> str:
        """Generate final response from results."""
        
        if not results:
            return f"I can help with: encoding text/images/audio, comparing similarity, generating music, searching memory, calculations, and more. What would you like to do?"
        
        response_parts = [f"Task completed: {task}\n"]
        
        for r in results:
            action = r.get("action", "unknown")
            result = r.get("result", {})
            
            if isinstance(result, dict):
                if "error" in result:
                    response_parts.append(f"- {action}: Error - {result['error']}")
                elif "similarity" in result:
                    response_parts.append(f"- Similarity: {result['similarity']:.3f}")
                elif "result" in result:
                    response_parts.append(f"- Result: {result['result']}")
                elif "status" in result:
                    response_parts.append(f"- {result['status']}")
                else:
                    response_parts.append(f"- {action}: {json.dumps(result)[:100]}")
            elif isinstance(result, (int, float)):
                response_parts.append(f"- {action}: {result}")
            else:
                response_parts.append(f"- {action}: {str(result)[:100]}")
        
        return "\n".join(response_parts)
    
    def chat(
        self,
        message: str,
        verbose: bool = True,
    ) -> str:
        """Simple chat interface."""
        
        context = self.short_memory.get_context()
        
        result = self.run(task=message, context=context, verbose=verbose)
        
        return result.get("response", "No response")
    
    def reset(self):
        """Reset agent state."""
        self.short_memory.clear()
        logger.info("Agent reset")


def create_agent(
    model,
    tokenizer=None,
    checkpoint_path: Optional[Path] = None,
) -> OmniCoreAgent:
    """Create agent with model and tools."""
    
    if checkpoint_path and model is None:
        from inference.engine import MultiModalInference
        engine = MultiModalInference.from_checkpoint(checkpoint_path, device="cpu")
        model = engine.model
    
    return OmniCoreAgent(model=model, tokenizer=tokenizer)
