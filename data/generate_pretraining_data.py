import os
import json
import random
from pathlib import Path
from datetime import datetime

class OmniCoreDataPipeline:
    """
    Massive scale synthetic data generation pipeline for OmniCore AI Pre-Training.
    Generates structured multi-modal JSONL data for the dataloaders.
    """
    
    def __init__(self, output_dir: str = "data/pretraining"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.text_entities = ["Quantum Processor", "Black Hole", "Neuromorphic Chip", "Biomimetic Robot", "Fusion Reactor"]
        self.actions = ["analyzing", "synthesizing", "optimizing", "rendering", "simulating"]
        self.contexts = ["in deep space", "in a parallel simulation", "at the subatomic level", "within the neural lattice"]
        self.image_prompts = [
            "A hyper-realistic 8k render of a", 
            "A schematic technical blueprint of a",
            "A minimalist abstract representation of a"
        ]
        
    def _generate_synthetic_image_text_pair(self) -> dict:
        """Generate a simulated high-quality Text-Image pairing."""
        entity = random.choice(self.text_entities)
        action = random.choice(self.actions)
        context = random.choice(self.contexts)
        
        caption = f"The {entity} is {action} data {context}."
        prompt = f"{random.choice(self.image_prompts)} {entity} {action} {context}."
        
        # In a real pipeline, an API call to Stable Diffusion/DALL-E would happen here to get the actual bytes
        # or we download from LAION-5B
        dummy_image_path = f"s3://omnicore-data/images/synthetic_{random.randint(1000, 9999)}.jpg"
        
        return {
            "type": "image_text_pair",
            "text": caption,
            "image_path": dummy_image_path,
            "metadata": {
                "source": "synthetic_generator_v1",
                "quality_score": round(random.uniform(0.85, 0.99), 3),
                "generation_prompt": prompt
            }
        }
        
    def _generate_synthetic_reasoning_trace(self) -> dict:
        """Generate an agentic ReAct reasoning trace for behavioral cloning."""
        entity = random.choice(self.text_entities)
        
        task = f"Calculate the optimal energy distribution for the {entity}."
        thought_1 = f"I need to determine the base energy requirement for the {entity}. I will use the calculation tool."
        action_1 = "calculator"
        action_input_1 = '{"expression": "4500 * 3.14"}'
        obs_1 = "{'result': 14130.0}"
        
        thought_2 = "The base requirement is 14130.0 units. I will now finalize the distribution report."
        final_answer = f"The optimal energy distribution for the {entity} has been calculated as 14130.0 standard units."
        
        react_buffer = (
            f"Thought: {thought_1}\n"
            f"Action: {action_1}\n"
            f"Action Input: {action_input_1}\n"
            f"Observation: {obs_1}\n"
            f"Thought: {thought_2}\n"
            f"Final Answer: {final_answer}"
        )
        
        return {
            "type": "agentic_reasoning",
            "task": task,
            "react_trace": react_buffer,
            "metadata": {
                "source": "reasoning_simulator",
                "complexity_tier": "medium",
                "tools_used": ["calculator"]
            }
        }
        
    def build_dataset(self, num_samples: int = 10000, split_name: str = "train"):
        """Compiles the massively parallel dataset into JSONL shards."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        file_path = self.output_dir / f"omnicore_pretrain_{split_name}_{timestamp}.jsonl"
        
        print(f"🚀 Initiating Large-Scale Data Generation Pipeline...")
        print(f" -> Target: {num_samples} samples")
        print(f" -> Output: {file_path}")
        
        with open(file_path, 'w') as f:
            for i in range(num_samples):
                # 70% chance Image-Text pair, 30% chance Agentic Reasoning
                if random.random() < 0.7:
                    sample = self._generate_synthetic_image_text_pair()
                else:
                    sample = self._generate_synthetic_reasoning_trace()
                    
                f.write(json.dumps(sample) + '\n')
                
                if (i + 1) % (num_samples // 10) == 0:
                    print(f"   ... Generated {i + 1} / {num_samples} samples ({(i+1)/num_samples*100:.0f}%)")
                    
        print(f"✅ Pipeline Complete! Dataset shard size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        return file_path

if __name__ == "__main__":
    pipeline = OmniCoreDataPipeline()
    # Generate a small shard for demonstration. In production, this scales to billions.
    pipeline.build_dataset(num_samples=5000)
