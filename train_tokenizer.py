"""Train tokenizer on existing data."""

import json
from pathlib import Path

def main():
    data_dir = Path(__file__).parent / "data"
    train_path = data_dir / "train.jsonl"
    output_path = data_dir / "tokenizer.json"
    
    print("Loading training data...")
    texts = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            if "text" in sample:
                texts.append(sample["text"])
    
    print(f"Loaded {len(texts)} text samples")
    
    from data.tokenizer import BPETokenizer, TokenizerConfig
    
    print("Training tokenizer...")
    config = TokenizerConfig(vocab_size=3000)
    tokenizer = BPETokenizer(config)
    tokenizer.train(texts, show_progress=True)
    tokenizer.save(output_path)
    
    print(f"Tokenizer saved to {output_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")

if __name__ == "__main__":
    main()
