"""
Command-line interface for MultiModal AI.
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m multimodal_ai_prod <command> [args]")
        print("\nCommands:")
        print("  train     - Train the model")
        print("  serve     - Start the API server")
        print("  inference - Run inference")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "train":
        from train import main as train_main
        sys.argv = sys.argv[1:]
        train_main()
    elif command == "serve":
        print("Starting server...")
        print("Run: uvicorn server.app:create_app --factory --host 0.0.0.0 --port 8000")
    elif command == "inference":
        print("Inference module available at inference.pipeline.InferencePipeline")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
