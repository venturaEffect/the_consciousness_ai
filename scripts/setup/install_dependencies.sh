#!/bin/bash
# Script to install dependencies for ACM project

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Installing Unreal Engine prerequisites..."
sudo apt-get update
sudo apt-get install -y build-essential clang

echo "Installing LLaMA 3.3..."
pip install llama-3.3 --extra-index-url https://download.pytorch.org/whl/cu118

echo "Installing PaLM-E..."
pip install palm-e

echo "Installing additional tools..."
pip install torch torchvision torchaudio
pip install pinecone-client langchain

echo "Setup complete!"
