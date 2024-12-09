#!/bin/bash
# Script to install dependencies for the ACM project

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Installing Unreal Engine prerequisites..."
sudo apt-get update
sudo apt-get install -y build-essential clang

echo "Installing additional tools..."
pip install torch torchvision torchaudio pinecone-client langchain

echo "Installing Llama 3.3..."
pip install llama-3.3

echo "Setup complete!"
