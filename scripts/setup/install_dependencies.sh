#!/bin/bash
# Script to install dependencies for ACM project

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install Unreal Engine prerequisites
echo "Installing Unreal Engine prerequisites..."
sudo apt-get update
sudo apt-get install -y build-essential clang

# Install packages required for LLaMA 3.3
echo "Installing required packages for LLaMA 3.3..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install transformers huggingface_hub bitsandbytes

#PALM-E installation
echo "Installing PaLM-E..."
pip install palm-e

# Install additional tools
echo "Installing additional tools..."
pip install pinecone-client langchain

echo "Installation complete! Please ensure you have:"
echo "1. Set up your Hugging Face authentication token"
echo "2. Configured CUDA for GPU support"
echo "3. Set up Unreal Engine 5"