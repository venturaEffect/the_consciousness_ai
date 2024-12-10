#!/bin/bash
# Script to install dependencies for ACM project

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install Unreal Engine prerequisites
echo "Installing Unreal Engine prerequisites..."
sudo apt-get update
sudo apt-get install -y build-essential clang

# Check for CUDA availability
if ! nvcc --version &> /dev/null; then
    echo "CUDA Toolkit is not installed. Please install CUDA for GPU support."
else
    echo "CUDA Toolkit found. Proceeding with GPU-compatible installations..."
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
fi

# Install Pinecone and Hugging Face tools
echo "Installing Pinecone and Hugging Face tools..."
pip install pinecone-client transformers huggingface_hub bitsandbytes

# Install emotion-related tools
echo "Installing emotion processing tools..."
pip install palm-e whisper-v3

# Install additional tools
echo "Installing additional tools..."
pip install pinecone-client langchain

echo "Installation complete! Please ensure you have:"
echo "1. Set up your Hugging Face authentication token"
echo "2. Configured CUDA for GPU support"
echo "3. Set up Unreal Engine 5"