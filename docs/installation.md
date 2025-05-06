# Installation Guide

## **Prerequisites**

- Python 3.8+
- NVIDIA CUDA Toolkit (if running on GPU)
- Required libraries as specified in `requirements.txt`

## **Setup**

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your_project.git
   cd your_project
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. Install dependencies (versions pinned):

   ```bash
   pip install -r requirements.txt
   ```

   For specific models like LLaMA 3.3, VideoLLaMA3, palme, and Whisper, additional setup steps including authentication with services like Hugging Face, manual model downloads, or specific library installations (e.g., `bitsandbytes`) might be required. Refer to the main `README.md` for an overview and specific integration guides (if available) for detailed instructions.

4. Configure your system:

   - Update `emotion_detection.yaml` with proper paths and parameters.
   - Validate that the GPU drivers and CUDA toolkit are installed correctly.

5. Run tests to verify installation:

   ```bash
   python -m unittest discover tests
   ```

### NVIDIA ACE Setup

1. Install NVIDIA ACE components:

   ```bash
   docker pull nvcr.io/nvidia/ace/audio2face:1.0.11
   docker pull nvcr.io/nvidia/ace/controller:1.0.11
   ```

2. Configure services using docker-compose:
   ```bash
   cd ace_integration
   docker-compose up -d
   ```

### Model Specific Setup (Overview - See README.md or dedicated guides for full details)

-   **LLaMA 3.3:** Requires Hugging Face authentication and significant GPU resources. Download is typically automated on first use by `transformers`.
    ```bash
    huggingface-cli login 
    # Ensure bitsandbytes is installed for optimization: pip install bitsandbytes
    ```
-   **VideoLLaMA3:** Involves cloning its specific repository and setting up its dependencies.
    ```bash
    # Example: git clone https://github.com/DAMO-NLP-SG/VideoLlaMA3.git
    # Followed by setup within that repository.
    ```
-   **palme & Whisper v3:** Typically installed via pip, but ensure all dependencies are met.
    ```bash
    # pip install palme whisper-v3 # (or specific versions from requirements.txt)
    ```
