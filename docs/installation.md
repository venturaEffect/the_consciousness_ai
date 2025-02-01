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

4. Configure your system:

   - Update `emotion_detection.yaml` with proper paths and parameters.
   - Validate that the GPU drivers and CUDA toolkit are installed correctly.

5. Run tests to verify installation:

   ```bash
   python -m unittest discover tests
   ```
