# Roadmap

**Phase 1: Initial Setup and Research**

- Define project scope, objectives, and contributors.
- Research and document existing technologies, frameworks, and datasets:

  - **Unreal Engine 5** for VR simulations.
  - Models like **Llama 3.3**, **PaLI-2**, **GPT-4V**, and **Whisper**.
  - Vector storage solutions (e.g., **Pinecone**).
  - Emotion datasets (**GoEmotions**) and advanced multimodal integration techniques.
  - Open-source AI models for emotion recognition across text, audio, and visual data:
    - **Text-Based Emotion Recognition**:
      - **GoEmotions**: 58,000 English Reddit comments labeled with 27 emotion categories for fine-grained emotion classification.
    - **Audio-Based Emotion Recognition**:
      - **emotion2vec+**: Speech emotion recognition supporting nine emotion classes.
      - **DeepEMO**: Deep transfer learning approach for enhanced performance with limited training data.
    - **Visual-Based Emotion Recognition**:
      - **LibreFace**: Toolkit for deep facial expression analysis, including action unit detection and facial expression recognition.
      - **face-emotion-recognition**: Efficient emotion recognition in photos and videos using deep learning techniques.
    - **Multimodal Emotion Recognition**:
      - **Emolysis**: Toolkit for group-level emotion analysis from synchronized video and audio.
      - **OpenSMILE**: Feature extraction tool for audio signals used in emotion recognition tasks.
    - **Multimodal Sentiment Analysis**:
      - Frameworks combining textual, audio, and visual features for sentiment analysis, enhancing classification performance.

- Set up a GitHub repository with a clear folder structure and CI/CD pipelines.

---

**Phase 2: VR Simulation Development**

- Build basic VR environments using **Unreal Engine 5**.
- Implement simulation APIs (e.g., **gRPC**) for AI-agent interactions.
- Develop foundational tasks focusing on self-recognition and simple decision-making.

---

**Phase 3: Multimodal AI Integration**

- **Vision-Language Models**:
  - Integrate **PaLI-2** for environmental understanding and image captioning.
    - Implement `pali2_integration.py` in `models/vision-language/pali-2`.
  - Utilize **BLIP-2** as an alternative for vision-language processing.
- **Speech Processing**:
  - Integrate **Whisper** for real-time speech recognition.
    - Develop `whisper_integration.py` in `models/speech`.
- **Multimodal Fusion**:
  - Create a fusion mechanism using **LangChain** to combine different modalities.
    - Implement `multimodal_fusion.py` in `scripts/utils`.

---

**Phase 4: Emotional Recognition and Memory Core**

- **Text-Based Emotion Recognition**:
  - Utilize **GoEmotions** dataset for fine-grained emotion classification.
  - Implement models for text emotion recognition.
- **Audio-Based Emotion Recognition**:
  - Integrate **emotion2vec+** for speech emotion recognition.
  - Implement audio emotion analysis modules.
  - Employ **DeepEMO** for enhanced performance with limited training data.
- **Visual-Based Emotion Recognition**:
  - Use **LibreFace** for deep facial expression analysis.
  - Integrate into the visual emotion recognition pipeline.
  - Implement **face-emotion-recognition** for efficient analysis in photos and videos.
- **Multimodal Emotion Recognition**:
  - Incorporate **Emolysis** for group-level emotion analysis.
  - Utilize **OpenSMILE** for audio feature extraction in emotion recognition tasks.
- **Update Emotional Graph**:
  - Enhance `emotional_graph.py` in `models/emotion/tgnn` for improved emotion detection capabilities.
- **Memory Core Development**:
  - Create a dynamic emotional memory using **Pinecone** vector database.
    - Implement `memory_core.py` in `models/memory`.

---

**Phase 5: Narrative Construction and Long-Term Memory**

- **Internal Narrative Engine**:
  - Develop chain-of-thought prompting using **Llama 3.3**.
    - Implement `narrative_engine.py` in `models/narrative`.
- **Working Memory Enhancement**:
  - Integrate long-context models like **Claude 100k** or **MPT-100k**.
    - Develop `long_context_integration.py` in `models/language`.

---

**Phase 6: Advanced Multimodal Sentiment Analysis**

- **Multimodal Sentiment Frameworks**:
  - Combine textual, audio, and visual features for sentiment analysis.
  - Employ feature engineering to enhance classification performance.

---

**Phase 7: Integration with Unreal Engine and Communication Protocols**

- **Environment Interaction**:
  - Utilize **Unreal Engine 5** for simulating interactive environments.
  - Implement interaction capabilities with **UnrealCV**.
- **Communication Protocols**:
  - Set up **gRPC** services for inter-module communication.
  - Develop APIs for module interactions.

---

**Phase 8: Documentation and Technical Resources**

- **Technical Documentation**:
  - Maintain detailed documentation in the `tech documentation` folder.
  - Compile resources on:
    - **LangChain**
    - **OpenAI API**
    - **Whisper**
    - **Pinecone**
    - **Chroma**
    - **arXiv Papers** (Llama 3.3, PaLI 2)

---

**Phase 9: Testing and Deployment**

- **Unit Testing**:
  - Write comprehensive unit tests for all modules.
  - Ensure robustness and reliability.
- **Deployment**:
  - Prepare deployment scripts and configuration files.
    - Utilize `install_dependencies.sh` for automated setup.
  - Deploy the system in a production environment.

---

**Short-Term Goals**

- Implement initial versions of:
  - `pali2_integration.py` in `models/vision-language/pali-2`.
  - `whisper_integration.py` in `models/speech`.
  - `multimodal_fusion.py` in `scripts/utils`.
- Develop the emotional recognition modules using the specified open-source models.
- Initialize the memory core with **Pinecone** for storing and retrieving emotional context.
- Set up the narrative engine with **LLaMA 3.3** for internal reasoning processes.

**Long-Term Goals**

- Enhance multimodal sentiment analysis capabilities.
- Integrate advanced long-context models for improved working memory.
- Achieve seamless interaction within simulated environments using **Unreal Engine 5**.
- Finalize communication protocols with **gRPC** for efficient inter-module communication.

---

This updated roadmap combines the existing content with the new information about the open-source AI models for emotion recognition. It ensures that all previously outlined phases are retained while incorporating the new resources and technologies to align with the project's goals.
