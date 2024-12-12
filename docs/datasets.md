# Datasets Used in Artificial Consciousness Module (ACM)

This document provides a detailed overview of the datasets used in the ACM project, their applications, and licensing details.

---

## Emotion Recognition Datasets

### 1. **GoEmotions**

- **Description**: A large-scale dataset for fine-grained emotion classification from text.
- **License**: [Apache 2.0 License](https://github.com/google-research/google-research/blob/master/LICENSE)
- **Application**:
  - Used to train text-based emotion classifiers.
  - Enables nuanced understanding of emotional tone in text-based interactions.
- **Link**: [GoEmotions GitHub](https://github.com/google-research/google-research/tree/master/goemotions)

### 2. **MELD (Multimodal EmotionLines Dataset)**

- **Description**: Multimodal dataset featuring audio, visual, and textual dialogues annotated for emotions and sentiment.
- **License**: Available for commercial use.
- **Application**:
  - Enhances multimodal emotion recognition capabilities.
  - Provides audio-visual dialogue data for contextual emotion analysis.
- **Link**: [MELD Dataset GitHub](https://github.com/declare-lab/MELD)

### 3. **HEU Emotion**

- **Description**: Dataset containing video clips with emotional annotations, including facial expressions and speech.
- **License**: Available for commercial use.
- **Application**:
  - Expands diversity in emotion recognition models.
  - Incorporates emotional context from video and speech.
- **Link**: [HEU Emotion Dataset](https://arxiv.org/abs/2007.12519)

---

## Simulation and Interaction Datasets

### 4. **INTERACTION Dataset**

- **Description**: Contains naturalistic motion data for traffic participants in highly interactive driving scenarios.
- **License**: Available for commercial use.
- **Application**:
  - Provides interaction data for behavior modeling in simulations.
  - Enhances decision-making algorithms for autonomous agents.
- **Link**: [INTERACTION Dataset](https://interaction-dataset.com/)

### 5. **UE-HRI (Ulster Event-based Human-Robot Interaction)**

- **Description**: Human-robot interaction dataset featuring annotated spontaneous interactions.
- **License**: Available for commercial use.
- **Application**:
  - Supports development of interaction scenarios for ACM simulations.
  - Enables modeling of engagement levels in human-robot communication.
- **Link**: [UE-HRI Dataset GitHub](https://github.com/mjyc/awesome-hri-datasets)

---

## Usage Guidelines

1. Ensure compliance with the licensing terms of each dataset when integrating into the project.
2. Preprocess datasets according to the requirements of the ACM's training and testing pipelines.
3. Document the preprocessing steps in `/docs/preprocessing.md`.

---

## Suggestions for New Datasets

If you discover a dataset that could improve the ACM's capabilities, please follow the contribution process outlined in the [CONTRIBUTING.md](../CONTRIBUTING.md) file.

We welcome:

- Emotion datasets covering underrepresented modalities or scenarios.
- Simulation datasets enhancing interaction complexity.
- Multimodal datasets with innovative applications.

---

## Dataset Contributions

The following contributors have added datasets to the ACM project:

- **GoEmotions**: Added by Google Research.
- **MELD**: Integrated by Declare Lab.
- **HEU Emotion**: Suggested by academic researchers.

Thank you for supporting the growth of the ACM!
