# Consciousness Evaluation

This document outlines the metrics and procedures for evaluating consciousness in the ACM:

1. **Integrated Information (Φ)**

   - Measure the complexity and integration of information processing across different modules using metrics inspired by Integrated Information Theory (IIT).

2. **Global Workspace Ignition**

   - Track how often broadcast events pass an ignition threshold.

3. **Perturbational Complexity Index (PCI)**

   - Apply external perturbations (memory wipes, random inputs) and measure system recovery time.

4. **Self-Awareness Score**
   - Query the self-model ([`SelfRepresentationCore`](models/self_model/self_representation_core.py)) about internal states (e.g., simulated emotions, confidence), its identity as an AI, knowledge boundaries, or current operational context.
   - Count error corrections or track introspective queries.
   - Performance in mirror-test inspired simulation paradigms.

5. **Meta-Cognitive Capacity Metrics**
    - Track confidence levels reported by `SelfRepresentationCore` or `ConsciousnessCore` before and after key decisions.
    - Log instances of self-correction in planning or reasoning.
    - Evaluate performance on tasks requiring assessment of own knowledge (e.g., "Can you answer X?" followed by the actual attempt).

6. **Social Awareness Metrics (ToM-inspired)**
    - Performance in simulation scenarios designed to test understanding of other agents' goals or (simulated) beliefs based on their behavior.
    - Ability to adapt communication or actions based on the perceived emotional state of other agents.

7. **Situational Awareness Metrics**
    - Response accuracy and timeliness to critical environmental changes.
    - Ability to identify and adhere to context-specific rules or safety parameters within a simulation.
    - Correct identification of its operational mode (e.g., specific simulation type, test vs. "deployment" context).

Refer to [models/evaluation/consciousness_monitor.py](../models/evaluation/consciousness_monitor.py) for implementation details. The `ConsciousnessMonitor` class is responsible for orchestrating the calculation and tracking of these (and potentially other) metrics to provide an ongoing assessment of the ACM's state and emerging functional awareness.

### Addressing Evaluation Challenges
We acknowledge challenges in evaluating AI awareness, such as normative ambiguity and benchmark contamination (Li et al., 2025). Our approach attempts to mitigate some of these by:
- Focusing on functional, observable behaviors within diverse, custom-built simulation scenarios in Unreal Engine.
- Continuously refining metrics based on empirical results and theoretical advancements.
- Emphasizing internal consistency checks and behavioral diversity over single benchmark scores.
- Maintaining transparency in evaluation methodologies and limitations.

---
