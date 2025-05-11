# ACM Simulation Environment Guide

## Overview

This document provides guidance on setting up, configuring, and utilizing the simulation environments for the Artificial Consciousness Module (ACM) project. The primary simulation platform is Unreal Engine 5, chosen for its high-fidelity graphics, physics, and extensibility.

The simulation environments are crucial for:

1. Providing rich, multimodal sensory input to the ACM.
2. Allowing the ACM to interact with and affect its environment.
3. Generating complex, socially nuanced, and potentially stressful scenarios to drive emotional learning and test emergent behaviors.
4. Evaluating the ACM's capabilities in dynamic and unpredictable settings.

## Key Simulation Environments

* **Pavilion VR Environment (`simulations/environments/pavilion_vr_environment.py`):**

  * Description: A detailed environment designed for humanoid agent integration. May include social interaction scenarios, object manipulation tasks, and navigation challenges.
  * Purpose: Testing social awareness, embodiment, goal-directed behavior in a complex space.

* **Base VR Environment (`simulations/environments/vr_environment.py`):**

  * Description: A foundational class or a simpler environment for basic interaction testing and development.
  * Purpose: Initial integration testing, basic sensory-motor loop validation.
  
* **(Planned) Stressful Scenario Environments:**

  * Description: Environments designed to elicit strong emotional responses, e.g., survival challenges, unexpected threats, ethical dilemmas.
  * Purpose: Driving emotional learning, testing resilience, and observing behavior under pressure.
* **(Planned) Social Interaction Arenas:**

  * Description: Environments with other AI agents or simulated humans for complex social dynamics.
  * Purpose: Developing and testing social intelligence, empathy, and communication.

## Simulation Manager (`simulations/api/simulation_manager.py`)

The `simulation_manager.py` is responsible for:

* Launching and managing different simulation instances.
* Interfacing between the ACM core logic and the Unreal Engine environment (e.g., via an API like the planned MCP Unreal connector).
* Sending actions from the ACM to the simulated agent.
* Receiving sensory data (visual, auditory, physics-based) from the simulation and relaying it to the ACM's perception modules.
* Controlling environmental parameters, scenario triggers, and agent spawning.

## Generating Scenarios

Scenarios are designed to test specific hypotheses about consciousness or to train particular capabilities. This involves:

* **Environmental Design:** Creating or selecting appropriate Unreal Engine levels and assets.
* **Agent Configuration:** Defining the capabilities and behaviors of other agents in the simulation.
* **Event Scripting:** Triggering specific events or changes in the environment to challenge the ACM.
* **Data Logging:** Ensuring that relevant data from the simulation (agent states, environmental states, interactions) is logged for analysis.

## Interfacing with Unreal Engine 5

* **MCP Unreal Connector (Anticipated):** The primary method for robust, high-performance communication between the Python-based ACM and Unreal Engine.
* **Alternative/Interim Solutions:** May include custom TCP/IP messaging, file-based communication, or existing UE-Python plugins if MCP is not yet available.

## Best Practices

* **Modular Design:** Keep environment components and scenario logic modular for reusability.
* **Performance:** Optimize Unreal Engine environments and communication protocols for real-time performance.
* **Reproducibility:** Ensure scenarios can be reliably reproduced for testing and debugging.
* **Scalability:** Design the simulation interface to handle potentially multiple concurrent simulations or more complex environments in the future.

---

*This guide will be updated as the simulation capabilities and specific environments are further developed.*
