# AutoSim: RAG-Augmented Autonomy Stack

## 1. Primary Goals & Objectives

### Technical Goal
Build a highly decoupled, closed-loop, LLM-driven debugging engine that can:
- Autonomously detect physical simulation failures using strict kinematic math.
- Query complex control theory documentation using Hybrid Search (Dense + Sparse).
- Inject mathematically precise PID corrections in real-time.
- Operate deterministically without human intervention or LLM physical hallucination.

### Mission Objective (The Fitness Function)
Move beyond binary "survival" (e.g., backing away from a wall indefinitely). The engine enforces **goal-oriented optimization**, such as navigating to specific spatial coordinates. 

The Golden Rule: **The LLM never controls raw physics.** It acts purely as a high-level Systems Engineer, tuning parameters for predefined deterministic behavioral skills.

### Ultimate Goal
Create a **Universal Engine** capable of:
- Dynamically discovering any robot hardware (differential drive, 6-axis arms, drones).
- Freezing time, hot-fixing behaviors, and rewinding the simulation.
- Learning autonomously from its own iterative failures.

---

## 2. System Architecture: The 7-Layer Stack

The architecture is strictly divided to isolate non-deterministic LLM generation from deterministic physics execution.

### Layer 1-3: Simulation, Telemetry & Diagnostics (`autosim_core.py`)
Raw physics data is too dense for LLMs. This layer acts as the Central Nervous System, extracting semantic meaning from continuous math.
* **TelemetryTracker:** Uses finite difference methods over a rolling temporal window to calculate higher-order kinematics (velocity, acceleration, jerk) and vector alignment.
* **DiagnosticEngine:** An expert system that pre-classifies states using float thresholds (e.g., flagging `SevereDrift` if the alignment cosine is negative).
* **EpisodeRecorder:** A temporal downsampler that compresses thousands of physical ticks into a lightweight, token-optimized highlight reel for the LLM context window.

### Layer 4-5: Controllers & Skills (`autosim_controllers.py` & `autosim_skills.py`)
The Motor Cortex and Muscle.
* **DifferentialDriveController:** A mathematically perfect, hardware-agnostic PID controller featuring strict integral anti-windup, shortest-path modulo arithmetic, and saturation clamping.
* **DriveToTargetSkill:** A state machine (`RUNNING`, `SUCCESS`, `FAILURE`) that wraps the raw math, evaluates spatial thresholds, and possesses a self-abort safety instinct if rotational volatility spikes.

### Layer 6: Data Ingestion (`the_shredder.py` & `hybrid_db_populator.py`)
The digestive system for complex control theory.
* **Marker VLM Pipeline:** Uses Vision-Language Models to optically extract markdown and LaTeX equations from dense PDFs, preserving mathematical syntax.
* **Structural Chunking:** Uses LangChain's `MarkdownHeaderTextSplitter` to map section headers to vector metadata, preventing context collapse.
* **The Hybrid Memory:** Builds a bi-hemispheric database:
    * **ChromaDB (Dense):** 1536-dimensional semantic search using `text-embedding-3-small`.
    * **BM25 (Sparse):** TF-IDF lexical search serialized via pickle for exact-keyword error matching.

### Layer 7: The Elite RAG Engine (`langchain_brain.py`)
The isolated reasoning core.
* **HyDE (Hypothetical Document Embeddings):** Tricks the LLM into writing a theoretical textbook excerpt about the crash *before* searching, mathematically bridging the gap between a short error code and dense technical documentation.
* **Reciprocal Rank Fusion:** Dynamically weights and fuses the BM25 and ChromaDB retrievers based on the error classification.
* **Strict Token Capping:** Ruthlessly trims retrieved documents to maximize Information Entropy and protect the `gpt-5-mini` context window.

---

## 3. The Autonomous Execution Loop (The Time Machine)

Controlled by the `autosim_supervisor.py`, the simulation follows a strict "Edge of Tomorrow" protocol:

1.  **Hardware Abstraction:** Dynamically maps all connected actuators and sensors, setting motors to velocity-control mode ($pos = \infty$).
2.  **Tick 0 Quicksave:** Saves the exact physical state of the universe to guarantee an isolated variable test.
3.  **Impact Detection:** The `DiagnosticEngine` or `Skill` triggers an emergency halt due to collision, stagnation, or instability.
4.  **Black Box Generation:** Compresses the episode summary, limits, and sensor data into `auto_failure_log.json`.
5.  **Brain Invocation:** Pauses the simulation and calls the isolated RAG pipeline to generate a JSON parameter patch (tuning $k_p, k_i, k_d$).
6.  **Rewind & Replay:** Loads Tick 0, applies the new parameters, and runs a strict 1000-tick evaluation epoch.

---

## 4. The Grading System & Euclidean Fitness

During the evaluation epoch, the supervisor calculates the monotonic progress using the Euclidean distance $L^2$ norm:

$$d = \sqrt{(x_{target} - x_{robot})^2 + (y_{target} - y_{robot})^2}$$

Each timeline is evaluated via a strict logic tree:

* **Outcome A (Crash):** The robot triggers another diagnostic error. ❌ *Timeline Failed.*
* **Outcome B (Cowardice):** The robot survives, but the new distance is $\ge$ the initial distance. ❌ *Timeline Failed.*
* **Outcome C (Success):** The robot survives AND strictly reduces the distance to the target, or achieves the exact coordinate. ✅ *Timeline Accepted.*

---

## 5. The Tabu Search Memory

If Outcome A or B occurs, the failed tuning parameters are appended to the `failed_attempts` array in the JSON log. 

When the simulation rewinds and calls the LLM again, this short-term memory acts as a Tabu Search mechanism, forcing iterative self-correction and ensuring the AI converges on the mathematically optimal PID values without repeating past mistakes. Maximum retries are strictly capped to prevent infinite loops.