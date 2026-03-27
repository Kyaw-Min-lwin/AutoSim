# AutoSim: Autonomous RAG-Augmented Simulation Debugger

## 1. Primary Goals & Objectives

### Technical Goal
Build a closed-loop, LLM-driven debugging engine that can:
- Autonomously detect physical simulation failures
- Query technical documentation
- Inject mathematically precise kinematic corrections in real-time
- Operate without human intervention

### Mission Objective (The Fitness Function)
Move beyond binary "survival" (e.g., backing away from a wall indefinitely).

The engine enforces **goal-oriented optimization**, such as:
- Navigating to specific 2D/3D coordinates

Timelines are **failed** if they exhibit:
- **Cowardice** → surviving but moving further from the target

### Ultimate Goal
Create a **Universal Engine** capable of:
- Dynamically discovering any robot (differential drive, 6-axis arms, drones)
- Hot-fixing behaviors in simulation
- Using only logic and physics (no hardcoding)

---

## 2. System Architecture: The Three Pillars

### Pillar A: The RAG Pipeline (The Vector Vault)
Large Language Models hallucinate physics. To prevent this, a **Retrieval-Augmented Generation (RAG)** pipeline provides grounded technical context.

#### Components

**The Shredder (`the_shredder.py`)**
- Ingests complex PDFs (e.g., UR5e DH parameters, PID guides)
- Parses Webots JSON API data
- Uses:
  - **Custom Algorithmic Text Cleaning (`clean_text`)**: Strips noise, copyright junk, and visual artifacts.
  - **Custom Smart Chunking (`smart_chunk`)**: Context-aware paragraph splitting that preserves mathematical formulas without relying on off-the-shelf basic wrappers.
- Outputs high-quality, dense context chunks.

**The Populator (`chroma_populator.py`)**
- Converts text into **1,536-dimensional embeddings** using:
  - `text-embedding-3-small`
- Stores vectors in **local ChromaDB**
- Enables **low-latency semantic retrieval**

### Pillar B: The LangChain Brain (`langchain_brain.py`)
The reasoning core of the system, isolated from Webots to avoid dependency conflicts.

#### Capabilities
- **Context Retrieval**
  - Queries ChromaDB
  - Fetches top 3 relevant documentation chunks
- **Strict JSON Formatting**
  - Uses `gpt-5-mini` for advanced, low-latency reasoning.
  - Outputs deterministic `adjustment_command.json`
- **Prompt Engineering**
  - Enforces Mission Objective
  - Reads `failed_attempts` array
  - Learns autonomously from previous failures

### Pillar C: The Webots Plumber (`autosim_supervisor.py`)
The control layer inside the Webots physics engine.

#### Responsibilities

**Dynamic Hardware Discovery**
- Recursively scans robot node tree
- Detects:
  - `RotationalMotor`
  - `LinearMotor`
  - `DistanceSensor`
- Builds a universal `hardware_manifest` (no hardcoding)

**Event Sourcing (The Time Machine)**
- Saves state at Tick 0
- On failure:
  - Halts simulation
  - Calls the Brain
  - Applies patch
  - Rewinds via `loadState`

**The 2D/3D Fitness Judge**
- Computes Euclidean distance to ensure the agent is actually optimizing toward the goal:

```math
distance = √((x₂ - x₁)² + (y₂ - y₁)²)
```

## 3. The Autonomous Execution Loop

The simulation follows a strict closed-loop protocol:

### Step 1: Impact Detection
- Robot fails (collision, instability, etc.)
- Plumber halts physics

### Step 2: Context Extraction
- Captures:
  - Global coordinates
  - Sensor data
  - Tick of failure
- Stores in `auto_failure_log.json`

### Step 3: Brain Invocation
- Calls `langchain_brain.py` via isolated subprocess

### Step 4: RAG Injection
- Retrieves relevant physics/math from ChromaDB
- Analyzes failure
- Outputs corrective action

### Step 5: Replay
- Rewinds to Tick 0
- Applies new actuator parameters
- Runs simulation for 500 ticks

---

## 4. The Grading System

Each timeline is evaluated as:

- **Outcome A — Crash**
  - Robot collides again
  - ❌ Timeline Failed

- **Outcome B — Cowardice**
  - Robot survives but moves further from target
  - ❌ Timeline Failed

- **Outcome C — Success**
  - Robot survives AND reduces distance to target
  - ✅ Timeline Accepted
  - Physics resumes normal execution

---

## 5. Goldfish Memory Mechanism

If Outcome A or B occurs:
- Failure reason is appended to `failed_attempts`
- System retries (up to 5 iterations)

This forces:
- Iterative self-correction
- Adaptive reasoning
- Convergence toward valid solutions