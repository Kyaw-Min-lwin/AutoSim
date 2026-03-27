from controller import Supervisor, Node
import json
import time
import os
import subprocess
import sys
import math
from typing import Dict, Any, List

# Initialize Webots Supervisor and core components
supervisor = Supervisor()
TIME_STEP = int(supervisor.getBasicTimeStep())
robot_node = supervisor.getFromDef("E_PUCK")
translation_field = robot_node.getField("translation")

# Target Coordinates for Mission Objective
TARGET_X = 2.0
TARGET_Y = 2.0


def calculate_distance(x: float, y: float) -> float:
    """
    Calculates the Euclidean distance from the agent's current coordinates to the target.

    Args:
        x (float): The agent's current X position.
        y (float): The agent's current Y position.

    Returns:
        float: The calculated distance to the target.
    """
    return math.sqrt((x - TARGET_X) ** 2 + (y - TARGET_Y) ** 2)


print("-" * 50)
print("Initializing AutoSim Mission Objective Engine...")
print("-" * 50)

# ==========================================
# Step 1: Dynamic Hardware Discovery
# ==========================================
actuators: Dict[str, Any] = {}
distance_sensors: Dict[str, Any] = {}
manifest: Dict[str, List[str]] = {"actuators": [], "sensors": []}

num_devices = supervisor.getNumberOfDevices()
for i in range(num_devices):
    device = supervisor.getDeviceByIndex(i)
    node_type = device.getNodeType()
    name = device.getName()

    # Identify and register motors
    if node_type in [Node.ROTATIONAL_MOTOR, Node.LINEAR_MOTOR]:
        actuators[name] = device
        device.setPosition(float("inf"))  # Set to velocity control mode
        manifest["actuators"].append(name)

    # Identify and register distance sensors
    elif node_type == Node.DISTANCE_SENSOR:
        distance_sensors[name] = device
        device.enable(TIME_STEP)
        manifest["sensors"].append(name)

supervisor.step(TIME_STEP)

# ==========================================
# Step 2: Save Initial State (Snapshot Tick 0)
# ==========================================
print("Saving initial simulation state (Tick 0)...")
robot_node.saveState("tick_0")

# ==========================================
# Step 3: Induce Initial Collision for Baseline
# ==========================================
SPEED = 4.0
for name, motor in actuators.items():
    motor.setVelocity(SPEED)

tick = 0
has_crashed = False

print("Simulation started. Monitoring agent trajectory...")

# ==========================================
# Step 4: Main Physics Execution Loop
# ==========================================
while supervisor.step(TIME_STEP) != -1:
    tick += 1

    position = translation_field.getSFVec3f()
    x, y, z = position[0], position[1], position[2]

    # Check for boundary collision
    if not has_crashed and (abs(x) > 0.45 or abs(y) > 0.45):
        has_crashed = True
        print(f"\nCollision detected at tick {tick}.")

        initial_distance = calculate_distance(x, y)
        print(f"Distance to target before fix: {initial_distance:.3f} meters")

        # Halt agent
        for name, motor in actuators.items():
            motor.setVelocity(0)

        # ==========================================
        # Step 5: Active Debugging and Retry Loop
        # ==========================================
        failed_attempts: List[Dict[str, Any]] = []
        MAX_RETRIES = 5
        mission_accomplished = False

        for attempt in range(1, MAX_RETRIES + 1):
            print(
                f"\nInitiating debugging phase: Attempt {attempt} of {MAX_RETRIES}..."
            )

            current_sensor_readings = {
                name: sensor.getValue() for name, sensor in distance_sensors.items()
            }

            # Construct the execution log with context and memory
            log = {
                "timestamp": time.time(),
                "simulation_tick": tick,
                "error_type": "BoundaryCollision",
                "message": "Robot collided with boundary.",
                "mission_objective": {
                    "goal": "reach_target",
                    "target_position": {"x": TARGET_X, "y": TARGET_Y},
                    "current_distance": initial_distance,
                },
                "hardware_manifest": manifest,
                "agent_state": {"position": {"x": x, "y": y, "z": z}},
                "sensor_data": {"distance_sensors": current_sensor_readings},
                "environment_context": {
                    "grid_boundaries": {"max_x": 0.45, "max_y": 0.45}
                },
                "failed_attempts": failed_attempts,  # Injecting historical context
            }

            with open("auto_failure_log.json", "w") as f:
                json.dump(log, f, indent=4)

            try:
                # Execute the Langchain AI Debugger
                # Note: Using explicit venv path as required by the environment
                subprocess.run(
                    [r"D:\AutoSim\venv\Scripts\python.exe", "langchain_brain.py"],
                    check=True,
                )

                with open("adjustment_command.json", "r") as f:
                    command = json.load(f)

                targets = command.get("target_parameters", {})
                print(f"AI Reasoning: {command.get('reasoning')}")

                # Rewind the simulation state to apply the patch
                print("Rewinding simulation timeline to Tick 0...")
                robot_node.loadState("tick_0")

                for actuator_name, new_velocity in targets.items():
                    if actuator_name in actuators:
                        actuators[actuator_name].setVelocity(new_velocity)
                        print(
                            f"  -> {actuator_name} velocity updated to {new_velocity}"
                        )

                print("Initiating parallel timeline observation (500 ticks)...")

                # ==========================================
                # Step 6: Outcome Evaluation
                # ==========================================
                survived_crash = True
                for t in range(1, 500):
                    if supervisor.step(TIME_STEP) == -1:
                        break
                    new_pos = translation_field.getSFVec3f()

                    # Check if the new parameters still result in a crash
                    if abs(new_pos[0]) > 0.45 or abs(new_pos[1]) > 0.45:
                        survived_crash = False
                        break

                # Grade the outcome based on survival and target proximity
                if not survived_crash:
                    print(
                        "Outcome A (Crash): The generated patch failed. Re-evaluating..."
                    )
                    failed_attempts.append(
                        {
                            "patch_applied": targets,
                            "failure_reason": "Crash: Robot hit the wall again. You must reverse or turn.",
                        }
                    )
                    continue  # Proceed to next retry

                else:
                    new_distance = calculate_distance(new_pos[0], new_pos[1])
                    print(
                        f"Final distance to target: {new_distance:.3f}m (Previous: {initial_distance:.3f}m)"
                    )

                    if new_distance >= initial_distance:
                        print(
                            "Outcome B (Suboptimal): Agent survived but moved away from the target. Re-evaluating..."
                        )
                        failed_attempts.append(
                            {
                                "patch_applied": targets,
                                "failure_reason": f"Cowardice: Distance to target increased from {initial_distance:.3f}m to {new_distance:.3f}m. You are backing up forever. Use asymmetric velocities (e.g. left: 2, right: -2) to turn towards the objective.",
                            }
                        )
                        continue  # Proceed to next retry

                    else:
                        print(
                            "Outcome C (Success): Agent survived and improved proximity to the target."
                        )
                        mission_accomplished = True
                        break  # Patch successful, exit retry loop

            except Exception as e:
                print(f"Error: The AI Debugger failed to execute properly: {e}")
                break

        # Final Evaluation Check
        if mission_accomplished:
            print("-" * 50)
            print("Mission Accomplished. Agent behavior successfully optimized.")
            print("-" * 50)
            supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            break
        else:
            print("\nCritical Failure: The AI exhausted all optimization retries.")
            supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            break
