from controller import Supervisor, Node
import json
import time
import os
import subprocess
import sys
import math

supervisor = Supervisor()
TIME_STEP = int(supervisor.getBasicTimeStep())
robot_node = supervisor.getFromDef("E_PUCK")
translation_field = robot_node.getField("translation")

# THE MISSION OBJECTIVE
TARGET_X = 2.0
TARGET_Y = 2.0


def calculate_distance(x, y):
    return math.sqrt((x - TARGET_X) ** 2 + (y - TARGET_Y) ** 2)


print("==================================================")
print(" AUTOSIM MISSION OBJECTIVE ENGINE INITIATED")
print("==================================================")

# 1. DYNAMIC HARDWARE DISCOVERY
actuators = {}
distance_sensors = {}
manifest = {"actuators": [], "sensors": []}

num_devices = supervisor.getNumberOfDevices()
for i in range(num_devices):
    device = supervisor.getDeviceByIndex(i)
    node_type = device.getNodeType()
    name = device.getName()

    if node_type in [Node.ROTATIONAL_MOTOR, Node.LINEAR_MOTOR]:
        actuators[name] = device
        device.setPosition(float("inf"))
        manifest["actuators"].append(name)

    elif node_type == Node.DISTANCE_SENSOR:
        distance_sensors[name] = device
        device.enable(TIME_STEP)
        manifest["sensors"].append(name)

supervisor.step(TIME_STEP)

# 2. THE TIME MACHINE (Snapshot Tick 0)
print("[WEBOTS TIME MACHINE] Saving initial timeline state...")
robot_node.saveState("tick_0")

# 3. FORCE A CRASH
SPEED = 4.0
for name, motor in actuators.items():
    motor.setVelocity(SPEED)

tick = 0
has_crashed = False

print("[WEBOTS WATCHER] Sim started. Agent moving to impact...")

# 4. The Main Physics Loop
while supervisor.step(TIME_STEP) != -1:
    tick += 1

    position = translation_field.getSFVec3f()
    x, y, z = position[0], position[1], position[2]

    if not has_crashed and (abs(x) > 0.45 or abs(y) > 0.45):
        has_crashed = True
        print(f"\n[WEBOTS WATCHER] FATAL COLLISION AT TICK {tick}!")

        initial_distance = calculate_distance(x, y)
        print(f"[WATCHER] Distance to target before fix: {initial_distance:.3f} meters")

        for name, motor in actuators.items():
            motor.setVelocity(0)

        # 5. THE ACTIVE RETRY LOOP (Goldfish Memory)
        failed_attempts = []
        MAX_RETRIES = 5
        mission_accomplished = False

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"\n>>> DEBUGGING PHASE: ATTEMPT {attempt}/{MAX_RETRIES} <<<")

            current_sensor_readings = {
                name: sensor.getValue() for name, sensor in distance_sensors.items()
            }

            # Build the Universal Log with Mission & Memory
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
                "failed_attempts": failed_attempts,  # Injecting the feedback!
            }

            with open("auto_failure_log.json", "w") as f:
                json.dump(log, f, indent=4)

            try:
                # Use your specific venv path here if sys.executable fails!
                subprocess.run(
                    [r"D:\AutoSim\venv\Scripts\python.exe", "langchain_brain.py"],
                    check=True,
                )

                with open("adjustment_command.json", "r") as f:
                    command = json.load(f)

                targets = command.get("target_parameters", {})
                print(f"[WEBOTS PLUMBER] AI Reasoning: {command.get('reasoning')}")

                # --- THE TIME MACHINE REWIND ---
                print("[WEBOTS TIME MACHINE] REWINDING TIMELINE TO TICK 0...")
                robot_node.loadState("tick_0")

                for actuator_name, new_velocity in targets.items():
                    if actuator_name in actuators:
                        actuators[actuator_name].setVelocity(new_velocity)
                        print(f"  -> {actuator_name} set to {new_velocity}")

                print(">>> INITIATING PARALLEL TIMELINE OBSERVATION (500 TICKS) <<<")

                # 6. THE FITNESS JUDGE
                survived_crash = True
                for t in range(1, 500):
                    if supervisor.step(TIME_STEP) == -1:
                        break
                    new_pos = translation_field.getSFVec3f()

                    if abs(new_pos[0]) > 0.45 or abs(new_pos[1]) > 0.45:
                        survived_crash = False
                        break

                # Grade the outcome
                if not survived_crash:
                    print("[NEW TIMELINE] 💀 OUTCOME A (CRASH): The AI's patch failed.")
                    failed_attempts.append(
                        {
                            "patch_applied": targets,
                            "failure_reason": "Crash: Robot hit the wall again. You must reverse or turn.",
                        }
                    )
                    continue  # Try again!

                else:
                    new_distance = calculate_distance(new_pos[0], new_pos[1])
                    print(
                        f"[NEW TIMELINE] Final distance to target: {new_distance:.3f}m (Old: {initial_distance:.3f}m)"
                    )

                    if new_distance >= initial_distance:
                        print(
                            "[NEW TIMELINE] ⚠️ OUTCOME B (COWARDICE): Survived, but moved away from target."
                        )
                        failed_attempts.append(
                            {
                                "patch_applied": targets,
                                "failure_reason": f"Cowardice: Distance to target increased from {initial_distance:.3f}m to {new_distance:.3f}m. You are backing up forever. Use asymmetric velocities (e.g. left: 2, right: -2) to turn towards the objective.",
                            }
                        )
                        continue  # Try again!

                    else:
                        print(
                            "[NEW TIMELINE] 🎯 OUTCOME C (SUCCESS): Survived AND moved closer to target!"
                        )
                        mission_accomplished = True
                        break  # Exit the retry loop!

            except Exception as e:
                print(f"[ERROR] The Brain failed to execute: {e}")
                break

        if mission_accomplished:
            print("\n==================================================")
            print(" MISSION ACCOMPLISHED. THE AGENT IS OPTIMIZING.")
            print("==================================================")
            supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            break
        else:
            print(
                "\n[CRITICAL FAILURE] The AI exhausted all retries and failed to optimize."
            )
            supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            break
