from controller import Supervisor, Node
import json
import time
import sys
import subprocess
from typing import Dict, Any, List

# --- IMPORT OUR NEW DECOUPLED CORE ENGINE ---
from autosim_core import TelemetryTracker, DiagnosticEngine, EpisodeRecorder

# Initialize Webots Supervisor and core components
supervisor = Supervisor()
TIME_STEP = int(supervisor.getBasicTimeStep())
robot_node = supervisor.getFromDef("E_PUCK")
translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

TARGET = [2.0, 2.0, 0.0]

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

    if node_type in [Node.ROTATIONAL_MOTOR, Node.LINEAR_MOTOR]:
        actuators[name] = device
        device.setPosition(float("inf"))
        manifest["actuators"].append(name)
    elif node_type == Node.DISTANCE_SENSOR:
        distance_sensors[name] = device
        device.enable(TIME_STEP)
        manifest["sensors"].append(name)

supervisor.step(TIME_STEP)

if not actuators:
    print("\n[CRITICAL ERROR] No actuators discovered. Agent is a brick.")
    supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
    sys.exit(1)

# Initialize decoupled logic engines
dt_seconds = TIME_STEP / 1000.0
telemetry = TelemetryTracker(dt_seconds=dt_seconds, window_size=15)
diagnostician = DiagnosticEngine()
episode_recorder = EpisodeRecorder()  # <-- New!

# ==========================================
# Step 2: Save Initial State
# ==========================================
robot_node.saveState("tick_0")

# ==========================================
# Step 3: Induce Initial Action (DUMB BASELINE)
# ==========================================
SPEED = 4.0
for name, motor in actuators.items():
    motor.setVelocity(SPEED)

tick = 0
has_crashed = False

print("Simulation started. Monitoring telemetry, diagnostics, and episode history...")

# ==========================================
# Step 4: Main Physics Execution Loop
# ==========================================
while supervisor.step(TIME_STEP) != -1:
    tick += 1

    position = translation_field.getSFVec3f()
    rotation = rotation_field.getSFRotation()
    current_motor_vels = [m.getVelocity() for m in actuators.values()]

    # Update Math Engines
    telemetry.record_state(position, rotation, current_motor_vels)
    current_features = telemetry.get_features(TARGET)

    # Record for Episode Summary
    episode_recorder.record_tick(position, current_features)

    # Check for Failures
    is_failure, err_type, err_message = diagnostician.evaluate_state(
        position, current_features, telemetry.ready, current_motor_vels
    )

    if not has_crashed and is_failure:
        has_crashed = True
        print(f"\n[DIAGNOSTIC TRIGGER] Tick {tick} | Type: {err_type}")
        print(f"Details: {err_message}")

        initial_distance = current_features.get("spatial", {}).get(
            "distance_to_goal_m", 0
        )

        # Halt agent
        for name, motor in actuators.items():
            motor.setVelocity(0)

        failed_attempts: List[Dict[str, Any]] = []
        MAX_RETRIES = 5
        mission_accomplished = False

        # ==========================================
        # Step 5: Active Debugging Loop
        # ==========================================
        for attempt in range(1, MAX_RETRIES + 1):
            print(
                f"\nInitiating debugging phase: Attempt {attempt} of {MAX_RETRIES}..."
            )

            current_sensor_readings = {
                name: sensor.getValue() for name, sensor in distance_sensors.items()
            }

            # Build the God-Tier JSON Payload
            log = {
                "timestamp": time.time(),
                "error_type": err_type,
                "message": err_message,
                "mission_objective": {
                    "goal": "reach_target",
                    "target_position": TARGET,
                },
                "hardware_manifest": manifest,
                "current_crash_state": {
                    "raw_position": {
                        "x": position[0],
                        "y": position[1],
                        "z": position[2],
                    },
                    "sensor_data": current_sensor_readings,
                },
                "engineered_features": current_features,
                "episode_summary": episode_recorder.get_summary(
                    initial_distance
                ),  # <-- Injecting the summary!
                "failed_attempts": failed_attempts,
            }

            with open("auto_failure_log.json", "w") as f:
                json.dump(log, f, indent=4)

            try:
                subprocess.run(
                    [r"D:\AutoSim\venv\Scripts\python.exe", "langchain_brain.py"],
                    check=True,
                )

                with open("adjustment_command.json", "r") as f:
                    command = json.load(f)

                targets = command.get("target_parameters", {})
                print(f"AI Reasoning: {command.get('reasoning')}")

                # Rewind and Reset Enignes
                robot_node.loadState("tick_0")
                telemetry = TelemetryTracker(dt_seconds=dt_seconds, window_size=15)
                episode_recorder.reset()  # Restart the summary tracker for the test run

                for actuator_name, new_velocity in targets.items():
                    if actuator_name in actuators:
                        actuators[actuator_name].setVelocity(new_velocity)
                        print(
                            f"  -> {actuator_name} velocity updated to {new_velocity}"
                        )

                # ==========================================
                # Step 6: Outcome Evaluation
                # ==========================================
                survived_crash = True
                for t in range(1, 500):
                    if supervisor.step(TIME_STEP) == -1:
                        break

                    test_pos = translation_field.getSFVec3f()
                    test_rot = rotation_field.getSFRotation()
                    test_vels = [m.getVelocity() for m in actuators.values()]

                    telemetry.record_state(test_pos, test_rot, test_vels)
                    test_features = telemetry.get_features(TARGET)
                    episode_recorder.record_tick(
                        test_pos, test_features
                    )  # Track test run too

                    test_fail, test_err, test_msg = diagnostician.evaluate_state(
                        test_pos, test_features, telemetry.ready, test_vels
                    )

                    if test_fail:
                        survived_crash = False
                        break

                if not survived_crash:
                    print(
                        f"Outcome A (Failure): Agent triggered another error ({test_err}). Re-evaluating..."
                    )
                    failed_attempts.append(
                        {
                            "patch_applied": targets,
                            "failure_reason": f"{test_err}: {test_msg}",
                        }
                    )
                    continue
                else:
                    new_distance = test_features["spatial"]["distance_to_goal_m"]
                    print(
                        f"Final distance to target: {new_distance:.3f}m (Previous: {initial_distance:.3f}m)"
                    )

                    if new_distance >= initial_distance:
                        print(
                            "Outcome B (Suboptimal): Agent survived but moved away from target."
                        )
                        failed_attempts.append(
                            {
                                "patch_applied": targets,
                                "failure_reason": "Distance increased. Turn towards target.",
                            }
                        )
                        continue
                    else:
                        print(
                            "Outcome C (Success): Agent improved proximity to target."
                        )
                        mission_accomplished = True
                        break

            except Exception as e:
                print(f"Error: AI Debugger failed: {e}")
                break

        if mission_accomplished:
            print("-" * 50)
            print("Mission Accomplished.")
            supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            break
        else:
            print("\nCritical Failure: AI exhausted retries.")
            supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            break
