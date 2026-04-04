from controller import Supervisor, Node
import json
import time
import sys
import subprocess
from typing import Dict, Any, List

# --- IMPORT OUR NEW DECOUPLED STACK ---
from autosim_core import TelemetryTracker, DiagnosticEngine, EpisodeRecorder
from autosim_skills import DriveToTargetSkill, SkillStatus

# Initialize Webots Supervisor and core components
supervisor = Supervisor()
TIME_STEP = int(supervisor.getBasicTimeStep())
robot_node = supervisor.getFromDef("E_PUCK")
translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

TARGET = [2.0, 2.0, 0.0]

print("-" * 50)
print("Initializing AutoSim Skill-Based Autonomy Engine...")
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

# CRITIQUE 5 FIX: Safely map motors by name instead of assuming index order
left_motor_name = next(
    (n for n in manifest["actuators"] if "left" in n.lower()), manifest["actuators"][0]
)
right_motor_name = next(
    (n for n in manifest["actuators"] if "right" in n.lower()),
    manifest["actuators"][-1],
)

# Initialize decoupled logic engines
dt_seconds = TIME_STEP / 1000.0
telemetry = TelemetryTracker(dt_seconds=dt_seconds, window_size=15)
diagnostician = DiagnosticEngine()
episode_recorder = EpisodeRecorder()

# ==========================================
# Step 2: Save Initial State
# ==========================================
robot_node.saveState("tick_0")

# ==========================================
# Step 3: Instantiate Initial Skill (Unoptimized Baseline)
# ==========================================
current_skill = DriveToTargetSkill(
    kp=0.5,
    ki=0.0,
    kd=0.0,
    base_speed=5.0,
    left_motor_name=left_motor_name,
    right_motor_name=right_motor_name,
)

tick = 0
has_crashed = False

print(f"Simulation started. Executing Skill: {current_skill.__class__.__name__}...")

# ==========================================
# Step 4: Main Physics Execution Loop
# ==========================================
while supervisor.step(TIME_STEP) != -1:
    tick += 1

    position = translation_field.getSFVec3f()
    rotation = rotation_field.getSFRotation()
    current_motor_vels = [m.getVelocity() for m in actuators.values()]

    # Update Math & Logging Engines
    telemetry.record_state(position, rotation, current_motor_vels)
    current_features = telemetry.get_features(TARGET)
    episode_recorder.record_tick(position, current_features)

    # Ask the Skill for the next motor commands
    motor_commands = current_skill.step(
        pos=position,
        rot=rotation,
        features=current_features,
        target=TARGET,
        dt=dt_seconds,
    )

    # Apply Layer 7 Execution (Execute the validated commands)
    for actuator_name, velocity in motor_commands.items():
        if actuator_name in actuators:
            actuators[actuator_name].setVelocity(velocity)

    # Check for Failures
    is_failure, err_type, err_message = diagnostician.evaluate_state(
        position, current_features, telemetry.ready, current_motor_vels
    )

    skill_status = current_skill.get_status()
    if skill_status == SkillStatus.FAILURE and not is_failure:
        is_failure = True
        err_type = "SkillSelfAbort"
        err_message = "The skill detected internal instability and aborted execution."

    if skill_status == SkillStatus.SUCCESS:
        print("\n[OBJECTIVE REACHED] The Skill successfully navigated to the target!")
        supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        break

    if not has_crashed and is_failure:
        has_crashed = True
        print(f"\n[DIAGNOSTIC TRIGGER] Tick {tick} | Type: {err_type}")
        print(f"Details: {err_message}")

        # CRITIQUE 3 FIX: Safe infinity fallback instead of 0
        initial_distance = current_features.get("spatial", {}).get(
            "distance_to_goal_m", float("inf")
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
            print(f"\nInitiating tuning phase: Attempt {attempt} of {MAX_RETRIES}...")

            current_sensor_readings = {
                name: sensor.getValue() for name, sensor in distance_sensors.items()
            }

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
                "episode_summary": episode_recorder.get_summary(initial_distance),
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

                tuning_params = command.get("target_parameters", {})
                print(f"AI Reasoning: {command.get('reasoning')}")
                print(f"Applying new Skill Parameters: {tuning_params}")

                # Rewind and Reset Engines
                robot_node.loadState("tick_0")
                telemetry = TelemetryTracker(dt_seconds=dt_seconds, window_size=15)
                episode_recorder.reset()

                current_skill = DriveToTargetSkill(
                    kp=tuning_params.get("kp", 1.0),
                    ki=tuning_params.get("ki", 0.0),
                    kd=tuning_params.get("kd", 0.0),
                    base_speed=tuning_params.get("base_speed", 3.0),
                    left_motor_name=left_motor_name,
                    right_motor_name=right_motor_name,
                )

                # ==========================================
                # Step 6: Outcome Evaluation
                # ==========================================
                survived_crash = True
                for t in range(1, 1000):
                    if supervisor.step(TIME_STEP) == -1:
                        break

                    test_pos = translation_field.getSFVec3f()
                    test_rot = rotation_field.getSFRotation()
                    test_vels = [m.getVelocity() for m in actuators.values()]

                    telemetry.record_state(test_pos, test_rot, test_vels)
                    test_features = telemetry.get_features(TARGET)
                    episode_recorder.record_tick(test_pos, test_features)

                    test_motor_commands = current_skill.step(
                        test_pos, test_rot, test_features, TARGET, dt_seconds
                    )
                    for act_name, vel in test_motor_commands.items():
                        if act_name in actuators:
                            actuators[act_name].setVelocity(vel)

                    test_fail, test_err, test_msg = diagnostician.evaluate_state(
                        test_pos, test_features, telemetry.ready, test_vels
                    )
                    test_skill_status = current_skill.get_status()

                    if test_fail or test_skill_status == SkillStatus.FAILURE:
                        survived_crash = False
                        break

                    if test_skill_status == SkillStatus.SUCCESS:
                        survived_crash = True
                        break

                if not survived_crash:
                    fail_reason = (
                        test_msg
                        if test_fail
                        else "Skill destabilized and self-aborted."
                    )
                    print(
                        f"Outcome A (Failure): Agent triggered another error. Re-evaluating..."
                    )
                    failed_attempts.append(
                        {"patch_applied": tuning_params, "failure_reason": fail_reason}
                    )
                    continue
                else:
                    new_distance = test_features.get("spatial", {}).get(
                        "distance_to_goal_m", float("inf")
                    )

                    # CRITIQUE 6 FIX: Cleaned up the outcome logic tree
                    if test_skill_status == SkillStatus.SUCCESS:
                        print(
                            "Outcome C (Absolute Success): Agent successfully reached the target using tuned parameters!"
                        )
                        mission_accomplished = True
                        break
                    elif new_distance >= initial_distance:
                        print(
                            "Outcome B (Suboptimal): Agent survived but moved away from target."
                        )
                        failed_attempts.append(
                            {
                                "patch_applied": tuning_params,
                                "failure_reason": "Distance increased. Turn towards target.",
                            }
                        )
                        continue
                    else:
                        print(
                            "Outcome C (Partial Success): Agent improved proximity to target but ran out of evaluation time."
                        )
                        mission_accomplished = True
                        break

            except Exception as e:
                print(f"Error: AI Debugger failed: {e}")
                break

        if mission_accomplished:
            print("-" * 50)
            print("Mission Tuning Accomplished.")
            supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            break
        else:
            print("\nCritical Failure: AI exhausted retries.")
            supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            break
