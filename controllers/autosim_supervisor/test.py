from controller import Supervisor, Node
import sys
import math

# ==========================================
# Step 1: Initialize Engine & Targets
# ==========================================
supervisor = Supervisor()
TIME_STEP = int(supervisor.getBasicTimeStep())
robot_node = supervisor.getFromDef("E_PUCK")

translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")
 
TARGET = [0.8, 0.8, 0.0]
MAX_SPEED = 6.28

# --- INIT PRINTS ---
print("-" * 60)
print("Initializing Manual P-Controller Test...")
print(f"Target Coordinates: X={TARGET[0]}, Y={TARGET[1]}")
print("-" * 60)

# ==========================================
# Step 2: Dynamic Hardware Discovery
# ==========================================
actuators = {}
manifest = {"actuators": []}

num_devices = supervisor.getNumberOfDevices()
for i in range(num_devices):
    device = supervisor.getDeviceByIndex(i)
    node_type = device.getNodeType()
    name = device.getName()

    if node_type in [Node.ROTATIONAL_MOTOR, Node.LINEAR_MOTOR]:
        actuators[name] = device
        device.setPosition(float("inf"))
        manifest["actuators"].append(name)

if not actuators:
    print("\n[CRITICAL ERROR] No actuators discovered. Agent is a brick.")
    supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
    sys.exit(1)

left_motor_name = next(
    (n for n in manifest["actuators"] if "left" in n.lower()), manifest["actuators"][0]
)
right_motor_name = next(
    (n for n in manifest["actuators"] if "right" in n.lower()),
    manifest["actuators"][-1],
)

left_motor = actuators[left_motor_name]
right_motor = actuators[right_motor_name]

# --- HARDWARE PRINTS ---
print(
    f"[SYSTEM] Hardware linked. Left: '{left_motor_name}', Right: '{right_motor_name}'"
)
print("[SYSTEM] Starting control loop...\n")

# ==========================================
# Step 3: The P-Controller Loop
# ==========================================
# K_rho = 2.5
# K_alpha = 4.0

step_counter = 0  # Prevents console spam

while supervisor.step(TIME_STEP) != -1:
    current_pos = translation_field.getSFVec3f()
    current_rot = rotation_field.getSFRotation()

    x = current_pos[0]
    y = current_pos[1]

    current_heading = current_rot[3]
    if current_rot[2] < 0:
        current_heading = -current_heading

    dx = TARGET[0] - x
    dy = TARGET[1] - y
    distance = math.sqrt(dx**2 + dy**2)
    target_heading = math.atan2(dy, dx)

    heading_error = target_heading - current_heading
    heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

    if distance > 0.02:
        # forward_velocity = K_rho * distance
        # rotational_velocity = K_alpha * heading_error

        # left_speed = forward_velocity - rotational_velocity
        # right_speed = forward_velocity + rotational_velocity

        # left_speed = max(min(left_speed, MAX_SPEED), -MAX_SPEED)
        # right_speed = max(min(right_speed, MAX_SPEED), -MAX_SPEED)

        # --- NEW CONTROL LAW ---
        K_rho = 1.0
        K_alpha = 2.0
        K_beta = -0.5  # stabilizes final orientation

        beta = -current_heading - heading_error

        forward_velocity = K_rho * distance
        rotational_velocity = K_alpha * heading_error + K_beta * beta

        left_speed = forward_velocity - rotational_velocity
        right_speed = forward_velocity + rotational_velocity

        # --- TELEMETRY DASHBOARD (Prints every ~0.3 seconds) ---
        if step_counter % 10 == 0:
            # Convert heading error to degrees for easier human reading
            h_err_deg = math.degrees(heading_error)
            print(
                f"Pos: [{x:.3f}, {y:.3f}] | Dist: {distance:.3f}m | H-Err: {h_err_deg:>6.1f}° | Motors: L={left_speed:>5.2f}, R={right_speed:>5.2f}"
            )

    else:
        left_speed = 0.0
        right_speed = 0.0

        # --- SUCCESS PRINT ---
        print("-" * 60)
        print(f"[SUCCESS] Target Reached at [{x:.3f}, {y:.3f}].")
        print(f"Final Error Margin: {distance:.4f}m")
        print("-" * 60)

        # Stop the motors and break the loop so it doesn't spam the success message forever
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        break

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    step_counter += 1
