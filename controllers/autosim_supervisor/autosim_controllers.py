import math
from typing import List


class DifferentialDriveController:
    """
    Layer 4: Deterministic Math Controller for Differential Drive Robots.
    Pure math. No simulation dependencies. Hardened against PID windup and saturation.
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        max_motor_speed: float = 6.28,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.max_motor_speed = max_motor_speed

        self.integral = 0.0
        self.prev_error = 0.0

        # Anti-Windup Limit: Stop the integral from growing to infinity if the robot gets stuck.
        # We dynamically set a reasonable ceiling based on the max motor speed.
        self.integral_limit = max_motor_speed / (ki if ki > 0.001 else 1.0)

    def compute(
        self,
        current_heading_deg: float,
        target_heading_deg: float,
        base_speed: float,
        dt: float,
    ) -> List[float]:
        """
        Calculates left and right wheel velocities required to reach target heading.
        """
        # 1. Calculate heading error (Find the shortest path) in degrees
        error_deg = (target_heading_deg - current_heading_deg + 180) % 360 - 180

        # CRITIQUE 1 FIX (Yours!): Dimensional Consistency.
        # Convert degrees to radians so it scales properly with motor rad/s.
        error_rad = math.radians(error_deg)

        # 2. PID Terms
        self.integral += error_rad * dt

        # CRITIQUE 2 FIX: Integral Anti-Windup
        # Clamp the integral memory so it doesn't explode during physical collisions.
        self.integral = max(
            -self.integral_limit, min(self.integral_limit, self.integral)
        )

        derivative = (error_rad - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error_rad

        # 3. Calculate rotational adjustment
        turn_rate = (
            (self.kp * error_rad) + (self.ki * self.integral) + (self.kd * derivative)
        )

        # CRITIQUE 3 FIX: Clamp the turn rate BEFORE blending.
        # This prevents a massive turn rate from entirely swallowing proportional control.
        turn_rate = max(-self.max_motor_speed, min(self.max_motor_speed, turn_rate))

        # CRITIQUE 7 FIX: Sanitize the base speed.
        safe_base_speed = max(
            -self.max_motor_speed, min(self.max_motor_speed, base_speed)
        )

        # 4. Differential Drive Kinematics (Blending)
        left_vel = safe_base_speed + turn_rate
        right_vel = safe_base_speed - turn_rate

        # 5. Hardware Safety Limiter (Final Output Clamp)
        left_vel = max(-self.max_motor_speed, min(self.max_motor_speed, left_vel))
        right_vel = max(-self.max_motor_speed, min(self.max_motor_speed, right_vel))

        return [left_vel, right_vel]

    def reset(self):
        """Clears the PID memory for rewinding simulations."""
        self.integral = 0.0
        self.prev_error = 0.0
