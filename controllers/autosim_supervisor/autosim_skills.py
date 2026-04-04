import math
from typing import Dict, Any, List
from autosim_controllers import DifferentialDriveController


class SkillStatus:
    """Standardized enums so the supervisor knows when to kill a skill."""

    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class Skill:
    """
    Layer 5: Base Skill Interface.
    Every skill the LLM uses MUST inherit from this and implement these methods.
    """

    def step(
        self,
        pos: List[float],
        rot: List[float],
        features: Dict[str, Any],
        target: List[float],
        dt: float,
    ) -> Dict[str, float]:
        """Executes one tick of logic and returns a dictionary of actuator commands."""
        raise NotImplementedError

    def get_status(self) -> str:
        """Returns the current state of the skill."""
        raise NotImplementedError

    def reset(self):
        """Clears memory for rewinding simulations."""
        raise NotImplementedError


class DriveToTargetSkill(Skill):
    """
    Drives a differential-drive robot to a specific coordinate.
    The LLM will interact with this by providing the kp, ki, kd, and base_speed!
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        base_speed: float = 3.0,
        distance_threshold: float = 0.05,
        max_safe_speed: float = 6.28,
        left_motor_name: str = "left wheel motor",
        right_motor_name: str = "right wheel motor",
    ):

        # Instantiate the Layer 4 deterministic math controller!
        self.controller = DifferentialDriveController(
            kp=kp, ki=ki, kd=kd, max_motor_speed=max_safe_speed
        )

        self.base_speed = base_speed
        self.distance_threshold = distance_threshold
        self.max_safe_speed = max_safe_speed

        self.left_motor_name = left_motor_name
        self.right_motor_name = right_motor_name

        self.status = SkillStatus.RUNNING

    def _get_yaw_from_webots_rot(self, rot: List[float]) -> float:
        """
        Extracts the yaw angle from Webots' standard [x, y, z, angle] rotation vector.
        Assuming Z is the vertical axis, yaw is angle * sign(z).
        """
        if rot and len(rot) >= 4:
            return rot[3] * (1.0 if rot[2] >= 0 else -1.0)
        return 0.0

    def step(
        self,
        pos: List[float],
        rot: List[float],
        features: Dict[str, Any],
        target: List[float],
        dt: float,
    ) -> Dict[str, float]:
        """
        The core execution loop.
        Returns exactly what velocities should be applied to which motors.
        """
        # 1. Early Exit if already finished or failed
        if self.status != SkillStatus.RUNNING:
            return {self.left_motor_name: 0.0, self.right_motor_name: 0.0}

        # CRITIQUE 5 & 7 FIX: Defensive Input Guards
        # Wait for the telemetry window to fill or protect against empty arrays
        if not features or not pos or len(pos) < 2 or not target or len(target) < 2:
            return {self.left_motor_name: 0.0, self.right_motor_name: 0.0}

        # CRITIQUE 3 FIX: Self-Reporting Failure
        # If the skill realizes it is violently unstable, it aborts itself.
        wobble = features.get("system_health", {}).get("rotational_volatility", 0)
        if wobble > 0.5:
            self.status = SkillStatus.FAILURE
            return {self.left_motor_name: 0.0, self.right_motor_name: 0.0}

        # 2. Check Objective Success
        dist = features.get("spatial", {}).get("distance_to_goal_m", float("inf"))
        if dist < self.distance_threshold:
            self.status = SkillStatus.SUCCESS
            # Send a stop command as the final action
            return {self.left_motor_name: 0.0, self.right_motor_name: 0.0}

        # 3. Extract spatial math parameters
        current_yaw_rad = self._get_yaw_from_webots_rot(rot)
        current_heading_deg = math.degrees(current_yaw_rad)

        # Calculate angle to target using pure trig (atan2)
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        target_heading_deg = math.degrees(math.atan2(dy, dx))

        # 4. Trigger the Math Controller
        left_vel, right_vel = self.controller.compute(
            current_heading_deg=current_heading_deg,
            target_heading_deg=target_heading_deg,
            base_speed=self.base_speed,
            dt=dt,
        )

        # 5. LAYER 6: VALIDATION & SAFETY LAYER
        left_vel = max(-self.max_safe_speed, min(self.max_safe_speed, left_vel))
        right_vel = max(-self.max_safe_speed, min(self.max_safe_speed, right_vel))

        # 6. Map to exact hardware actuator names
        return {self.left_motor_name: left_vel, self.right_motor_name: right_vel}

    def get_status(self) -> str:
        return self.status

    def reset(self):
        """CRITIQUE 4 FIX: Clears internal memory when simulation rewinds."""
        self.status = SkillStatus.RUNNING
        self.controller.reset()
