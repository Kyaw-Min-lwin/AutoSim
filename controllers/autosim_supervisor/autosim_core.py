import math
from collections import deque
from typing import Dict, Any, List, Optional, Tuple


class TelemetryTracker:
    """
    Hardware-agnostic state observer.
    Maintains a rolling window of physics states to derive advanced kinematics.
    """

    def __init__(self, dt_seconds: float, window_size: int = 10):
        self.dt = dt_seconds
        self.window_size = window_size
        self.pos_dims: Optional[int] = None
        self.rot_dims: Optional[int] = None
        self.positions = deque(maxlen=window_size)
        self.rotations = deque(maxlen=window_size)
        self.raw_velocities = deque(maxlen=window_size)
        self.raw_accelerations = deque(maxlen=window_size)
        self.actuator_effort_penalty = 0.0
        self.ready = False

    def _get_smoothed_vector(self, vector_deque: deque, dims: int) -> List[float]:
        if not vector_deque:
            return [0.0] * dims
        avg = [0.0] * dims
        for vec in vector_deque:
            for i in range(dims):
                avg[i] += vec[i]
        return [val / len(vector_deque) for val in avg]

    def record_state(self, pos: List[float], rot: List[float], motor_vels: List[float]):
        if not pos or motor_vels is None:
            return

        if self.pos_dims is None:
            self.pos_dims = len(pos)
        elif len(pos) != self.pos_dims:
            raise ValueError(f"CRITICAL: Position dimension mismatch!")

        if rot is not None and len(rot) > 0:
            if self.rot_dims is None:
                self.rot_dims = len(rot)
            elif len(rot) != self.rot_dims:
                raise ValueError(f"CRITICAL: Rotation dimension mismatch!")
            self.rotations.append(rot)

        self.positions.append(pos)

        if len(self.positions) == self.window_size:
            self.ready = True

        curr_vel = [0.0] * self.pos_dims
        if len(self.positions) >= 2:
            prev_pos = self.positions[-2]
            curr_vel = [(p - pp) / self.dt for p, pp in zip(pos, prev_pos)]
        self.raw_velocities.append(curr_vel)

        curr_acc = [0.0] * self.pos_dims
        if len(self.raw_velocities) >= 2:
            prev_vel = self.raw_velocities[-2]
            curr_acc = [(v - pv) / self.dt for v, pv in zip(curr_vel, prev_vel)]
        self.raw_accelerations.append(curr_acc)

        self.actuator_effort_penalty += sum(abs(v) for v in motor_vels) * self.dt

    def get_features(self, target_pos: List[float]) -> Dict[str, Any]:
        if not self.positions or self.pos_dims is None:
            return {}

        pos = self.positions[-1]
        safe_target = list(target_pos)
        if len(safe_target) < self.pos_dims:
            safe_target += [0.0] * (self.pos_dims - len(safe_target))
        elif len(safe_target) > self.pos_dims:
            safe_target = safe_target[: self.pos_dims]

        vel = self._get_smoothed_vector(self.raw_velocities, self.pos_dims)
        acc = self._get_smoothed_vector(self.raw_accelerations, self.pos_dims)

        speed = math.sqrt(sum(v**2 for v in vel))
        accel_mag = math.sqrt(sum(a**2 for a in acc))

        jerk_mag = 0.0
        if len(self.raw_accelerations) >= 2:
            oldest_acc = self.raw_accelerations[0]
            newest_acc = self.raw_accelerations[-1]
            time_delta = self.dt * (len(self.raw_accelerations) - 1)
            if time_delta > 0:
                jerk_vec = [
                    (n - o) / time_delta for n, o in zip(newest_acc, oldest_acc)
                ]
                jerk_mag = math.sqrt(sum(j**2 for j in jerk_vec))

        dir_to_target = [t - p for t, p in zip(safe_target, pos)]
        distance = math.sqrt(sum(d**2 for d in dir_to_target))

        vector_alignment = 0.0
        if distance > 0.001 and speed > 0.001:
            dot_product = sum(d * v for d, v in zip(dir_to_target, vel))
            vector_alignment = dot_product / (distance * speed)

        rot_volatility = 0.0
        if len(self.rotations) >= 2 and self.rot_dims is not None:
            diffs = []
            for i in range(1, len(self.rotations)):
                prev_r = self.rotations[i - 1]
                curr_r = self.rotations[i]
                diff = math.sqrt(sum((c - p) ** 2 for c, p in zip(curr_r, prev_r)))
                diffs.append(diff)
            rot_volatility = sum(diffs) / len(diffs)

        return {
            "kinematics": {
                "speed_m_s": round(speed, 4),
                "acceleration_m_s2": round(accel_mag, 4),
                "jerk_m_s3": round(jerk_mag, 4),
            },
            "spatial": {
                "distance_to_goal_m": round(distance, 4),
                "target_alignment_score": round(vector_alignment, 3),
            },
            "system_health": {
                "actuator_effort_penalty": round(self.actuator_effort_penalty, 4),
                "rotational_volatility": round(rot_volatility, 6),
            },
        }


class DiagnosticEngine:
    """Pre-classifies state failures based on mathematically derived telemetry thresholds."""

    def __init__(self):
        self.boundary_limit = 0.45
        self.min_speed = 0.01
        self.drift_speed_min = 0.05
        self.fleeing_alignment = -0.5
        self.max_wobble = 0.5
        self.max_jerk = 50.0
        self.stuck_effort_min = 1.0

    def evaluate_state(
        self,
        pos: List[float],
        features: Dict[str, Any],
        is_telemetry_ready: bool,
        current_motor_vels: List[float],
    ) -> Tuple[bool, str, str]:
        if any(abs(p) > self.boundary_limit for p in pos):
            return (
                True,
                "BoundaryCollision",
                "Agent breached the operational geofence. It hit a wall or ceiling.",
            )

        if not is_telemetry_ready or not features:
            return False, "Nominal", "Initializing..."

        k, s, h = (
            features.get("kinematics", {}),
            features.get("spatial", {}),
            features.get("system_health", {}),
        )

        current_effort = sum(abs(v) for v in current_motor_vels)
        if (
            k.get("speed_m_s", 0) < self.min_speed
            and current_effort > self.stuck_effort_min
        ):
            return (
                True,
                "KineticStagnation",
                f"Motors are driving (Effort: {current_effort:.2f}), but speed is negligible. Agent is stuck.",
            )

        if (
            s.get("target_alignment_score", 0) < self.fleeing_alignment
            and k.get("speed_m_s", 0) > self.drift_speed_min
        ):
            return (
                True,
                "SevereDrift",
                f"Agent is moving away from target (Alignment: {s.get('target_alignment_score')}).",
            )

        if h.get("rotational_volatility", 0) > self.max_wobble:
            return (
                True,
                "DynamicInstability",
                "Agent is exhibiting severe rotational wobble or oscillation.",
            )

        if k.get("jerk_m_s3", 0) > self.max_jerk:
            return (
                True,
                "Thrashing",
                "Agent is experiencing extreme jerk. Movement is highly erratic.",
            )

        return False, "Nominal", "Agent is operating within acceptable parameters."


class EpisodeRecorder:
    """
    Compresses thousands of ticks into a lightweight LLM-friendly summary.
    Tracks max extremes and downsamples the trajectory with temporal tracking.
    """

    def __init__(self, max_breadcrumbs: int = 100):
        self.max_breadcrumbs = max_breadcrumbs
        self.reset()

    def reset(self):
        self.tick_count = 0
        self.max_speed = 0.0
        self.max_jerk = 0.0
        self.max_wobble = 0.0
        # Enforce strict trajectory bound to protect LLM Context Window
        self.trajectory_sample = deque(maxlen=self.max_breadcrumbs)

    def record_tick(self, pos: List[float], features: Dict[str, Any]):
        self.tick_count += 1

        # Downsample: Save position only once every 20 ticks
        if self.tick_count % 20 == 0:
            #  Added tick_count as the first element for temporal awareness
            #  [p for p in pos] handles N-dimensions perfectly fine.
            breadcrumb = [self.tick_count] + [round(p, 3) for p in pos]
            self.trajectory_sample.append(breadcrumb)

        if not features:
            return

        spd = features.get("kinematics", {}).get("speed_m_s", 0)
        jerk = features.get("kinematics", {}).get("jerk_m_s3", 0)
        wobble = features.get("system_health", {}).get("rotational_volatility", 0)

        if spd > self.max_speed:
            self.max_speed = spd
        if jerk > self.max_jerk:
            self.max_jerk = jerk
        if wobble > self.max_wobble:
            self.max_wobble = wobble

    def get_summary(self, final_distance: float) -> Dict[str, Any]:
        return {
            "episode_duration_ticks": self.tick_count,
            "final_distance_to_target_m": round(final_distance, 3),
            "extremes": {
                "max_speed_m_s": round(self.max_speed, 3),
                "max_jerk_m_s3": round(self.max_jerk, 3),
                "max_wobble": round(self.max_wobble, 3),
            },
            # Deques must be cast back to lists for JSON serialization
            "trajectory_breadcrumbs": list(self.trajectory_sample),
        }
