# PID CONTROL CHEAT SHEET — AUTOSIM (DIFFERENTIAL DRIVE)

## 1. SYSTEM CONTEXT

This PID controller is used to control **heading (turning)** of a differential drive robot.

Controller output = **turn_rate**

Motor mapping:

* left_velocity  = base_speed + turn_rate
* right_velocity = base_speed - turn_rate

Implication:

* turn_rate > 0 → robot turns RIGHT
* turn_rate < 0 → robot turns LEFT

---

## 2. CORE PID BEHAVIOR

### Proportional (Kp)

* Reacts to current error
* Increases responsiveness

Effects:

* Faster turning toward target

- Too high → oscillation / overshoot / wobble

---

### Integral (Ki)

* Accumulates past error
* Eliminates steady-state drift

Effects:

* Fixes small persistent misalignment

- Too high → instability, drifting, slow oscillation

NOTE: Often not needed for simple navigation

---

### Derivative (Kd)

* Reacts to rate of error change
* Adds damping (stability)

Effects:

* Reduces oscillation
* Smooths motion

- Too high → sluggish / unresponsive

---

## 3. FAILURE → TUNING RULES (PRIMARY CONTROL LOGIC)

### DynamicInstability (Wobble / Oscillation)

Cause:

* Over-aggressive turning

Fix:

* Decrease Kp
* Increase Kd

---

### Thrashing (Violent Jerk / Instability)

Cause:

* Extremely unstable control loop

Fix:

* Significantly decrease Kp
* Increase Kd
* Optionally reduce base_speed

---

### SevereDrift (Moving away from target)

Cause:

* Not turning aggressively enough OR moving too fast

Fix:

* Increase Kp
* OR reduce base_speed

---

### KineticStagnation (Not moving)

Cause:

* Insufficient forward force

Fix:

* Increase base_speed
* (PID usually not the issue)

---

### Slow Response (Takes too long to turn)

Fix:

* Increase Kp slightly

---

### Overshoot (Passes target direction repeatedly)

Fix:

* Decrease Kp
* Increase Kd

---

### Steady-State Error (Never perfectly aligns)

Fix:

* Increase Ki slightly (use minimally)

---

## 4. TUNING STRATEGY (ORDER MATTERS)

1. Start with:

   * Ki = 0
   * Kd = 0

2. Tune Kp:

   * Increase until system responds quickly
   * Stop before strong oscillation begins

3. Add Kd:

   * Increase until oscillations are damped
   * Aim for smooth convergence

4. Add Ki (optional):

   * Only if persistent small error remains

---

## 5. SPEED INTERACTION (CRITICAL)

base_speed directly affects control stability:

* High base_speed:

  * Harder to turn
  * Requires higher Kp or lower speed

* Low base_speed:

  * Easier control
  * May cause stagnation

RULE:
If unstable → reduce base_speed before extreme PID changes

---

## 6. PRACTICAL LIMITS

* Kp too high → oscillation
* Kd too high → sluggish response
* Ki too high → drift + instability

Always clamp motor outputs to hardware limits.

---

## 7. QUICK DIAGNOSTIC TABLE

| Symptom         | Likely Cause       | Fix             |
| --------------- | ------------------ | --------------- |
| Wobbling        | Kp too high        | ↓ Kp, ↑ Kd      |
| Violent shaking | Unstable loop      | ↓↓ Kp, ↑ Kd     |
| Moves away      | Weak turning       | ↑ Kp or ↓ speed |
| Barely moves    | Low base speed     | ↑ base_speed    |
| Overshooting    | Too aggressive     | ↓ Kp, ↑ Kd      |
| Slow turning    | Low responsiveness | ↑ Kp            |
| Never aligns    | Bias error         | ↑ Ki (slightly) |

---

## 8. KEY PRINCIPLE

Kp = speed
Kd = stability
Ki = correction (use sparingly)

Goal:
Fast + stable convergence toward target heading

---

# Differential Drive + PID Integration Cheat Sheet

## Error Definition

```math
error = target\_heading - current\_heading
```

* Units: radians
* Normalize error to range: **[-π, π]**

---

## Differential Drive Kinematics

```math
\omega = \frac{v_{right} - v_{left}}{L}
```

* ( \omega ): angular velocity (rad/s)
* ( v_{right}, v_{left} ): wheel velocities (m/s)
* ( L ): distance between wheels (meters)

### Turn Rate Approximation

```math
turn\_rate \approx \frac{v_{left} - v_{right}}{2}
```

* `turn_rate` is the PID controller output
* Controls how sharply the robot rotates

---

## Units Convention

* All velocities are in **meters/second (m/s)**
* All angles are in **radians**

---

## Control Loop

1. Compute error
2. Apply PID controller
3. Compute `turn_rate`
4. Map to wheel velocities
5. Clamp outputs to motor limits

---

## Example

**Target heading:** 90°
**Current heading:** 60°

```math
error = +30°
```

### Interpretation

* Positive error → robot should turn **right**
* PID outputs **positive `turn_rate`**
* Left wheel speed > Right wheel speed
* Robot rotates toward target heading
