import time
from dataclasses import dataclass

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


SERVO_PINS = {
    "base": 17,
    "shoulder": 27,
    "elbow": 22,
    "wrist": 23,
    "gripper": 24,
}


# Calibrate these angles on your actual manipulator before running with waste.
POSES = {
    "home": {
        "base": 90,
        "shoulder": 80,
        "elbow": 95,
        "wrist": 90,
        "gripper": 80,
    },
    "pick_open": {
        "base": 90,
        "shoulder": 120,
        "elbow": 65,
        "wrist": 95,
        "gripper": 85,
    },
    "pick_closed": {
        "base": 90,
        "shoulder": 120,
        "elbow": 65,
        "wrist": 95,
        "gripper": 35,
    },
    "lift": {
        "base": 90,
        "shoulder": 80,
        "elbow": 95,
        "wrist": 90,
        "gripper": 35,
    },
    "bio_drop": {
        "base": 45,
        "shoulder": 85,
        "elbow": 95,
        "wrist": 90,
        "gripper": 35,
    },
    "non_bio_drop": {
        "base": 135,
        "shoulder": 85,
        "elbow": 95,
        "wrist": 90,
        "gripper": 35,
    },
    "drop_open": {
        "gripper": 85,
    },
}


@dataclass
class ServoConfig:
    min_angle: int = 0
    max_angle: int = 180
    min_duty: float = 2.5
    max_duty: float = 12.5


class ServoArm:
    def __init__(
        self,
        pins: dict[str, int] | None = None,
        poses: dict[str, dict[str, int]] | None = None,
        dry_run: bool = False,
    ) -> None:
        self.pins = pins or SERVO_PINS
        self.poses = poses or POSES
        self.dry_run = dry_run
        self.config = ServoConfig()
        self.current = dict(self.poses["home"])
        self.pwm = {}

        if self.dry_run:
            print("ServoArm running in dry-run mode. No GPIO output will be used.")
            return

        if GPIO is None:
            raise RuntimeError("RPi.GPIO is not available. Run this on the Raspberry Pi.")

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in self.pins.values():
            GPIO.setup(pin, GPIO.OUT)
            self.pwm[pin] = GPIO.PWM(pin, 50)
            self.pwm[pin].start(0)

        self.move_to("home", duration=1.0)

    def angle_to_duty(self, angle: float) -> float:
        angle = max(self.config.min_angle, min(self.config.max_angle, angle))
        span = self.config.max_duty - self.config.min_duty
        return self.config.min_duty + (angle / 180.0) * span

    def write_servo(self, servo_name: str, angle: float) -> None:
        angle = max(self.config.min_angle, min(self.config.max_angle, angle))

        if self.dry_run:
            print(f"{servo_name}: {angle:.1f} deg")
            return

        pin = self.pins[servo_name]
        duty = self.angle_to_duty(angle)
        self.pwm[pin].ChangeDutyCycle(duty)

    def move_to(self, pose_name: str, duration: float = 1.0, steps: int = 30) -> None:
        target = dict(self.current)
        target.update(self.poses[pose_name])

        start = dict(self.current)
        delay = duration / max(steps, 1)

        for step in range(1, steps + 1):
            ratio = step / steps
            for servo_name, target_angle in target.items():
                start_angle = start[servo_name]
                angle = start_angle + (target_angle - start_angle) * ratio
                self.write_servo(servo_name, angle)
            time.sleep(delay)

        self.current = target

        if not self.dry_run:
            time.sleep(0.1)
            for pin in self.pwm:
                self.pwm[pin].ChangeDutyCycle(0)

    def pick_object(self) -> None:
        self.move_to("home", duration=0.8)
        self.move_to("pick_open", duration=1.2)
        time.sleep(0.3)
        self.move_to("pick_closed", duration=0.6)
        time.sleep(0.3)
        self.move_to("lift", duration=1.0)

    def drop_bio(self) -> None:
        self.move_to("bio_drop", duration=1.0)
        time.sleep(0.3)
        self.move_to("drop_open", duration=0.5)
        time.sleep(0.3)
        self.move_to("home", duration=1.0)

    def drop_non_bio(self) -> None:
        self.move_to("non_bio_drop", duration=1.0)
        time.sleep(0.3)
        self.move_to("drop_open", duration=0.5)
        time.sleep(0.3)
        self.move_to("home", duration=1.0)

    def sort(self, label: str) -> None:
        normalized = label.lower().replace("-", "_").replace(" ", "_")

        self.pick_object()

        if "non" in normalized:
            print("Dropping into non-biodegradable box.")
            self.drop_non_bio()
        elif "bio" in normalized:
            print("Dropping into biodegradable box.")
            self.drop_bio()
        else:
            raise ValueError(f"Unknown waste label: {label}")

    def cleanup(self) -> None:
        if self.dry_run or GPIO is None:
            return

        for pwm in self.pwm.values():
            pwm.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    arm = ServoArm(dry_run=True)
    arm.sort("biodegradable")
    arm.sort("non_biodegradable")
