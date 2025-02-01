from dataclasses import dataclass

@dataclass
class Direction:
    yaw: float
    pitch: float
    yaw_threshold: int = 20
    pitch_threshold: int = 20
    value: list = None  # Will be calculated upon initialization

    def __post_init__(self):
        self.value = [0, 0]  # Default to straight values
        self.calculate_direction()

    def calculate_direction(self) -> None:
        # Calculate x based on yaw
        if self.yaw > self.yaw_threshold:
            self.value[0] = 1  # Right
        elif self.yaw < -self.yaw_threshold:
            self.value[0] = -1  # Left

        # Calculate y based on pitch
        if self.pitch > self.pitch_threshold:
            self.value[1] = -1  # Up
        elif self.pitch < -self.pitch_threshold:
            self.value[1] = 1  # Down

    @property
    def x(self) -> int:
        return self.value[0]

    @x.setter
    def x(self, new_x: int) -> None:
        self.value[0] = new_x

    @property
    def y(self) -> int:
        return self.value[1]

    @y.setter
    def y(self, new_y: int) -> None:
        self.value[1] = new_y

    def __str__(self) -> str:
        horizontal = { -1: "Left", 0: "Straight", 1: "Right" }
        vertical = { -1: "Down", 0: "Straight", 1: "Up" }

        h_dir = horizontal.get(self.value[0], "Invalid")
        v_dir = vertical.get(self.value[1], "Invalid")

        if h_dir == "Straight" and v_dir == "Straight":
            return "Straight"
        elif h_dir == "Straight":
            return f"Straight {v_dir.lower()}"
        elif v_dir == "Straight":
            return f"{h_dir} straight"
        else:
            return f"{h_dir} {v_dir.lower()}"

    def print_direction(self) -> None:
        print(str(self))
    
    def print_yaw_pitch(self) -> None:
        print(f"Yaw: {self.yaw:.2f} degrees, Pitch: {self.pitch:.2f} degrees")

