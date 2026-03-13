from dataclasses import dataclass
import math


@dataclass
class IDMParams:
    v0: float = 13.9
    T: float = 1.5
    s0: float = 2.0
    a: float = 1.5
    b: float = 2.0
    delta: int = 4


class IDM:
    """A minimal Intelligent Driver Model implementation."""

    def __init__(self, params: IDMParams):
        self.p = params

    def accel(self, v: float, gap: float, dv: float) -> float:
        """
        Compute longitudinal acceleration.
        :param v: ego speed [m/s]
        :param gap: bumper-to-bumper gap [m]
        :param dv: ego speed minus lead speed [m/s]
        """
        gap = max(gap, 0.1)
        s_star = self.p.s0 + v * self.p.T + (v * dv) / (2.0 * math.sqrt(max(self.p.a * self.p.b, 1e-3)))
        return self.p.a * (1.0 - (v / max(self.p.v0, 1e-3)) ** self.p.delta - (s_star / gap) ** 2)

    def free_accel(self, v: float) -> float:
        return self.p.a * (1.0 - (v / max(self.p.v0, 1e-3)) ** self.p.delta)

