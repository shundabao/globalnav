"""Geographic helpers for route geometry."""

from __future__ import annotations

import math


def great_circle_points(
    lat1: float, lon1: float, lat2: float, lon2: float, n: int = 24,
) -> list[tuple[float, float]]:
    """Sample lat/lon points along a great-circle arc."""
    phi1, lam1 = math.radians(lat1), math.radians(lon1)
    phi2, lam2 = math.radians(lat2), math.radians(lon2)
    d = 2 * math.asin(math.sqrt(
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin((lam2 - lam1) / 2) ** 2
    ))
    if d < 1e-9:
        return [(lat1, lon1), (lat2, lon2)]

    points: list[tuple[float, float]] = []
    for i in range(n + 1):
        f = i / n
        a = math.sin((1 - f) * d) / math.sin(d)
        b = math.sin(f * d) / math.sin(d)
        x = a * math.cos(phi1) * math.cos(lam1) + b * math.cos(phi2) * math.cos(lam2)
        y = a * math.cos(phi1) * math.sin(lam1) + b * math.cos(phi2) * math.sin(lam2)
        z = a * math.sin(phi1) + b * math.sin(phi2)
        points.append((
            math.degrees(math.atan2(z, math.sqrt(x * x + y * y))),
            math.degrees(math.atan2(y, x)),
        ))
    return points


def line_interpolate(
    lat1: float, lon1: float, lat2: float, lon2: float, n: int = 12,
) -> list[tuple[float, float]]:
    """Simple linear interpolation (good enough for short transit estimates)."""
    if n < 1:
        return [(lat1, lon1), (lat2, lon2)]
    return [
        (lat1 + (lat2 - lat1) * i / n, lon1 + (lon2 - lon1) * i / n)
        for i in range(n + 1)
    ]
