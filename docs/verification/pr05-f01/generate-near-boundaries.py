"""Exact Fraction oracle, then one IEEE-754 rounding and adjacent-float controls."""
from fractions import Fraction as F
import json
import math
from pathlib import Path
import sys

cases = [
    ([640, 480], [960, 540]), ([1280, 720], [800, 600]),
    ([1280, 720], [960, 540]), ([600, 600], [960, 540]),
    ([1280, 720], [600, 600]), ([640, 480], [F(3843, 4), F(2161, 4)]),
    ([480, 640], [F(2161, 4), F(3843, 4)]),
    ([853, 480], [F(3843, 4), F(2161, 4)]),
    ([1920, 1080], [F(3843, 4), F(2161, 4)]),
    ([3840, 2160], [F(3843, 4), F(2161, 4)]),
]
rows = []
for video, box in cases:
    w, h = map(F, box)
    scale = min(w / video[0], h / video[1])
    cw, ch = video[0] * scale, video[1] * scale
    for ox, oy in [(F(149, 4), F(165, 2)), (F(500), F(250)), (F(-20), F(-250))]:
        left, top = ox + (w - cw) / 2, oy + (h - ch) / 2
        right, bottom = left + cw, top + ch
        for dpr in [1, 1.25, 2, 3]:
            for axis, edge, other, direction, normal in [
                (0, left, top + ch / 2, -1, [0, 0.5]),
                (0, right, top + ch / 2, 1, [1, 0.5]),
                (1, top, left + cw / 2, -1, [0.5, 0]),
                (1, bottom, left + cw / 2, 1, [0.5, 1]),
            ]:
                rounded = float(edge)
                values = [
                    (rounded, "boundary", normal),
                    (math.nextafter(rounded, math.inf * direction), "next-float-outside", None),
                    (math.nextafter(rounded, -math.inf * direction), "next-float-inside", "inside"),
                    (float(edge + F(direction, 10**12)), "outside-1e-12", None),
                    (float(edge + F(direction, 1000)), "outside-0.001", None),
                ]
                for coordinate, kind, expected in values:
                    point = [coordinate, float(other)] if axis == 0 else [float(other), coordinate]
                    rows.append(dict(video=video, rect=dict(left=float(ox), top=float(oy), width=float(w), height=float(h)),
                                     point=point, dpr=dpr, kind=kind, expected=expected))
Path(sys.argv[1]).write_text(json.dumps(rows, separators=(",", ":")) + "\n")
print(len(rows), "rational/adjacent-float controls")
