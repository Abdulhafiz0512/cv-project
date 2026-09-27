"""Official event ids and the COCO classes the detector keeps."""
from __future__ import annotations

CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

# COCO ids -> coarse road-user kind
COCO_KIND: dict[int, str] = {
    0: "person",
    1: "two_wheeler",   # bicycle
    2: "vehicle",       # car
    3: "two_wheeler",   # motorcycle
    5: "vehicle",       # bus
    7: "vehicle",       # truck
    15: "animal", 16: "animal", 17: "animal", 18: "animal", 19: "animal",  # cat dog horse sheep cow
}
DETECT_CLASSES: list[int] = sorted(COCO_KIND)
