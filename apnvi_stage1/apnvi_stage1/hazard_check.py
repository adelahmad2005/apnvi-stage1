"""
The check: one depth picture in, nearest distance + blocked + level out.

Plain Python and numpy only. Nothing from ROS, so a short script can still
use this if ROS fails. hazard_node.py calls these functions.

Depth picture = 2D grid of whole numbers, one per pixel, in millimetres.
0 means "no reading".
"""

import numpy as np

# ---- Agreed settings (stage1-weekend-plan.md, decided 2026-09-25) ----
LANE_LEFT, LANE_RIGHT = 1 / 3, 2 / 3    # walking lane: middle third, left to right
LANE_TOP, LANE_BOTTOM = 1 / 4, 3 / 4    # walking lane: middle half, top to bottom
BLOCKED_ZERO_FRACTION = 0.80            # more than 80% of the lane is 0 -> camera blocked
NEAREST_PERCENTILE = 5                  # nearest distance = 5th percentile of valid pixels

STOP_M = 0.5                            # 0.5 m or less  -> stop
WARNING_M = 1.5                         # 1.5 m or less  -> warning
NOTICE_M = 2.0                          # 2.0 m or less  -> notice, above -> clear
DEAD_ZONE_M = 0.1                       # only go to a safer level 0.1 m past the boundary

# Levels from safest to most dangerous. The position in this list is the "danger rank".
LEVELS = ['clear', 'notice', 'warning', 'stop']


def walking_lane(depth):
    """Cut the walking lane (middle third across, middle half down) out of the picture."""
    height, width = depth.shape
    top, bottom = int(height * LANE_TOP), int(height * LANE_BOTTOM)
    left, right = int(width * LANE_LEFT), int(width * LANE_RIGHT)
    return depth[top:bottom, left:right]


def level_for_distance(distance_m):
    """Turn a distance in metres into a level. A boundary belongs to the more dangerous level."""
    if distance_m <= STOP_M:
        return 'stop'
    if distance_m <= WARNING_M:
        return 'warning'
    if distance_m <= NOTICE_M:
        return 'notice'
    return 'clear'


def level_with_dead_zone(distance_m, previous_level):
    """
    Turn a distance into a level, but without flickering at a boundary.

    Getting more dangerous (or staying the same) happens at once.
    Getting safer only happens once the distance is DEAD_ZONE_M past the boundary.
    """
    new_level = level_for_distance(distance_m)
    if previous_level is None or LEVELS.index(new_level) >= LEVELS.index(previous_level):
        return new_level
    # Getting safer: judge as if the object were DEAD_ZONE_M closer than measured.
    return level_for_distance(distance_m - DEAD_ZONE_M)


def check_depth(depth_mm):
    """
    Run the check on one depth picture.

    Returns three values: distance_m, blocked, level
      - camera blocked:  None, True, None   (hazard_node then uses the ultrasonic)
      - otherwise:       e.g. 1.234, False, 'warning'
    """
    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError(f'depth picture must be a 2D grid, got {depth.ndim} dimensions')

    lane = walking_lane(depth)
    if lane.size == 0:
        raise ValueError(f'depth picture {depth.shape} is too small to have a walking lane')

    zero_fraction = np.count_nonzero(lane == 0) / lane.size
    if zero_fraction > BLOCKED_ZERO_FRACTION:
        return None, True, None

    valid = lane[lane > 0]                                  # throw away the "no reading" pixels
    nearest_mm = np.percentile(valid, NEAREST_PERCENTILE)   # 5% of the lane is closer than this
    distance_m = round(float(nearest_mm) / 1000.0, 3)       # millimetres -> metres
    return distance_m, False, level_for_distance(distance_m)


def nearest_distance(camera_m, ultrasonic_m):
    """
    Pick the nearer of the two sensors' distances (metres).

    None or a negative number means "no reading" from that sensor and is ignored.
    Returns -1.0 when neither sensor has a reading ("nothing in range").
    """
    readings = [d for d in (camera_m, ultrasonic_m) if d is not None and d >= 0]
    return min(readings) if readings else -1.0
