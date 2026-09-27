"""
The check: one depth picture in, nearest distance + blocked + level out.

Plain Python and numpy only. Nothing from ROS, so a short script can still
use this if ROS fails. hazard_node.py calls these functions.

Depth picture = 2D grid of whole numbers, one per pixel, in millimetres.
0 means "no reading".

What counts as "in the way" (changed 2026-09-27, Adel): a real-size path in front of
the user, PATH_WIDTH_M wide, from PATH_MIN_HEIGHT_M above the floor up to
PATH_MAX_HEIGHT_M. Every pixel is turned into a real position (metres left/right and
height above the floor) using the camera's lens numbers (intrinsics) and the camera's
height. So something close and a bit to the side still counts, and a wall far to the
side does not. This replaces the old fixed box on the picture, which was only about
25 cm wide close up. Assumes the camera is mounted level (not tilted up or down).
"""

from functools import lru_cache

import numpy as np

# ---- Agreed settings (stage1-weekend-plan.md, decided 2026-09-25) ----
LANE_LEFT, LANE_RIGHT = 1 / 3, 2 / 3    # box used ONLY for the "camera blocked" check
LANE_TOP, LANE_BOTTOM = 1 / 4, 3 / 4
BLOCKED_ZERO_FRACTION = 0.80            # more than 80% of that box is 0 -> camera blocked
NEAREST_PERCENTILE = 5                  # nearest distance = 5th percentile of path pixels

STOP_M = 0.5                            # 0.5 m or less  -> stop
WARNING_M = 1.5                         # 1.5 m or less  -> warning
NOTICE_M = 2.0                          # 2.0 m or less  -> notice, above -> clear
DEAD_ZONE_M = 0.1                       # only go to a safer level 0.1 m past the boundary

# ---- The real-size path (decided 2026-09-27) ----
CAMERA_HEIGHT_M = 1.0                   # camera lens height above the floor (assumed)
PATH_WIDTH_M = 0.70                     # shoulder width + margin, centred on the camera
PATH_MIN_HEIGHT_M = 0.10                # ignore the floor itself (and 10 cm of noise)
PATH_MAX_HEIGHT_M = 2.00                # a bit above head height
MIN_PATH_PIXELS = 50                    # fewer path pixels than this = nothing in the path

# Lens numbers (intrinsics) of our D435 at 848 x 480, read from Mo'ath's recording:
# fx, fy = focal length in pixels; cx, cy = the picture's optical centre in pixels.
# The live camera sends its own; these are only used when it does not.
DEFAULT_INTRINSICS = (427.278931, 427.278931, 424.819305, 238.817108)
DEFAULT_SIZE = (480, 848)               # (height, width) those numbers belong to

# Levels from safest to most dangerous. The position in this list is the "danger rank".
LEVELS = ['clear', 'notice', 'warning', 'stop']


def walking_lane(depth):
    """Cut the middle box (third across, half down) out of the picture: the blocked check."""
    height, width = depth.shape
    top, bottom = int(height * LANE_TOP), int(height * LANE_BOTTOM)
    left, right = int(width * LANE_LEFT), int(width * LANE_RIGHT)
    return depth[top:bottom, left:right]


def intrinsics_for(shape, intrinsics=None):
    """Return (fx, fy, cx, cy): the given ones, or a best guess for this picture size."""
    if intrinsics is not None:
        return tuple(float(v) for v in intrinsics)
    if tuple(shape) == DEFAULT_SIZE:
        return DEFAULT_INTRINSICS
    height, width = shape                               # rough guess for other sizes
    focal = DEFAULT_INTRINSICS[1] * height / DEFAULT_SIZE[0]
    return (focal, focal, width / 2.0, height / 2.0)


@lru_cache(maxsize=8)
def _ray_slopes(shape, intrinsics):
    """
    For every pixel: how far sideways and down it points, per metre of distance.

    Worked out once per picture size and reused (lru_cache remembers it).
    """
    fx, fy, cx, cy = intrinsics
    height, width = shape
    across = (np.arange(width, dtype=np.float32) - cx) / fx    # + = right of centre
    down = (np.arange(height, dtype=np.float32) - cy) / fy     # + = below centre
    return across[np.newaxis, :], down[:, np.newaxis]


def path_mask(depth_mm, intrinsics=None, camera_height_m=CAMERA_HEIGHT_M):
    """
    Mark the pixels that are inside the real-size path in front of the user.

    Returns a grid of True/False, the same size as the picture.
    """
    depth = np.asarray(depth_mm)
    intr = intrinsics_for(depth.shape, intrinsics)
    across, down = _ray_slopes(depth.shape, intr)
    z = depth.astype(np.float32) / 1000.0                     # metres straight ahead
    sideways = np.abs(across * z)                             # metres left/right of centre
    height = camera_height_m - down * z                       # metres above the floor
    return ((depth > 0)
            & (sideways <= PATH_WIDTH_M / 2)
            & (height >= PATH_MIN_HEIGHT_M)
            & (height <= PATH_MAX_HEIGHT_M))


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


def check_depth(depth_mm, intrinsics=None, camera_height_m=CAMERA_HEIGHT_M):
    """
    Run the check on one depth picture.

    Returns three values: distance_m, blocked, level
      - camera blocked:        None, True, None   (hazard_node then uses the ultrasonic)
      - nothing in the path:   None, False, 'clear'
      - otherwise:             e.g. 1.234, False, 'warning'
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

    in_path = depth[path_mask(depth, intrinsics, camera_height_m)]
    if in_path.size < MIN_PATH_PIXELS:
        return None, False, 'clear'                          # nothing in the path

    nearest_mm = np.percentile(in_path, NEAREST_PERCENTILE)  # 5% of the path is closer
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
