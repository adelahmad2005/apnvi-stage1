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

from collections import deque
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

# ---- Things coming in from the side (decided 2026-09-27) ----
SIDE_ZONE_M = 1.5                       # watch up to 1.5 m left and right of the centre
APPROACH_MAX_DISTANCE_M = 3.0           # only things closer than 3 m ahead
APPROACH_MIN_SPEED = 0.3                # closing in faster than 0.3 m/s (walls: 0 m/s)
APPROACH_MAX_SPEED = 2.5                # faster than this = a different object, not motion
APPROACH_WARN_TIME_S = 1.5              # warn if it would enter the path within 1.5 s
APPROACH_WINDOW_S = 0.5                 # measure the speed over the last 0.5 s
MIN_SIDE_PIXELS = 150                   # fewer side pixels than this = nothing on that side

# ---- Telling "it moves" from "we move" (decided 2026-09-27) ----
SCAN_STEP = 2                           # use every 2nd row and column (fast enough)
SCAN_MIN_PIXELS = 6                     # a column needs 6 (of every 2nd row) near pixels
SCAN_WIDEN_M = 0.15                     # "was it there before?" looks 15 cm higher/lower
SCAN_MAX_M = 6.0                        # further than this is too rough to line up
ICP_ROUNDS = 12                         # rounds of lining up the outlines
ICP_MIN_POINTS = 20                     # need at least this many outline points
ICP_SEARCH = 8                          # look 8 columns either side for a partner
ICP_GOOD_FIT = 0.7                      # 70% of points fit = no need to try more guesses
ICP_FIT_M = 0.02                        # a point within 2 cm of the old outline fits
ICP_MAX_TURN_DEG = 25                   # more turn than this between two pictures = error
ICP_MAX_SHIFT_M = 0.3                   # more shift than this between two pictures = error
COMPARE_S = 0.5                         # compare with the picture up to 0.5 s ago
COMPARE_MIN_S = 0.15                    # but at least 0.15 s ago
MOVED_MARGIN_M = 0.15                   # must stand this much nearer than before to count
NEIGHBOUR_COLUMNS = 3                   # compare with the old outline 3 columns either side
MIN_MOVING_COLUMNS = 5                  # fewer moved columns than this = nothing moved
APPROACH_CONFIRM_PICTURES = 3           # must say "closing in" 3 pictures in a row
HEADING_SMOOTHING_S = 1.0               # the path follows the camera's direction over ~1 s

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


def _positions(depth, intrinsics, camera_height_m):
    """For every pixel: metres ahead, metres sideways (+ = right), height above the floor."""
    intr = intrinsics_for(depth.shape, intrinsics)
    across, down = _ray_slopes(depth.shape, intr)
    ahead = depth.astype(np.float32) / 1000.0
    return ahead, across * ahead, camera_height_m - down * ahead


def path_mask(depth_mm, intrinsics=None, camera_height_m=CAMERA_HEIGHT_M):
    """
    Mark the pixels that are inside the real-size path in front of the user.

    Returns a grid of True/False, the same size as the picture.
    """
    depth = np.asarray(depth_mm)
    _, sideways, height = _positions(depth, intrinsics, camera_height_m)
    return ((depth > 0)
            & (np.abs(sideways) <= PATH_WIDTH_M / 2)
            & (height >= PATH_MIN_HEIGHT_M)
            & (height <= PATH_MAX_HEIGHT_M))


def side_gaps(depth_mm, intrinsics=None, camera_height_m=CAMERA_HEIGHT_M):
    """
    For the left and the right: how far the nearest thing is from the edge of the path.

    Returns {'left': (gap_m, ahead_m) or None, 'right': ...}. gap_m = metres between the
    thing and the path's edge; ahead_m = how far ahead it is.
    """
    depth = np.asarray(depth_mm)
    ahead, sideways, height = _positions(depth, intrinsics, camera_height_m)
    watched = ((depth > 0)
               & (ahead <= APPROACH_MAX_DISTANCE_M)
               & (height >= PATH_MIN_HEIGHT_M)
               & (height <= PATH_MAX_HEIGHT_M)
               & (np.abs(sideways) > PATH_WIDTH_M / 2)
               & (np.abs(sideways) <= SIDE_ZONE_M))
    result = {}
    for side, on_side in (('left', sideways < 0), ('right', sideways > 0)):
        mask = watched & on_side
        if np.count_nonzero(mask) < MIN_SIDE_PIXELS:
            result[side] = None
            continue
        gaps = np.abs(sideways[mask]) - PATH_WIDTH_M / 2
        gap = float(np.percentile(gaps, NEAREST_PERCENTILE))
        nearest = gaps <= gap + 0.05
        result[side] = (gap, round(float(np.median(ahead[mask][nearest])), 3))
    return result


class ApproachWatcher:
    """
    Spot things that are moving by themselves and closing in on the path from the side.

    The camera moves with the user, so on its own a wall seems to slide toward the path
    whenever the user turns. To tell the two apart (the same idea as the published
    "dynamic obstacle detection" work, e.g. Zhefan-Xu/onboard_detector, MIT licence):
      1. Work out how the camera itself moved since the last picture, by lining up the
         top-down outline of the room with the one before (scan matching / ICP).
      2. Undo that movement and look back about half a second: only something that now
         stands where there was empty space before has moved by itself. Walls and
         furniture never do, however the camera moves.
      3. Follow the edge of such a thing that is nearest the path, in fixed room
         coordinates, and warn if it will enter the path within APPROACH_WARN_TIME_S.
    The path points the way the user is heading (the camera's direction averaged over
    about a second), so sweeping the cane left and right does not swing the path.
    """

    def __init__(self):
        self.pose = (np.eye(2), np.zeros(2))   # camera -> room: rotation, position
        self.last_motion = None                # last picture-to-picture camera movement
        self.scans = deque()                   # (time, pose, scan) for the last COMPARE_S
        self.heading = None                    # the way the user is heading, in the room
        self.history = {'left': deque(), 'right': deque()}
        self.streak = {'left': 0, 'right': 0}   # pictures in a row that said "closing in"

    def reset(self):
        """Forget everything: used when the camera's own movement cannot be worked out."""
        self.__init__()

    def update(self, depth_mm, now_s, intrinsics=None, camera_height_m=CAMERA_HEIGHT_M):
        """
        Add one picture taken at time now_s (seconds).

        Returns (ahead_m, side) for the nearest thing closing in, or (None, None).
        """
        depth = np.asarray(depth_mm)
        new_scan = top_down_scan(depth, intrinsics, camera_height_m)
        if self.scans and not 0 < now_s - self.scans[-1][0] <= COMPARE_S:
            self.reset()                       # time went backwards or a long pause
        if self.scans:
            motion = camera_motion(self.scans[-1][2], new_scan, self.last_motion)
            if motion is None:                 # lost track of our own movement
                self.reset()
            else:
                self.last_motion = motion
                rot, pos = self.pose
                self.pose = (rot @ motion[0], rot @ motion[1] + pos)
        self.scans.append((now_s, self.pose, new_scan))
        while now_s - self.scans[0][0] > COMPARE_S:
            self.scans.popleft()

        forward = self.pose[0][:, 1]           # the way the camera points, in the room
        if self.heading is None:
            self.heading = forward
        else:
            mix = 1.0 - np.exp(-(now_s - self.scans[-2][0]) / HEADING_SMOOTHING_S)
            self.heading = (1 - mix) * self.heading + mix * forward
            self.heading = self.heading / np.hypot(*self.heading)
        path_pose = (np.array([[self.heading[1], self.heading[0]],
                               [-self.heading[0], self.heading[1]]]), self.pose[1])

        oldest_s, old_pose, old_scan = self.scans[0]
        moved = None
        if now_s - oldest_s >= COMPARE_MIN_S:
            moved = moved_in(new_scan, self.pose, old_scan, old_pose)
            spread = np.convolve(moved, np.ones(2 * NEIGHBOUR_COLUMNS + 1), 'same') > 0
            new_scan['moved'] = spread     # so the next line-up ignores them

        found = (None, None)
        for side in ('left', 'right'):
            history = self.history[side]
            lead = None
            if moved is not None:
                lead = leading_edge(new_scan, moved, side, self.pose, path_pose)
            if lead is not None:
                history.append((now_s, lead))                   # in room coordinates
            while history and now_s - history[0][0] > APPROACH_WINDOW_S:
                history.popleft()
            if lead is None or len(history) < 3:
                self.streak[side] = 0
                continue
            gaps = [(t, *gap_and_ahead(path_pose, point, side)) for t, point in history]
            self.streak[side] = self.streak[side] + 1 if closing_in(gaps) else 0
            confirmed = self.streak[side] >= APPROACH_CONFIRM_PICTURES
            if confirmed and (found[0] is None or gaps[-1][2] < found[0]):
                found = (round(float(gaps[-1][2]), 3), side)
        return found


def top_down_scan(depth, intrinsics=None, camera_height_m=CAMERA_HEIGHT_M):
    """
    Squash the picture into a top-down outline, like a laser scanner would see.

    For every SCAN_STEP-th column: the nearest thing between the path's low and high
    heights (at least SCAN_MIN_PIXELS pixels of it, so specks do not count). Returns
    {'ahead': metres (inf = nothing there, nan = no reading), 'points': (sideways,
    ahead) of each column, 'ahead_wide': the same with the heights widened by
    SCAN_WIDEN_M (used as "was anything there before?", so something sitting right at
    the height limit does not flicker in and out), plus the lens numbers}.
    """
    intr = intrinsics_for(depth.shape, intrinsics)
    across, down = _ray_slopes(depth.shape, intr)
    across, down = across[0, ::SCAN_STEP], down[::SCAN_STEP]
    ahead = depth[::SCAN_STEP, ::SCAN_STEP].astype(np.float32) / 1000.0
    height = camera_height_m - down * ahead
    has_reading = ahead > 0
    seen = has_reading.mean(axis=0) >= 0.2

    def nearest_in(low, high):
        """Return each column's nearest distance between heights low and high."""
        ok = has_reading & (height >= low) & (height <= high)
        near = np.partition(np.where(ok, ahead, np.inf), SCAN_MIN_PIXELS - 1, axis=0)
        near = near[SCAN_MIN_PIXELS - 1]
        return np.where(seen | np.isfinite(near), near, np.nan)

    nearest = nearest_in(PATH_MIN_HEIGHT_M, PATH_MAX_HEIGHT_M)
    wide = nearest_in(PATH_MIN_HEIGHT_M - SCAN_WIDEN_M, PATH_MAX_HEIGHT_M + SCAN_WIDEN_M)
    points = np.stack([across * nearest, nearest], axis=1)
    return {'ahead': nearest, 'ahead_wide': wide, 'points': points,
            'fx': intr[0] / SCAN_STEP, 'cx': intr[2] / SCAN_STEP}


def _rotation(angle):
    """Return the 2x2 grid that turns a top-down point by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _outline(scan):
    """
    Return, for every column, the usable outline point and the direction it faces.

    Unusable columns (too far, no reading, a moving thing, an edge) are nan.
    """
    pts = scan['points']
    if 'moved' in scan:                                     # leave out things that move
        pts = np.where(scan['moved'][:, None], np.nan, pts)
    before, after = np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0)
    along = after - before                                  # along the surface
    length = np.hypot(along[:, 0], along[:, 1])
    with np.errstate(invalid='ignore'):
        keep = (np.isfinite(pts[:, 1]) & (pts[:, 1] <= SCAN_MAX_M)
                & np.isfinite(length) & (length > 0) & (length <= 0.2))
    keep[[0, -1]] = False
    with np.errstate(invalid='ignore', divide='ignore'):
        along = along / length[:, np.newaxis]
    facing = np.stack([-along[:, 1], along[:, 0]], axis=1)  # at right angles to it
    pts = np.where(keep[:, None], pts, np.nan)
    return pts, np.where(keep[:, None], facing, np.nan)


def camera_motion(old_scan, new_scan, guess=None):
    """
    Work out how the camera moved between two scans (2D ICP, point-to-line, robust).

    Tries a few starting guesses (standing still, same as last time, a step forward) and
    keeps the answer that lines up the most points, so a person walking along with the
    user cannot pull the answer their way. Returns (rotation, shift) that take a point
    seen now into the old camera's view: old = rotation @ new + shift. Returns None if
    there is too little to go on.
    """
    src, _ = _outline(new_scan)
    src = src[np.isfinite(src[:, 1])][::2]
    dst, facing = _outline(old_scan)
    if len(src) < ICP_MIN_POINTS or np.count_nonzero(np.isfinite(dst[:, 1])) < ICP_MIN_POINTS:
        return None
    lens = (old_scan['fx'], old_scan['cx'])
    starts = [(0.0, np.zeros(2))] + [(0.0, np.array([0.0, step])) for step in (0.03, 0.06)]
    if guess is not None:
        starts.insert(0, (np.arctan2(guess[0][1, 0], guess[0][0, 0]), guess[1].copy()))
    best, best_score = None, -1
    for angle, shift in starts:
        result = _line_up(src, dst, facing, lens, angle, shift)
        if result is not None and result[2] > best_score:
            best, best_score = result, result[2]
        if best_score >= ICP_GOOD_FIT * len(src):
            break                              # good enough: skip the other guesses
    if best is None:
        return None
    angle, shift, _ = best
    if abs(np.degrees(angle)) > ICP_MAX_TURN_DEG or np.hypot(*shift) > ICP_MAX_SHIFT_M:
        return None                            # an impossible jump: do not trust it
    return _rotation(angle), shift


def _pair_up(moved, dst, lens):
    """
    For each point, find the nearest old outline point in the same direction (+-ICP_SEARCH).

    Looking only in the same direction (the column it lands in) is much faster than
    comparing every point with every other. Returns (index, distance); distance inf
    where there is none.
    """
    fx, cx = lens
    with np.errstate(invalid='ignore', divide='ignore'):
        column = np.round(fx * moved[:, 0] / moved[:, 1] + cx)
    column = np.where(np.isfinite(column) & (moved[:, 1] > 0), column, -10 * ICP_SEARCH)
    candidates = column[:, None].astype(int) + np.arange(-ICP_SEARCH, ICP_SEARCH + 1)
    inside = (candidates >= 0) & (candidates < len(dst))
    candidates = np.clip(candidates, 0, len(dst) - 1)
    dist2 = ((moved[:, None, :] - dst[candidates]) ** 2).sum(axis=2)
    dist2 = np.where(inside & np.isfinite(dist2), dist2, np.inf)
    best = dist2.argmin(axis=1)
    rows = np.arange(len(moved))
    return candidates[rows, best], np.sqrt(dist2[rows, best])


def _line_up(src, dst, facing, lens, angle, shift):
    """Run ICP from one starting guess. Returns (angle, shift, points that fit) or None."""
    for _ in range(ICP_ROUNDS):
        moved = src @ _rotation(angle).T + shift
        pair, dist = _pair_up(moved, dst, lens)
        keep = dist <= 0.5
        if keep.sum() < ICP_MIN_POINTS:
            return None
        p, n = moved[keep], facing[pair[keep]]
        miss = ((p - dst[pair[keep]]) * n).sum(axis=1)       # distance off the line
        turn = n[:, 0] * -p[:, 1] + n[:, 1] * p[:, 0]        # effect of a small turn
        rows = np.stack([turn, n[:, 0], n[:, 1]], axis=1)
        # Far points count more (a far wall is as big as a near person but gets fewer
        # columns), and points that do not fit well count less (they may be moving).
        weight = p[:, 1] / (1.0 + (miss / ICP_FIT_M) ** 2)
        step = np.linalg.solve((rows * weight[:, None]).T @ rows + 1e-3 * np.eye(3),
                               -(rows * weight[:, None]).T @ miss)
        angle += step[0]
        shift = _rotation(step[0]) @ shift + step[1:]
        if np.abs(step).max() < 1e-4:
            break
    moved = src @ _rotation(angle).T + shift
    pair, dist = _pair_up(moved, dst, lens)
    near = dist <= 0.1
    miss = np.abs(((moved[near] - dst[pair[near]]) * facing[pair[near]]).sum(axis=1))
    return angle, shift, int(np.count_nonzero(miss <= ICP_FIT_M))


def moved_in(new_scan, new_pose, old_scan, old_pose):
    """
    Mark the columns where something now stands in space that was empty before.

    The current points are moved into the old camera's view (undoing the camera's own
    movement) and compared with what the old camera saw in that direction.
    """
    rot = old_pose[0].T @ new_pose[0]
    shift = old_pose[0].T @ (new_pose[1] - old_pose[1])
    pts = new_scan['points'] @ rot.T + shift                  # nan rows stay nan
    old = old_scan['ahead_wide']
    # For every old column: the nearest thing within NEIGHBOUR_COLUMNS either side
    # (nan if any of them had no reading).
    padded = np.pad(old, NEIGHBOUR_COLUMNS, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * NEIGHBOUR_COLUMNS + 1)
    nearest_before = windows.min(axis=1)                      # nan wins, on purpose
    with np.errstate(invalid='ignore', divide='ignore'):
        column = np.round(old_scan['fx'] * pts[:, 0] / pts[:, 1] + old_scan['cx'])
        inside = np.isfinite(column) & (pts[:, 1] > 0) & (column >= 0) & (column < len(old))
    before = np.full(len(pts), np.nan)
    before[inside] = nearest_before[column[inside].astype(int)]
    with np.errstate(invalid='ignore'):
        return pts[:, 1] < before - MOVED_MARGIN_M           # nan compares False


def leading_edge(scan, moved, side, camera_pose, path_pose):
    """Return the room position of the moved thing's edge nearest the path, or None."""
    room = scan['points'] @ camera_pose[0].T + camera_pose[1]
    sideways, ahead = ((room - path_pose[1]) @ path_pose[0]).T  # as seen along the path
    on_side = sideways < 0 if side == 'left' else sideways > 0
    with np.errstate(invalid='ignore'):
        mask = (moved & on_side & (ahead <= APPROACH_MAX_DISTANCE_M)
                & (np.abs(sideways) > PATH_WIDTH_M / 2) & (np.abs(sideways) <= SIDE_ZONE_M))
    if _longest_run(mask) < MIN_MOVING_COLUMNS:
        return None                            # only scattered specks moved
    gaps = np.abs(sideways[mask]) - PATH_WIDTH_M / 2
    nearest = gaps <= np.percentile(gaps, 10) + 0.05
    return np.array([np.median(room[mask, 0][nearest]), np.median(room[mask, 1][nearest])])


def _longest_run(mask):
    """Return the length of the longest stretch of True next to each other."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int((ends - starts).max()) if len(starts) else 0


def gap_and_ahead(pose, room_point, side):
    """For a point in room coordinates: (gap to the path's edge, metres ahead) as seen now."""
    sideways, ahead = pose[0].T @ (room_point - pose[1])
    gap = (-sideways if side == 'left' else sideways) - PATH_WIDTH_M / 2
    return float(gap), float(ahead)


def closing_in(history):
    """Return True if the gaps in history (time, gap, ahead) show something coming in."""
    if len(history) < 3 or history[-1][0] - history[0][0] < APPROACH_WINDOW_S / 2:
        return False
    (t_first, gap_first, _), (_, gap_mid, _) = history[0], history[len(history) // 2]
    t_now, gap_now, ahead_now = history[-1]
    if not (gap_now <= gap_mid + 0.03 and gap_mid <= gap_first + 0.03):
        return False                                  # not shrinking steadily
    speed = (gap_first - gap_now) / (t_now - t_first)  # metres per second, + = closing in
    return (APPROACH_MIN_SPEED <= speed <= APPROACH_MAX_SPEED
            and gap_now / speed <= APPROACH_WARN_TIME_S
            and ahead_now <= APPROACH_MAX_DISTANCE_M)


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
