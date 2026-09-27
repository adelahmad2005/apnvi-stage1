"""Tests for hazard_check.py using fake depth pictures made in Python (no camera, no ROS)."""

import math

from apnvi_stage1.hazard_check import (
    ApproachWatcher, CAMERA_HEIGHT_M, camera_motion, check_depth, DEFAULT_INTRINSICS,
    level_for_distance, level_with_dead_zone, nearest_distance, path_mask, side_gaps,
    top_down_scan, walking_lane)
import numpy as np
import pytest


def picture(distance_mm, width=848, height=480):
    """Make a fake depth picture where every pixel is the same distance (mm)."""
    return np.full((height, width), distance_mm, dtype=np.uint16)


def lane_box(depth):
    """Return (top, bottom, left, right) of the walking lane, to paint things into it."""
    height, width = depth.shape
    return int(height / 4), int(height * 3 / 4), int(width / 3), int(width * 2 / 3)


def cover_lane(depth, fraction, value_mm):
    """Set the first `fraction` of the lane's pixels to value_mm. Returns the picture."""
    top, bottom, left, right = lane_box(depth)
    lane = depth[top:bottom, left:right]            # a view: changing it changes depth
    flat = lane.reshape(-1)
    flat[:int(round(flat.size * fraction))] = value_mm
    depth[top:bottom, left:right] = flat.reshape(lane.shape)
    return depth


# ---- Plain distances ----

def test_everything_far_is_clear():
    assert check_depth(picture(3000)) == (3.0, False, 'clear')


def test_object_filling_lane_at_1m_is_warning():
    assert check_depth(picture(1000)) == (1.0, False, 'warning')


# ---- Boundaries belong to the more dangerous level (decision 1) ----

@pytest.mark.parametrize('mm, level', [
    (2000, 'notice'), (1500, 'warning'), (500, 'stop'),
    (2001, 'clear'), (1501, 'notice'), (501, 'warning'),
])
def test_boundaries(mm, level):
    assert check_depth(picture(mm))[2] == level


# ---- Camera blocked ----

def test_85_percent_zeros_is_blocked():
    depth = cover_lane(picture(1000), 0.85, 0)
    assert check_depth(depth) == (None, True, None)


def test_exactly_80_percent_zeros_is_not_blocked():
    depth = cover_lane(picture(1000), 0.80, 0)
    assert check_depth(depth) == (1.0, False, 'warning')


def test_all_zeros_is_blocked():
    assert check_depth(picture(0)) == (None, True, None)


# ---- Blocked check box / percentile ----

def test_tiny_speck_is_ignored():
    depth = picture(3000)
    depth[230:234, 420:425] = 300                   # 20 noisy pixels: far under 5% of the path
    assert check_depth(depth)[2] == 'clear'


def test_bigger_object_is_found():
    depth = cover_lane(picture(3000), 0.10, 400)    # 10% of the lane at 0.4 m
    assert check_depth(depth) == (0.4, False, 'stop')


def test_zeros_do_not_count_as_near():
    depth = cover_lane(picture(2500), 0.50, 0)      # half the lane has no reading
    assert check_depth(depth) == (2.5, False, 'clear')


# ---- Picture size does not matter ----

@pytest.mark.parametrize('width, height', [(848, 480), (640, 480), (1280, 720)])
def test_any_picture_size(width, height):
    depth = cover_lane(picture(3000, width, height), 0.10, 1200)
    assert check_depth(depth) == (1.2, False, 'warning')


def test_lane_is_middle_third_and_middle_half():
    assert walking_lane(picture(1000, 848, 480)).shape == (240, 283)


# ---- Bad input gives a clear error ----

def test_not_a_2d_grid_is_an_error():
    with pytest.raises(ValueError):
        check_depth(np.zeros((480, 848, 3), dtype=np.uint16))


def test_too_small_picture_is_an_error():
    with pytest.raises(ValueError):
        check_depth(picture(1000, width=2, height=1))


# ---- Dead zone (decision 2) ----

def test_level_for_distance_table():
    assert [level_for_distance(d) for d in (0.3, 0.5, 1.0, 1.5, 1.8, 2.0, 3.0)] == \
        ['stop', 'stop', 'warning', 'warning', 'notice', 'notice', 'clear']


def test_dead_zone_holds_the_level_near_a_boundary():
    assert level_with_dead_zone(2.05, 'notice') == 'notice'      # inside the dead zone
    assert level_with_dead_zone(2.11, 'notice') == 'clear'       # past it
    assert level_with_dead_zone(0.55, 'stop') == 'stop'
    assert level_with_dead_zone(0.61, 'stop') == 'warning'


def test_dead_zone_never_delays_danger():
    assert level_with_dead_zone(1.49, 'notice') == 'warning'     # more dangerous: at once
    assert level_with_dead_zone(0.2, 'clear') == 'stop'


def test_dead_zone_with_no_previous_level():
    assert level_with_dead_zone(2.05, None) == 'clear'


# ---- Picking the nearer sensor (used by hazard_node) ----

def test_nearer_sensor_wins():
    assert nearest_distance(2.0, 0.8) == 0.8
    assert nearest_distance(0.6, 1.9) == 0.6


def test_missing_or_minus_one_is_ignored():
    assert nearest_distance(None, 1.2) == 1.2          # camera blocked
    assert nearest_distance(1.5, -1.0) == 1.5          # ultrasonic: nothing in range
    assert nearest_distance(None, None) == -1.0        # nothing from either
    assert nearest_distance(None, -1.0) == -1.0


# ---- The real-size path (70 cm wide, 10 cm to 2 m above the floor, camera at 1 m) ----

FX, FY, CX, CY = DEFAULT_INTRINSICS


def put_object(depth, z_m, left_m, right_m, low_m, high_m):
    """Paint a flat object at distance z_m, between left_m..right_m and low_m..high_m."""
    height, width = depth.shape
    u0, u1 = (int(np.clip(CX + m * FX / z_m, 0, width)) for m in (left_m, right_m))
    v0, v1 = (int(np.clip(CY + (CAMERA_HEIGHT_M - m) * FY / z_m, 0, height))
              for m in (high_m, low_m))
    depth[v0:v1, u0:u1] = int(z_m * 1000)
    return depth


def add_floor(depth):
    """Make every pixel below the horizon show the floor, as a level camera at 1 m sees it."""
    for v in range(int(CY) + 1, depth.shape[0]):
        depth[v, :] = min(65000, int(1000 * CAMERA_HEIGHT_M * FY / (v - CY)))
    return depth


def test_close_object_to_the_side_is_found():
    # 0.6 m away, 22 to 34 cm left of centre: outside the OLD box, inside the 70 cm path
    depth = put_object(picture(4000), 0.6, -0.34, -0.22, 0.5, 1.2)
    assert check_depth(depth) == (0.6, False, 'warning')


def test_wall_far_to_the_side_is_ignored():
    depth = put_object(picture(4000), 1.5, -1.3, -0.9, 0.2, 1.8)   # 0.9 m+ to the left
    assert check_depth(depth)[2] == 'clear'


def test_floor_is_ignored():
    depth = add_floor(picture(4000))
    assert check_depth(depth) == (4.0, False, 'clear')


def test_low_obstacle_on_the_floor_is_found():
    # A level camera at 1 m cannot see the floor closer than about 1.8 m (its view only
    # reaches 29 degrees down), so a 25 cm box is tested at 1.9 m.
    depth = put_object(add_floor(picture(4000)), 1.9, -0.2, 0.2, 0.0, 0.25)
    assert check_depth(depth) == (1.9, False, 'notice')


def test_something_above_head_height_is_ignored():
    depth = put_object(picture(4000), 3.0, -0.3, 0.3, 2.1, 2.6)
    assert check_depth(depth)[2] == 'clear'


def test_nothing_in_the_path_is_clear():
    depth = add_floor(np.zeros((480, 848), dtype=np.uint16))       # only floor, no wall
    assert check_depth(depth) == (None, False, 'clear')


def test_path_is_wider_close_up_than_far_away():
    near = path_mask(picture(500))[240].sum()      # pixels across at 0.5 m
    far = path_mask(picture(3000))[240].sum()      # pixels across at 3 m
    assert near > 5 * far


# ---- Things coming in from the side ----

def run_watcher(frames, fps=30):
    """Feed pictures to a fresh ApproachWatcher; return every answer it gives."""
    watcher = ApproachWatcher()
    return [watcher.update(depth, i / fps) for i, depth in enumerate(frames)]


def side_object_at(inner_edge_m, ahead_m=1.2, side=-1):
    """Make a 30 cm wide object whose edge nearest the path is inner_edge_m from centre."""
    near, far = inner_edge_m, inner_edge_m + 0.3
    left, right = (-far, -near) if side < 0 else (near, far)
    return put_object(picture(4000), ahead_m, left, right, 0.3, 1.6)


def test_side_gap_is_measured():
    gaps = side_gaps(side_object_at(0.85))           # 50 cm from the path's edge (at 0.35)
    assert gaps['left'][0] == pytest.approx(0.50, abs=0.02)
    assert gaps['left'][1] == pytest.approx(1.2, abs=0.01)
    assert gaps['right'] is None


def test_still_wall_next_to_the_path_never_warns():
    answers = run_watcher([side_object_at(0.6)] * 30)
    assert all(a == (None, None) for a in answers)


def test_object_coming_in_from_the_left_warns_early():
    # moves 1 m/s toward the path, starting 1.2 m left of centre
    frames = [side_object_at(1.2 - i / 30) for i in range(20)]
    answers = run_watcher(frames)
    first = next(i for i, a in enumerate(answers) if a[0] is not None)
    assert answers[first] == (1.2, 'left')
    gap_when_warned = 1.2 - first / 30 - 0.35
    assert gap_when_warned > 0.3                     # warned while still well outside the path


def test_object_coming_in_from_the_right_warns():
    frames = [side_object_at(1.2 - i / 30, side=1) for i in range(20)]
    assert any(a == (1.2, 'right') for a in run_watcher(frames))


def test_object_moving_away_never_warns():
    frames = [side_object_at(0.5 + i / 30) for i in range(20)]
    assert all(a == (None, None) for a in run_watcher(frames))


def test_slow_object_does_not_warn_yet():
    frames = [side_object_at(1.2 - 0.1 * i / 30) for i in range(30)]   # 0.1 m/s
    assert all(a == (None, None) for a in run_watcher(frames))


def test_sudden_jump_is_not_motion():
    frames = [side_object_at(1.3)] * 10 + [side_object_at(0.5)] * 10  # a different object
    assert all(a == (None, None) for a in run_watcher(frames))


def test_far_away_object_is_ignored():
    frames = [side_object_at(1.2 - i / 30, ahead_m=3.5) for i in range(20)]
    assert all(a == (None, None) for a in run_watcher(frames))


# ---- The camera moves too: "it moves" versus "we move" ----
# A tiny top-down ray caster: a level camera 1 m up, walking and turning among upright
# boxes. Each box is (left, right, near, far, low, high) in room metres.

ROOM = [(-1.2, -1.0, 0.2, 6.0, 0.0, 2.5),      # wall 1 m to the left
        (1.3, 1.5, 0.2, 6.0, 0.0, 2.5),        # wall 1.3 m to the right
        (-0.6, 0.2, 4.0, 4.4, 0.0, 1.5),       # cabinet ahead
        (0.6, 1.0, 1.5, 1.9, 0.0, 0.9)]        # table front-right


def person(x, z):
    """Return a 40 x 30 cm box, 1.8 m tall, centred at (x, z)."""
    return (x - 0.2, x + 0.2, z - 0.15, z + 0.15, 0.0, 1.8)


def room_picture(boxes, cam_x=0.0, cam_z=0.0, yaw_deg=0.0):
    """Draw what the camera sees at (cam_x, cam_z), turned yaw_deg to the right."""
    yaw = math.radians(yaw_deg)
    ahead_dir = np.array([math.sin(yaw), math.cos(yaw)])
    right_dir = np.array([math.cos(yaw), -math.sin(yaw)])
    rays = ((np.arange(848) - CX) / FX)[:, None] * right_dir + ahead_dir
    down = ((np.arange(480) - CY) / FY)[:, None]
    depth = np.full((480, 848), np.inf)
    for x0, x1, z0, z1, low, high in boxes:
        with np.errstate(divide='ignore', invalid='ignore'):
            tx = np.sort([(x0 - cam_x) / rays[:, 0], (x1 - cam_x) / rays[:, 0]], axis=0)
            tz = np.sort([(z0 - cam_z) / rays[:, 1], (z1 - cam_z) / rays[:, 1]], axis=0)
        enter, leave = np.maximum(tx[0], tz[0]), np.minimum(tx[1], tz[1])
        dist = np.where((leave >= enter) & (enter > 0.05), enter, np.inf)[None, :]
        height = CAMERA_HEIGHT_M - down * dist
        hit = (height >= low) & (height <= high) & (dist < depth)
        depth = np.where(hit, dist, depth)
    return np.where(np.isfinite(depth), depth * 1000, 8000).round().astype(np.uint16)


def warnings_while(camera, extra=lambda t: [], seconds=1.5, fps=30, boxes=ROOM):
    """Move the camera along camera(t) -> (x, z, yaw); return every warning given."""
    watcher, found = ApproachWatcher(), []
    for i in range(int(seconds * fps)):
        t = i / fps
        answer = watcher.update(room_picture(boxes + extra(t), *camera(t)), t)
        if answer[0] is not None:
            found.append((round(t, 2), answer))
    return found


def test_camera_turn_is_measured():
    before = top_down_scan(room_picture(ROOM, yaw_deg=0))
    after = top_down_scan(room_picture(ROOM, 0.0, 0.05, yaw_deg=5))
    rotation, shift = camera_motion(before, after)
    assert math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])) == \
        pytest.approx(-5, abs=0.5)
    assert shift == pytest.approx([0.0, 0.05], abs=0.02)


@pytest.mark.parametrize('name, camera', [
    ('turning left', lambda t: (0, 0, -45 * t)),
    ('turning right', lambda t: (0, 0, 45 * t)),
    ('walking', lambda t: (0, 0.8 * t, 0)),
    ('walking at an angle', lambda t: (0.4 * t, 0.8 * t, 0)),
    ('walking and sweeping the cane', lambda t: (0, 0.8 * t, 20 * math.sin(2 * math.pi * t))),
])
def test_still_room_never_warns_however_the_camera_moves(name, camera):
    assert warnings_while(camera) == []


def test_person_walking_alongside_never_warns():
    alongside = warnings_while(lambda t: (0, 0.8 * t, 0),
                               lambda t: [person(-0.9, 1.5 + 0.8 * t)], boxes=ROOM[1:])
    assert alongside == []


@pytest.mark.parametrize('name, camera', [
    ('standing still', lambda t: (0, 0, 0)),
    ('walking', lambda t: (0, 0.8 * t, 0)),
    ('walking and sweeping the cane', lambda t: (0, 0.8 * t, 20 * math.sin(2 * math.pi * t))),
])
def test_person_coming_in_from_the_left_warns_early(name, camera):
    # walks toward the path at 1 m/s, from 1.5 m left of it
    found = warnings_while(camera, lambda t: [person(-1.5 + t, 2.4)], seconds=1.0,
                           boxes=ROOM[1:])
    assert found, 'no warning at all'
    first_t, (_, side) = found[0]
    assert side == 'left'
    assert 1.5 - first_t - 0.2 - 0.35 > 0.3          # still 30 cm+ outside the path
