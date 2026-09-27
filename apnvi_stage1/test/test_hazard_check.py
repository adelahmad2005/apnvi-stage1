"""Tests for hazard_check.py using fake depth pictures made in Python (no camera, no ROS)."""

from apnvi_stage1.hazard_check import (
    CAMERA_HEIGHT_M, check_depth, DEFAULT_INTRINSICS, level_for_distance,
    level_with_dead_zone, nearest_distance, path_mask, walking_lane)
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
    u0 = max(0, int(CX + left_m * FX / z_m))
    u1 = min(width, int(CX + right_m * FX / z_m))
    v0 = max(0, int(CY + (CAMERA_HEIGHT_M - high_m) * FY / z_m))
    v1 = min(height, int(CY + (CAMERA_HEIGHT_M - low_m) * FY / z_m))
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
