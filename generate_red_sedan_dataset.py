#!/usr/bin/env python3
"""
Deterministic CARLA Dataset Generator for Prompt-Driven Tracking Benchmark

Generates a dataset with:
- Exactly one RED sedan (target)
- Multiple non-red distractor sedans
- Fixed overhead/drone RGB camera
- GT 2D bounding boxes per frame

Everything runs in SYNCHRONOUS mode for determinism.

Prompt for tracker: "red sedan"
"""

import argparse
import glob
import json
import os
import queue
import random
import sys
import numpy as np
from PIL import Image
import cv2

import carla


# =============================================================================
# CONFIGURATION
# =============================================================================

# Sedan blueprints allow-list (will skip if not available in current CARLA version)
SEDAN_BLUEPRINTS = [
    "vehicle.tesla.model3",
    "vehicle.audi.a2",
    "vehicle.bmw.grandtourer",
    "vehicle.mercedes.coupe",
    "vehicle.toyota.prius",
    "vehicle.ford.mustang",
]

# Target color (RED) - using deeper red for consistent appearance across metallic paints
TARGET_COLOR = "180,30,30"

# Colors for distractors (explicitly NOT red)
DISTRACTOR_COLORS = [
    "0,0,255",      # Blue
    "255,255,255",  # White
    "0,0,0",        # Black
    "128,128,128",  # Gray
    "0,255,0",      # Green
    "255,255,0",    # Yellow
    "0,255,255",    # Cyan
    "64,64,64",     # Dark gray
]

# Non-sedan blueprints for same-color-different-class scenario
NON_SEDAN_BLUEPRINTS = [
    "vehicle.nissan.patrol_2021",
    "vehicle.dodge.charger_2020",
    "vehicle.jeep.wrangler_rubicon",
    "vehicle.chevrolet.impala",
    "vehicle.lincoln.mkz_2020",
    "vehicle.mini.cooper_s_2021",
]

# Colors easily confused with target red
CONFUSABLE_COLORS = [
    "200,100,0",   # Orange
    "130,0,0",     # Maroon
    "150,20,20",   # Dark red
]

# 17 evaluation scenarios
EVAL_SCENARIO_CONFIGS = [
    # --- Weather variations ---
    {
        "name": "clear_day_baseline",
        "scenario_id": 1,
        "description": "Explicit clear sky with high sun — static camera, target drives through",
        "map": "Town10HD_Opt",
        "weather": {"cloudiness": 0, "precipitation": 0, "sun_altitude_angle": 60,
                     "sun_azimuth_angle": 220, "fog_density": 0, "wetness": 0,
                     "precipitation_deposits": 0, "wind_intensity": 10},
        "num_distractors": 10,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "static",
        "num_frames": 800,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 0.0},
        "seed": 100,
    },
    {
        "name": "overcast",
        "scenario_id": 2,
        "description": "Heavy cloud cover with flat diffuse light — static camera",
        "map": "Town10HD_Opt",
        "weather": {"cloudiness": 90, "precipitation": 0, "sun_altitude_angle": 45,
                     "sun_azimuth_angle": 220, "fog_density": 0, "wetness": 0,
                     "precipitation_deposits": 0, "wind_intensity": 30},
        "num_distractors": 10,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "static",
        "num_frames": 800,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 0.0},
        "seed": 200,
    },
    {
        "name": "heavy_rain",
        "scenario_id": 3,
        "description": "Heavy precipitation with wet roads — loose follow, target drifts in/out of FOV",
        "map": "Town10HD_Opt",
        "weather": {"cloudiness": 80, "precipitation": 80, "sun_altitude_angle": 40,
                     "sun_azimuth_angle": 220, "fog_density": 10, "wetness": 100,
                     "precipitation_deposits": 80, "wind_intensity": 50},
        "num_distractors": 10,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "loose_follow",
        "num_frames": 800,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 0.0,
                    "loose_follow_lag": 40},
        "seed": 300,
    },
    {
        "name": "dusk_golden_hour",
        "scenario_id": 4,
        "description": "Low sun angle with long shadows — loose follow",
        "map": "Town10HD_Opt",
        "weather": {"cloudiness": 20, "precipitation": 0, "sun_altitude_angle": 15,
                     "sun_azimuth_angle": 280, "fog_density": 0, "wetness": 0,
                     "precipitation_deposits": 0, "wind_intensity": 10},
        "num_distractors": 10,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "loose_follow",
        "num_frames": 800,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 0.0,
                    "loose_follow_lag": 40},
        "seed": 400,
    },
    {
        "name": "night",
        "scenario_id": 5,
        "description": "Nighttime with streetlights only — loose follow",
        "map": "Town10HD_Opt",
        "weather": {"cloudiness": 50, "precipitation": 0, "sun_altitude_angle": -30,
                     "sun_azimuth_angle": 220, "fog_density": 0, "wetness": 0,
                     "precipitation_deposits": 0, "wind_intensity": 10},
        "num_distractors": 10,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "loose_follow",
        "num_frames": 800,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 0.0,
                    "loose_follow_lag": 40},
        "seed": 500,
    },
    {
        "name": "dense_fog",
        "scenario_id": 6,
        "description": "Dense fog with short visibility — loose follow",
        "map": "Town10HD_Opt",
        "weather": {"cloudiness": 60, "precipitation": 0, "sun_altitude_angle": 45,
                     "sun_azimuth_angle": 220, "fog_density": 70, "fog_distance": 10,
                     "wetness": 30, "precipitation_deposits": 0, "wind_intensity": 5},
        "num_distractors": 10,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "loose_follow",
        "num_frames": 800,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 0.0,
                    "loose_follow_lag": 40},
        "seed": 600,
    },
    # --- Distractor variations ---
    {
        "name": "color_confusable",
        "scenario_id": 7,
        "description": "Orange/maroon/dark-red distractors — loose follow",
        "map": "Town10HD_Opt",
        "weather": None,
        "num_distractors": 10,
        "num_targets": 1,
        "distractor_colors": ["200,100,0", "130,0,0", "150,20,20"],
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "loose_follow",
        "num_frames": 800,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 0.0,
                    "loose_follow_lag": 40},
        "seed": 700,
    },
    {
        "name": "same_color_diff_class",
        "scenario_id": 8,
        "description": "Red SUVs/trucks as distractors — loose follow",
        "map": "Town10HD_Opt",
        "weather": None,
        "num_distractors": 10,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": ["vehicle.nissan.patrol_2021", "vehicle.dodge.charger_2020",
                                   "vehicle.jeep.wrangler_rubicon", "vehicle.chevrolet.impala",
                                   "vehicle.lincoln.mkz_2020", "vehicle.mini.cooper_s_2021"],
        "distractor_use_target_color": True,
        "camera_mode": "loose_follow",
        "num_frames": 800,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 0.0,
                    "loose_follow_lag": 40},
        "seed": 800,
    },
    {
        "name": "high_density",
        "scenario_id": 9,
        "description": "25 distractor vehicles — static camera, crowded scene",
        "map": "Town10HD_Opt",
        "weather": None,
        "num_distractors": 25,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "static",
        "num_frames": 800,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 0.0},
        "seed": 900,
    },
    # --- Camera variations ---
    {
        "name": "high_altitude",
        "scenario_id": 10,
        "description": "Camera at 25m height — static, wide overhead view",
        "map": "Town10HD_Opt",
        "weather": None,
        "num_distractors": 10,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "static",
        "num_frames": 800,
        "camera": {"height_offset": 25.0, "follow_distance": 30.0, "pitch": -15.0, "lateral_offset": 0.0},
        "seed": 1000,
    },
    {
        "name": "low_altitude_steep",
        "scenario_id": 11,
        "description": "Camera at 6m, steep -40 pitch — loose follow, street-level view",
        "map": "Town10HD_Opt",
        "weather": None,
        "num_distractors": 10,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "loose_follow",
        "num_frames": 800,
        "camera": {"height_offset": 6.0, "follow_distance": 20.0, "pitch": -40.0, "lateral_offset": 0.0,
                    "loose_follow_lag": 30},
        "seed": 1100,
    },
    {
        "name": "side_follow",
        "scenario_id": 12,
        "description": "Camera with 8m lateral offset — loose follow, side perspective",
        "map": "Town10HD_Opt",
        "weather": None,
        "num_distractors": 10,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "loose_follow",
        "num_frames": 800,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 8.0,
                    "loose_follow_lag": 30},
        "seed": 1200,
    },
    # --- Map variations ---
    {
        "name": "town03_suburban",
        "scenario_id": 13,
        "description": "Suburban layout with Town03 — static camera",
        "map": "Town03",
        "weather": None,
        "num_distractors": 10,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "static",
        "num_frames": 800,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 0.0},
        "seed": 1300,
    },
    {
        "name": "town05_highway",
        "scenario_id": 14,
        "description": "Multi-lane highway Town05 — loose follow",
        "map": "Town05",
        "weather": None,
        "num_distractors": 10,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "loose_follow",
        "num_frames": 800,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 0.0,
                    "loose_follow_lag": 40},
        "seed": 1400,
    },
    # --- Edge cases ---
    {
        "name": "multiple_red_sedans",
        "scenario_id": 15,
        "description": "3 red sedan targets — static camera, multi-target detection",
        "map": "Town10HD_Opt",
        "weather": None,
        "num_distractors": 10,
        "num_targets": 3,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "static",
        "num_frames": 800,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 0.0},
        "seed": 1500,
    },
    {
        "name": "long_sequence",
        "scenario_id": 16,
        "description": "Extended 1500-frame sequence — static camera, tests reappearance",
        "map": "Town10HD_Opt",
        "weather": None,
        "num_distractors": 10,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "static",
        "num_frames": 1500,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 0.0},
        "seed": 1600,
    },
    {
        "name": "dense_urban_traffic",
        "scenario_id": 17,
        "description": "Dense urban with 20 distractors — static camera, frequent stops",
        "map": "Town10HD_Opt",
        "weather": None,
        "num_distractors": 20,
        "num_targets": 1,
        "distractor_colors": None,
        "distractor_blueprints": None,
        "distractor_use_target_color": False,
        "camera_mode": "static",
        "num_frames": 800,
        "camera": {"height_offset": 12.0, "follow_distance": 20.0, "pitch": -15.0, "lateral_offset": 0.0},
        "seed": 1700,
    },
]

# Camera attributes
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FOV = 110

# Camera height above spawn point for drone view
CAMERA_HEIGHT_OFFSET = 12.0  # meters above ground
CAMERA_PITCH = -15.0  # degrees, looking downward


# =============================================================================
# PROJECTION FUNCTIONS (from original code, with fixes)
# =============================================================================

def build_projection_matrix(w, h, fov):
    """
    Build camera intrinsic matrix K for 3D->2D projection.

    Args:
        w: Image width in pixels
        h: Image height in pixels
        fov: Horizontal field of view in degrees

    Returns:
        3x3 numpy array (camera intrinsic matrix)
    """
    # Focal length from horizontal FOV
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))

    K = np.identity(3)
    K[0, 0] = focal  # fx
    K[1, 1] = focal  # fy
    K[0, 2] = w / 2.0  # cx (principal point x)
    K[1, 2] = h / 2.0  # cy (principal point y)

    return K


def get_image_point(loc, K, world_2_camera):
    """
    Project a 3D world location to 2D image coordinates.

    CARLA uses Unreal Engine coordinate system:
        - X: forward
        - Y: right
        - Z: up

    Standard camera coordinate system:
        - X: right
        - Y: down
        - Z: forward (into the image)

    Conversion: (x, y, z)_UE -> (y, -z, x)_camera

    Args:
        loc: carla.Location object (world coordinates)
        K: 3x3 camera intrinsic matrix
        world_2_camera: 4x4 world-to-camera transform matrix

    Returns:
        (u, v) pixel coordinates, or None if behind camera
    """
    # Homogeneous world coordinates
    point = np.array([loc.x, loc.y, loc.z, 1.0])

    # Transform to camera coordinates (still in UE convention)
    point_camera = np.dot(world_2_camera, point)

    # Convert from UE to standard camera coordinates
    # UE: (x=forward, y=right, z=up) -> Camera: (x=right, y=down, z=forward)
    point_camera_std = np.array([
        point_camera[1],   # x_cam = y_ue (right)
        -point_camera[2],  # y_cam = -z_ue (down)
        point_camera[0]    # z_cam = x_ue (forward/depth)
    ])

    # Check if point is behind camera
    if point_camera_std[2] <= 0:
        return None

    # Project to image plane
    point_img = np.dot(K, point_camera_std)

    # Normalize by depth
    u = point_img[0] / point_img[2]
    v = point_img[1] / point_img[2]

    return (u, v)


def compute_2d_bbox(actor, camera_transform, K, world_2_camera, img_w, img_h, max_dist=70.0):
    """
    Compute 2D bounding box for a vehicle actor.

    Args:
        actor: CARLA vehicle actor
        camera_transform: Camera transform
        K: Camera intrinsic matrix
        world_2_camera: World-to-camera matrix
        img_w: Image width
        img_h: Image height
        max_dist: Maximum distance to consider (meters)

    Returns:
        dict with bbox_xyxy, bbox_xywh, or None if invalid/not visible
    """
    # Get actor location and bounding box
    actor_transform = actor.get_transform()
    actor_location = actor_transform.location
    camera_location = camera_transform.location

    # Distance check
    dist = actor_location.distance(camera_location)
    if dist > max_dist:
        return None

    # Check if in front of camera (dot product of forward vector and ray)
    forward_vec = camera_transform.get_forward_vector()
    ray = actor_location - camera_location

    if forward_vec.dot(ray) <= 0:
        return None

    # Get 3D bounding box vertices in world coordinates
    bb = actor.bounding_box
    vertices_world = bb.get_world_vertices(actor_transform)

    # Project all vertices to 2D
    u_coords = []
    v_coords = []

    for vertex in vertices_world:
        point_2d = get_image_point(vertex, K, world_2_camera)
        if point_2d is not None:
            u_coords.append(point_2d[0])
            v_coords.append(point_2d[1])

    # Need at least one valid projection
    if len(u_coords) == 0:
        return None

    # Get min/max
    x_min = min(u_coords)
    x_max = max(u_coords)
    y_min = min(v_coords)
    y_max = max(v_coords)

    # Clamp to image bounds
    x_min = max(0, min(x_min, img_w))
    x_max = max(0, min(x_max, img_w))
    y_min = max(0, min(y_min, img_h))
    y_max = max(0, min(y_max, img_h))

    # Check for valid box (non-zero area)
    if x_max <= x_min or y_max <= y_min:
        return None

    # Convert to integers
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    # Compute xywh format
    w = x_max - x_min
    h = y_max - y_min

    return {
        "bbox_xyxy": [x_min, y_min, x_max, y_max],
        "bbox_xywh": [x_min, y_min, w, h]
    }


# =============================================================================
# CAMERA HELPERS
# =============================================================================

def create_camera_transform_from_spawn(spawn_point, height_offset=CAMERA_HEIGHT_OFFSET, pitch=CAMERA_PITCH,
                                       look_at_location=None):
    """
    Create a drone-style camera transform based on a spawn point.

    Args:
        spawn_point: carla.Transform of the spawn point
        height_offset: Height above ground for camera
        pitch: Camera pitch angle (negative = looking down)
        look_at_location: Optional carla.Location to point camera toward

    Returns:
        carla.Transform for camera placement
    """
    # Position camera above the spawn point
    camera_location = carla.Location(
        x=spawn_point.location.x,
        y=spawn_point.location.y,
        z=spawn_point.location.z + height_offset
    )

    if look_at_location is not None:
        # Compute yaw to look at the target location
        dx = look_at_location.x - camera_location.x
        dy = look_at_location.y - camera_location.y
        yaw = np.degrees(np.arctan2(dy, dx))
    else:
        # Use spawn point's yaw (direction road is facing)
        yaw = spawn_point.rotation.yaw

    camera_rotation = carla.Rotation(
        pitch=pitch,
        yaw=yaw,
        roll=0.0
    )

    return carla.Transform(camera_location, camera_rotation)


def compute_drone_follow_transform(target_transform, height_offset=CAMERA_HEIGHT_OFFSET,
                                   follow_distance=20.0, pitch=CAMERA_PITCH,
                                   lateral_offset=0.0):
    """
    Compute camera transform for drone-style following of target.

    Camera positions itself behind and above the target, looking at it.

    Args:
        target_transform: Current transform of target vehicle
        height_offset: Height above target
        follow_distance: Distance behind target (along its backward direction)
        pitch: Camera pitch angle
        lateral_offset: Perpendicular offset to the right of the target (meters)

    Returns:
        carla.Transform for camera
    """
    target_loc = target_transform.location
    target_yaw = target_transform.rotation.yaw

    # Compute position behind target
    yaw_rad = np.radians(target_yaw)
    # Behind = opposite of forward direction
    behind_x = -np.cos(yaw_rad) * follow_distance
    behind_y = -np.sin(yaw_rad) * follow_distance

    # Perpendicular offset (right direction relative to target heading)
    right_x = -np.sin(yaw_rad) * lateral_offset
    right_y = np.cos(yaw_rad) * lateral_offset

    camera_location = carla.Location(
        x=target_loc.x + behind_x + right_x,
        y=target_loc.y + behind_y + right_y,
        z=target_loc.z + height_offset
    )

    if lateral_offset != 0.0:
        # Compute yaw from camera position to target for off-center view
        dx = target_loc.x - camera_location.x
        dy = target_loc.y - camera_location.y
        camera_yaw = np.degrees(np.arctan2(dy, dx))
    else:
        # Camera looks at target (same yaw as target, so we look forward along its path)
        camera_yaw = target_yaw

    camera_rotation = carla.Rotation(
        pitch=pitch,
        yaw=camera_yaw,
        roll=0.0
    )

    return carla.Transform(camera_location, camera_rotation)


# =============================================================================
# WEATHER HELPER
# =============================================================================

def set_weather(world, weather_params=None):
    """
    Apply weather parameters to the world.

    Args:
        world: CARLA world
        weather_params: Dict of weather parameters, or None for ClearNoon default.
            Supported keys: cloudiness, precipitation, precipitation_deposits,
            wind_intensity, sun_azimuth_angle, sun_altitude_angle,
            fog_density, fog_distance, wetness.

    Returns:
        Dict of the applied weather parameter values.
    """
    weather = carla.WeatherParameters.ClearNoon
    if weather_params is not None:
        weather = carla.WeatherParameters(
            cloudiness=weather_params.get("cloudiness", 0),
            precipitation=weather_params.get("precipitation", 0),
            precipitation_deposits=weather_params.get("precipitation_deposits", 0),
            wind_intensity=weather_params.get("wind_intensity", 0),
            sun_azimuth_angle=weather_params.get("sun_azimuth_angle", 220),
            sun_altitude_angle=weather_params.get("sun_altitude_angle", 60),
            fog_density=weather_params.get("fog_density", 0),
            fog_distance=weather_params.get("fog_distance", 0),
            wetness=weather_params.get("wetness", 0),
        )
    world.set_weather(weather)

    applied = {
        "cloudiness": weather.cloudiness,
        "precipitation": weather.precipitation,
        "precipitation_deposits": weather.precipitation_deposits,
        "wind_intensity": weather.wind_intensity,
        "sun_azimuth_angle": weather.sun_azimuth_angle,
        "sun_altitude_angle": weather.sun_altitude_angle,
        "fog_density": weather.fog_density,
        "fog_distance": weather.fog_distance,
        "wetness": weather.wetness,
    }
    print(f"[INFO] Weather set: {applied}")
    return applied


# =============================================================================
# SPAWN HELPERS
# =============================================================================

def get_available_sedan_blueprints(bp_lib):
    """Get list of available sedan blueprints from our allow-list."""
    available = []
    for bp_id in SEDAN_BLUEPRINTS:
        bp = bp_lib.find(bp_id) if bp_lib.find(bp_id) else None
        try:
            bp = bp_lib.find(bp_id)
            if bp is not None:
                available.append(bp_id)
        except:
            print(f"[WARN] Blueprint {bp_id} not available, skipping")
    return available


def get_spawn_region(spawn_points, center_location, num_points=30):
    """
    Get spawn points nearest to center location.

    Args:
        spawn_points: List of all spawn points
        center_location: carla.Location to center around
        num_points: Number of nearest points to return

    Returns:
        List of spawn points sorted by distance
    """
    # Calculate distances
    points_with_dist = []
    for sp in spawn_points:
        dist = sp.location.distance(center_location)
        points_with_dist.append((sp, dist))

    # Sort by distance
    points_with_dist.sort(key=lambda x: x[1])

    # Return nearest points
    return [p[0] for p in points_with_dist[:num_points]]


def get_forward_spawn_points(spawn_points, camera_spawn, min_dist=15.0, max_dist=50.0):
    """
    Get spawn points that are IN FRONT of the camera spawn point.

    This ensures vehicles spawn where the camera can see them (along the road).

    Args:
        spawn_points: List of all spawn points
        camera_spawn: The spawn point where camera will be placed
        min_dist: Minimum distance from camera (meters)
        max_dist: Maximum distance from camera (meters)

    Returns:
        List of spawn points in front of camera, sorted by distance
    """
    # Get camera's forward direction (from spawn point rotation)
    yaw_rad = np.radians(camera_spawn.rotation.yaw)
    forward_x = np.cos(yaw_rad)
    forward_y = np.sin(yaw_rad)

    camera_loc = camera_spawn.location
    forward_points = []

    for sp in spawn_points:
        # Vector from camera to spawn point
        dx = sp.location.x - camera_loc.x
        dy = sp.location.y - camera_loc.y
        dist = np.sqrt(dx*dx + dy*dy)

        if dist < min_dist or dist > max_dist:
            continue

        # Normalize
        if dist > 0:
            dx_norm = dx / dist
            dy_norm = dy / dist

            # Dot product with forward vector (1.0 = directly ahead, 0 = perpendicular, -1 = behind)
            dot = forward_x * dx_norm + forward_y * dy_norm

            # Only keep points that are mostly in front (dot > 0.5 means within ~60 degrees of forward)
            if dot > 0.5:
                forward_points.append((sp, dist, dot))

    # Sort by dot product (most directly ahead first), then by distance
    forward_points.sort(key=lambda x: (-x[2], x[1]))

    return [p[0] for p in forward_points]


def spawn_vehicles(world, bp_lib, spawn_points, num_distractors, seed):
    """
    Spawn target (red sedan) and distractor sedans.

    DETERMINISM: Uses fixed seed and sorted spawn points.

    Args:
        world: CARLA world
        bp_lib: Blueprint library
        spawn_points: List of spawn points to use
        num_distractors: Number of distractor vehicles
        seed: Random seed

    Returns:
        (target_actor, distractor_actors, spawned_info)
    """
    random.seed(seed)

    # Get available blueprints
    available_bps = get_available_sedan_blueprints(bp_lib)
    if len(available_bps) == 0:
        raise RuntimeError("No sedan blueprints available!")

    print(f"[INFO] Available sedan blueprints: {available_bps}")

    # Shuffle spawn points deterministically
    spawn_points_copy = list(spawn_points)
    random.shuffle(spawn_points_copy)

    # We need num_distractors + 1 (target) spawn points
    total_needed = num_distractors + 1
    if len(spawn_points_copy) < total_needed:
        print(f"[WARN] Only {len(spawn_points_copy)} spawn points available, need {total_needed}")
        total_needed = len(spawn_points_copy)
        num_distractors = total_needed - 1

    spawned_info = []
    target_actor = None
    distractor_actors = []

    spawn_idx = 0

    # Spawn target (RED sedan)
    target_bp_id = random.choice(available_bps)
    target_bp = bp_lib.find(target_bp_id)

    # Set color to RED
    if target_bp.has_attribute('color'):
        target_bp.set_attribute('color', TARGET_COLOR)

    target_actor = world.try_spawn_actor(target_bp, spawn_points_copy[spawn_idx])
    if target_actor is None:
        raise RuntimeError("Failed to spawn target vehicle!")

    spawned_info.append({
        "actor_id": target_actor.id,
        "type_id": target_bp_id,
        "color": TARGET_COLOR,
        "is_target": True
    })
    print(f"[INFO] Spawned TARGET: {target_bp_id} (id={target_actor.id}) color={TARGET_COLOR}")
    spawn_idx += 1

    # Spawn distractors (NOT red)
    distractor_color_idx = 0
    for i in range(num_distractors):
        if spawn_idx >= len(spawn_points_copy):
            break

        # Pick blueprint
        dist_bp_id = available_bps[i % len(available_bps)]
        dist_bp = bp_lib.find(dist_bp_id)

        # Set color (cycle through distractor colors, never red)
        dist_color = DISTRACTOR_COLORS[distractor_color_idx % len(DISTRACTOR_COLORS)]
        distractor_color_idx += 1

        if dist_bp.has_attribute('color'):
            dist_bp.set_attribute('color', dist_color)

        actor = world.try_spawn_actor(dist_bp, spawn_points_copy[spawn_idx])
        if actor is not None:
            distractor_actors.append(actor)
            spawned_info.append({
                "actor_id": actor.id,
                "type_id": dist_bp_id,
                "color": dist_color,
                "is_target": False
            })
            print(f"[INFO] Spawned DISTRACTOR: {dist_bp_id} (id={actor.id}) color={dist_color}")
        else:
            print(f"[WARN] Failed to spawn distractor at spawn point {spawn_idx}")

        spawn_idx += 1

    return target_actor, distractor_actors, spawned_info


def spawn_vehicles_targeted(world, bp_lib, target_spawn_points, distractor_spawn_points,
                            num_distractors, seed, num_targets=1,
                            distractor_colors=None, distractor_blueprints=None,
                            distractor_use_target_color=False):
    """
    Spawn target(s) at forward-facing spawn points, distractors at nearby points.

    Args:
        world: CARLA world
        bp_lib: Blueprint library
        target_spawn_points: Spawn points in front of camera (for target)
        distractor_spawn_points: Nearby spawn points (for distractors)
        num_distractors: Number of distractor vehicles
        seed: Random seed
        num_targets: Number of red sedan targets to spawn
        distractor_colors: Override list of distractor colors (or None for default)
        distractor_blueprints: Override list of blueprint IDs for distractors (or None for sedans)
        distractor_use_target_color: If True, distractors use TARGET_COLOR

    Returns:
        (target_actors_list, distractor_actors, spawned_info)
    """
    random.seed(seed)

    available_bps = get_available_sedan_blueprints(bp_lib)
    if len(available_bps) == 0:
        raise RuntimeError("No sedan blueprints available!")

    print(f"[INFO] Available sedan blueprints: {available_bps}")

    # Resolve distractor blueprint list
    if distractor_blueprints is not None:
        available_dist_bps = []
        for bp_id in distractor_blueprints:
            try:
                bp = bp_lib.find(bp_id)
                if bp is not None:
                    available_dist_bps.append(bp_id)
            except Exception:
                print(f"[WARN] Distractor blueprint {bp_id} not available, skipping")
        if len(available_dist_bps) == 0:
            print("[WARN] No distractor blueprints available, falling back to sedans")
            available_dist_bps = available_bps
    else:
        available_dist_bps = available_bps

    # Resolve distractor color list
    colors_for_distractors = distractor_colors if distractor_colors is not None else DISTRACTOR_COLORS

    spawned_info = []
    target_actors = []
    distractor_actors = []
    used_spawn_locations = set()

    # SPAWN TARGET(S) at forward-facing points
    target_spawn_idx = 0
    for t_idx in range(num_targets):
        target_bp_id = random.choice(available_bps)
        target_bp = bp_lib.find(target_bp_id)

        if target_bp.has_attribute('color'):
            target_bp.set_attribute('color', TARGET_COLOR)

        spawned = False
        for target_spawn in target_spawn_points[target_spawn_idx:target_spawn_idx + 5]:
            actor = world.try_spawn_actor(target_bp, target_spawn)
            if actor is not None:
                used_spawn_locations.add((target_spawn.location.x, target_spawn.location.y))
                target_actors.append(actor)
                spawned_info.append({
                    "actor_id": actor.id,
                    "type_id": target_bp_id,
                    "color": TARGET_COLOR,
                    "is_target": True
                })
                print(f"[INFO] Spawned TARGET {t_idx + 1}/{num_targets}: {target_bp_id} "
                      f"(id={actor.id}) color={TARGET_COLOR}")
                spawned = True
                target_spawn_idx += 1
                break
            target_spawn_idx += 1

        if not spawned:
            if t_idx == 0:
                raise RuntimeError("Failed to spawn primary target vehicle!")
            print(f"[WARN] Failed to spawn target {t_idx + 1}/{num_targets}")

    # SPAWN DISTRACTORS at nearby points (avoiding target's spawn)
    distractor_color_idx = 0
    spawned_count = 0

    # Shuffle distractor spawn points
    distractor_spawns_shuffled = list(distractor_spawn_points)
    random.shuffle(distractor_spawns_shuffled)

    for spawn_point in distractor_spawns_shuffled:
        if spawned_count >= num_distractors:
            break

        # Skip if too close to used location
        loc_key = (spawn_point.location.x, spawn_point.location.y)
        if loc_key in used_spawn_locations:
            continue

        dist_bp_id = available_dist_bps[spawned_count % len(available_dist_bps)]
        dist_bp = bp_lib.find(dist_bp_id)

        if distractor_use_target_color:
            dist_color = TARGET_COLOR
        else:
            dist_color = colors_for_distractors[distractor_color_idx % len(colors_for_distractors)]
        distractor_color_idx += 1

        if dist_bp.has_attribute('color'):
            dist_bp.set_attribute('color', dist_color)

        actor = world.try_spawn_actor(dist_bp, spawn_point)
        if actor is not None:
            distractor_actors.append(actor)
            used_spawn_locations.add(loc_key)
            spawned_info.append({
                "actor_id": actor.id,
                "type_id": dist_bp_id,
                "color": dist_color,
                "is_target": False
            })
            print(f"[INFO] Spawned DISTRACTOR: {dist_bp_id} (id={actor.id}) color={dist_color}")
            spawned_count += 1

    print(f"[INFO] Spawned {len(target_actors)} targets, {spawned_count} distractors")
    return target_actors, distractor_actors, spawned_info


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def run_scenario(client, world, bp_lib, spawn_points, scenario_idx, scenario_dir,
                 num_frames, num_distractors, seed, camera_spawn_point, tm_port,
                 follow_mode=False, weather_params=None, num_targets=1,
                 distractor_colors=None, distractor_blueprints=None,
                 distractor_use_target_color=False, camera_config=None,
                 scenario_name=None, scenario_description=None,
                 camera_mode=None):
    """
    Run a single scenario: spawn vehicles, capture frames, create video.

    Args:
        client: CARLA client
        world: CARLA world
        bp_lib: Blueprint library
        spawn_points: All available spawn points
        scenario_idx: Scenario index number
        scenario_dir: Output directory for this scenario
        num_frames: Number of frames to capture
        num_distractors: Number of distractor vehicles
        seed: Random seed for this scenario
        camera_spawn_point: Spawn point for camera base position
        tm_port: Traffic manager port
        follow_mode: If True, camera follows target (drone mode). Ignored if camera_mode set.
        weather_params: Dict of weather parameters or None for default
        num_targets: Number of red sedan targets to spawn
        distractor_colors: Override distractor color list
        distractor_blueprints: Override distractor blueprint list
        distractor_use_target_color: If True, distractors use TARGET_COLOR
        camera_config: Dict with height_offset, follow_distance, pitch, lateral_offset,
                       loose_follow_lag
        scenario_name: Human-readable scenario name for metadata
        scenario_description: Scenario description for metadata
        camera_mode: "static", "follow", or "loose_follow". Overrides follow_mode.

    Returns:
        dict with scenario metadata
    """
    # Resolve camera_mode from camera_mode or legacy follow_mode
    if camera_mode is None:
        camera_mode = "follow" if follow_mode else "static"

    spawned_actors = []
    camera = None

    # Resolve camera config with defaults
    cam_height_offset = CAMERA_HEIGHT_OFFSET
    cam_follow_distance = 20.0
    cam_pitch = CAMERA_PITCH
    cam_lateral_offset = 0.0
    cam_loose_follow_lag = 40  # frames (~4 seconds at 10 FPS)
    if camera_config is not None:
        cam_height_offset = camera_config.get("height_offset", CAMERA_HEIGHT_OFFSET)
        cam_follow_distance = camera_config.get("follow_distance", 20.0)
        cam_pitch = camera_config.get("pitch", CAMERA_PITCH)
        cam_lateral_offset = camera_config.get("lateral_offset", 0.0)
        cam_loose_follow_lag = camera_config.get("loose_follow_lag", 40)

    # Scale max_dist based on camera height
    max_dist = max(70.0, cam_height_offset * 5)

    try:
        # Apply weather
        applied_weather = set_weather(world, weather_params)

        # Create output directories for this scenario
        images_dir = os.path.join(scenario_dir, "images")
        video_dir = os.path.join(scenario_dir, "video")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        # Get spawn points IN FRONT of camera (along the road direction)
        # This ensures target spawns where camera can see it
        # Try progressively wider search if no forward spawns found
        forward_spawns = get_forward_spawn_points(spawn_points, camera_spawn_point, min_dist=15.0, max_dist=50.0)

        if len(forward_spawns) < 1:
            print(f"[WARN] No forward spawns at default range, widening to 10-80m...")
            forward_spawns = get_forward_spawn_points(spawn_points, camera_spawn_point, min_dist=10.0, max_dist=80.0)

        if len(forward_spawns) < 1:
            # Last resort: use nearest spawn points regardless of direction
            print(f"[WARN] Still no forward spawns, using nearest spawn points as fallback")
            forward_spawns = [sp for sp in get_spawn_region(spawn_points, camera_spawn_point.location, num_points=10)]

        if len(forward_spawns) < 1:
            raise RuntimeError(f"No valid spawn points found near camera at {camera_spawn_point.location}")

        print(f"[INFO] Scenario {scenario_idx}: Found {len(forward_spawns)} spawn points in front of camera")

        # Target spawns at the most direct forward point
        # Distractors spawn at nearby points (mix of forward and general nearby)
        spawn_region = get_spawn_region(spawn_points, camera_spawn_point.location, num_points=30)

        # SPAWN VEHICLES - target(s) at forward point, distractors nearby
        targets, distractors, vehicle_info = spawn_vehicles_targeted(
            world, bp_lib, forward_spawns, spawn_region, num_distractors, seed,
            num_targets=num_targets,
            distractor_colors=distractor_colors,
            distractor_blueprints=distractor_blueprints,
            distractor_use_target_color=distractor_use_target_color
        )

        # Primary follow target is the first one
        target = targets[0]
        all_vehicles = targets + distractors
        spawned_actors.extend(all_vehicles)

        # Build lookup for vehicle info by actor ID
        vehicle_info_map = {v["actor_id"]: v for v in vehicle_info}

        # Tick world to register actors and get their positions
        world.tick()

        # Get target's initial location
        target_initial_location = target.get_transform().location
        print(f"[INFO] Target spawned at ({target_initial_location.x:.1f}, {target_initial_location.y:.1f}, {target_initial_location.z:.1f})")

        # CREATE CAMERA looking along road direction (where target is)
        camera_transform = create_camera_transform_from_spawn(
            camera_spawn_point,
            height_offset=cam_height_offset,
            pitch=cam_pitch,
            look_at_location=target_initial_location
        )
        camera_location = camera_transform.location
        camera_rotation = camera_transform.rotation

        print(f"[INFO] Camera positioned at ({camera_location.x:.1f}, {camera_location.y:.1f}, {camera_location.z:.1f}), "
              f"yaw={camera_rotation.yaw:.1f}° (pointing at target)")

        # Enable autopilot on all vehicles
        tm = client.get_trafficmanager(tm_port)
        for v in all_vehicles:
            v.set_autopilot(True, tm_port)
            tm.vehicle_percentage_speed_difference(v, 10.0)

        print(f"[INFO] Enabled autopilot for {len(all_vehicles)} vehicles")

        # Spawn camera
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(CAMERA_FOV))

        camera = world.spawn_actor(camera_bp, camera_transform)
        spawned_actors.append(camera)
        print(f"[INFO] Spawned RGB camera")

        # Setup image queue
        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # Build camera matrices
        K = build_projection_matrix(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV)

        # Prepare dataset structure
        map_name = world.get_map().name
        dataset = {
            "meta": {
                "prompt": "red sedan",
                "map": map_name,
                "fps": 10,
                "seed": seed,
                "scenario_id": scenario_idx,
                "scenario_name": scenario_name,
                "scenario_description": scenario_description,
                "camera_mode": camera_mode,
                "num_targets": num_targets,
                "weather": applied_weather,
                "camera": {
                    "initial_x": camera_location.x,
                    "initial_y": camera_location.y,
                    "initial_z": camera_location.z,
                    "pitch": camera_rotation.pitch,
                    "initial_yaw": camera_rotation.yaw,
                    "roll": camera_rotation.roll,
                    "w": CAMERA_WIDTH,
                    "h": CAMERA_HEIGHT,
                    "fov": CAMERA_FOV,
                    "height_offset": cam_height_offset,
                    "follow_distance": cam_follow_distance,
                    "lateral_offset": cam_lateral_offset,
                    "loose_follow_lag": cam_loose_follow_lag if camera_mode == "loose_follow" else None
                }
            },
            "images": [],
            "annotations": []
        }

        # Main capture loop
        mode_str = camera_mode.upper().replace("_", " ")
        print(f"[INFO] Scenario {scenario_idx}: Starting capture of {num_frames} frames ({mode_str})...")

        # Position history for loose_follow (lagged tracking)
        position_history = []

        for frame_idx in range(num_frames):
            # Update camera position based on camera_mode
            if camera_mode == "follow" and target.is_alive:
                # Strict follow: camera locked behind target every frame
                target_transform = target.get_transform()
                camera_transform = compute_drone_follow_transform(
                    target_transform,
                    height_offset=cam_height_offset,
                    follow_distance=cam_follow_distance,
                    pitch=cam_pitch,
                    lateral_offset=cam_lateral_offset
                )
                camera.set_transform(camera_transform)

            elif camera_mode == "loose_follow" and target.is_alive:
                # Lagged follow: camera goes where target WAS N frames ago
                target_transform = target.get_transform()
                position_history.append(carla.Transform(
                    carla.Location(
                        x=target_transform.location.x,
                        y=target_transform.location.y,
                        z=target_transform.location.z
                    ),
                    carla.Rotation(
                        pitch=target_transform.rotation.pitch,
                        yaw=target_transform.rotation.yaw,
                        roll=target_transform.rotation.roll
                    )
                ))

                if len(position_history) > cam_loose_follow_lag:
                    lagged_transform = position_history[-cam_loose_follow_lag]
                else:
                    lagged_transform = position_history[0]

                camera_transform = compute_drone_follow_transform(
                    lagged_transform,
                    height_offset=cam_height_offset,
                    follow_distance=cam_follow_distance,
                    pitch=cam_pitch,
                    lateral_offset=cam_lateral_offset
                )
                camera.set_transform(camera_transform)

            # camera_mode == "static": no camera update, stays at initial position

            world.tick()

            try:
                image = image_queue.get(timeout=2.0)
            except queue.Empty:
                print(f"[WARN] No image received for frame {frame_idx}, skipping")
                continue

            frame_id = image.frame

            # Get current camera transform (may have been updated in follow mode)
            current_camera_transform = camera.get_transform()

            # Convert and save image
            img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
            img_array = img_array.reshape((image.height, image.width, 4))
            img_rgb = img_array[:, :, :3][:, :, ::-1]

            img_filename = f"{frame_id:06d}.png"
            img_path = os.path.join(images_dir, img_filename)
            Image.fromarray(img_rgb).save(img_path)

            dataset["images"].append({
                "id": frame_id,
                "file_name": img_filename,
                "width": CAMERA_WIDTH,
                "height": CAMERA_HEIGHT
            })

            # Get world-to-camera matrix (use current transform, not initial)
            world_2_camera = np.array(current_camera_transform.get_inverse_matrix())

            # Compute bounding boxes for all vehicles
            for actor in all_vehicles:
                if not actor.is_alive:
                    continue

                info = vehicle_info_map[actor.id]
                actor_transform = actor.get_transform()

                bbox_result = compute_2d_bbox(
                    actor, current_camera_transform, K, world_2_camera,
                    CAMERA_WIDTH, CAMERA_HEIGHT, max_dist=max_dist
                )

                # Only annotate if visible (valid bbox)
                if bbox_result is not None:
                    dataset["annotations"].append({
                        "image_id": frame_id,
                        "gt_id": actor.id,
                        "type_id": info["type_id"],
                        "is_target": info["is_target"],
                        "color": info["color"],
                        "world_position": {
                            "x": actor_transform.location.x,
                            "y": actor_transform.location.y,
                            "z": actor_transform.location.z,
                            "yaw": actor_transform.rotation.yaw
                        },
                        "bbox_xyxy": bbox_result["bbox_xyxy"],
                        "bbox_xywh": bbox_result["bbox_xywh"]
                    })

            if (frame_idx + 1) % 100 == 0:
                print(f"[INFO] Scenario {scenario_idx}: Captured {frame_idx + 1}/{num_frames} frames")

        # Save dataset JSON
        gt_path = os.path.join(scenario_dir, "gt.json")
        with open(gt_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"[INFO] Scenario {scenario_idx}: Saved {len(dataset['images'])} images, {len(dataset['annotations'])} annotations")

        # Create video
        print(f"[INFO] Scenario {scenario_idx}: Creating video...")
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))

        if len(image_files) > 0:
            first_img = cv2.imread(image_files[0])
            height, width, _ = first_img.shape

            video_path = os.path.join(video_dir, f"scenario_{scenario_idx:03d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 10, (width, height))

            for img_file in image_files:
                frame = cv2.imread(img_file)
                video_writer.write(frame)

            video_writer.release()
            print(f"[INFO] Scenario {scenario_idx}: Video saved to {video_path}")

        return dataset["meta"]

    finally:
        # Cleanup actors for this scenario
        if camera is not None:
            camera.stop()

        for actor in spawned_actors:
            if actor is not None and actor.is_alive:
                actor.destroy()

        print(f"[INFO] Scenario {scenario_idx}: Cleaned up {len(spawned_actors)} actors")


def _setup_sync_mode(world, client, tm_port):
    """Enable synchronous mode on world and traffic manager. Returns (original_settings, tm_port)."""
    original_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  # 10 FPS
    world.apply_settings(settings)
    print("[INFO] Synchronous mode ENABLED (fixed_delta=0.1s, 10 FPS)")

    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    resolved_port = tm.get_port()
    print(f"[INFO] TrafficManager synchronous mode ENABLED (port {resolved_port})")

    return original_settings, resolved_port


def main():
    parser = argparse.ArgumentParser(description="CARLA Red Sedan Multi-Scenario Dataset Generator")
    parser.add_argument("--host", default="10.52.141.10", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--out_dir", default="dataset/carla_eval", help="Output directory")
    parser.add_argument("--num_frames", type=int, default=1000, help="Number of frames per scenario")
    parser.add_argument("--num_scenarios", type=int, default=1, help="Number of scenarios to generate")
    parser.add_argument("--num_distractors", type=int, default=10, help="Number of distractor vehicles")
    parser.add_argument("--seed", type=int, default=13, help="Base random seed for determinism")
    parser.add_argument("--tm_port", type=int, default=8000, help="Traffic manager port (change if 8000 is busy)")
    parser.add_argument("--follow_mode", action="store_true", help="Camera follows target (drone mode)")
    parser.add_argument("--eval_mode", action="store_true",
                        help="Run all 17 evaluation scenarios from EVAL_SCENARIO_CONFIGS")
    parser.add_argument("--resume_from", type=int, default=1,
                        help="Resume eval_mode from this scenario_id (skips earlier ones)")
    args = parser.parse_args()

    # Set global random seed for determinism
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create base output directory
    os.makedirs(args.out_dir, exist_ok=True)

    original_settings = None

    try:
        # =================================================================
        # CONNECT TO CARLA
        # =================================================================
        print(f"[INFO] Connecting to CARLA at {args.host}:{args.port}...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)

        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        map_name = world.get_map().name

        print(f"[INFO] Connected. Map: {map_name}, {len(spawn_points)} spawn points")

        # =================================================================
        # SETUP SYNCHRONOUS MODE
        # =================================================================
        original_settings, tm_port = _setup_sync_mode(world, client, args.tm_port)

        if args.eval_mode:
            # =============================================================
            # EVAL MODE: Run all 17 evaluation scenarios
            # =============================================================
            eval_out_dir = os.path.join(args.out_dir, "eval_scenarios")
            os.makedirs(eval_out_dir, exist_ok=True)

            # Group scenarios by map to minimize load_world calls
            from collections import defaultdict
            map_groups = defaultdict(list)
            for cfg in EVAL_SCENARIO_CONFIGS:
                map_groups[cfg["map"]].append(cfg)

            # Process current map first, then others
            current_map_short = map_name.split("/")[-1]  # Handle "Carla/Maps/Town10HD_Opt" format
            ordered_maps = []
            for m in map_groups:
                if m in current_map_short or current_map_short in m:
                    ordered_maps.insert(0, m)
                else:
                    ordered_maps.append(m)

            all_scenarios_meta = []
            total_scenarios = len(EVAL_SCENARIO_CONFIGS)
            completed = 0

            for target_map in ordered_maps:
                configs = map_groups[target_map]

                # Load map if needed
                current_map = world.get_map().name
                if target_map not in current_map:
                    print(f"\n[INFO] Loading map: {target_map}...")
                    client.load_world(target_map)
                    world = client.get_world()
                    bp_lib = world.get_blueprint_library()
                    spawn_points = world.get_map().get_spawn_points()
                    print(f"[INFO] Map loaded: {world.get_map().name}, {len(spawn_points)} spawn points")

                    # Re-apply sync mode after map load
                    original_settings, tm_port = _setup_sync_mode(world, client, args.tm_port)

                for cfg in configs:
                    completed += 1
                    scenario_name = cfg["name"]

                    # Skip scenarios before resume_from
                    if cfg["scenario_id"] < args.resume_from:
                        print(f"[INFO] Skipping scenario {cfg['scenario_id']}: {scenario_name} (resume_from={args.resume_from})")
                        continue

                    # Skip if gt.json already exists (already completed)
                    existing_gt = os.path.join(eval_out_dir, scenario_name, "gt.json")
                    if os.path.exists(existing_gt):
                        print(f"[INFO] Skipping scenario {cfg['scenario_id']}: {scenario_name} (gt.json exists)")
                        continue

                    print(f"\n{'='*60}")
                    print(f"[INFO] EVAL SCENARIO {completed}/{total_scenarios}: {scenario_name}")
                    print(f"[INFO] {cfg['description']}")
                    print(f"{'='*60}")

                    # Select camera spawn point deterministically from scenario seed
                    random.seed(cfg["seed"])
                    spawn_points_copy = list(spawn_points)
                    random.shuffle(spawn_points_copy)
                    camera_spawn = spawn_points_copy[0]

                    # Scenario output directory
                    scenario_dir = os.path.join(eval_out_dir, scenario_name)
                    os.makedirs(scenario_dir, exist_ok=True)

                    scenario_meta = run_scenario(
                        client=client,
                        world=world,
                        bp_lib=bp_lib,
                        spawn_points=spawn_points,
                        scenario_idx=cfg["scenario_id"],
                        scenario_dir=scenario_dir,
                        num_frames=cfg["num_frames"],
                        num_distractors=cfg["num_distractors"],
                        seed=cfg["seed"],
                        camera_spawn_point=camera_spawn,
                        tm_port=tm_port,
                        camera_mode=cfg["camera_mode"],
                        weather_params=cfg["weather"],
                        num_targets=cfg["num_targets"],
                        distractor_colors=cfg["distractor_colors"],
                        distractor_blueprints=cfg["distractor_blueprints"],
                        distractor_use_target_color=cfg["distractor_use_target_color"],
                        camera_config=cfg["camera"],
                        scenario_name=scenario_name,
                        scenario_description=cfg["description"]
                    )

                    all_scenarios_meta.append(scenario_meta)

                    # Brief pause between scenarios
                    for _ in range(10):
                        world.tick()

            # Save master index for eval scenarios
            master_index = {
                "mode": "eval",
                "num_scenarios": total_scenarios,
                "scenarios": all_scenarios_meta
            }

            master_path = os.path.join(eval_out_dir, "master_index.json")
            with open(master_path, 'w') as f:
                json.dump(master_index, f, indent=2)

            print(f"\n{'='*60}")
            print(f"[INFO] ALL {total_scenarios} EVAL SCENARIOS COMPLETE!")
            print(f"[INFO] Master index saved to {master_path}")
            print(f"{'='*60}")

        else:
            # =============================================================
            # STANDARD MODE: Original behavior (unchanged)
            # =============================================================
            if len(spawn_points) < args.num_scenarios:
                print(f"[WARN] Only {len(spawn_points)} spawn points, reducing scenarios")
                args.num_scenarios = len(spawn_points)

            # Select random spawn points for camera positions
            spawn_points_copy = list(spawn_points)
            random.shuffle(spawn_points_copy)
            camera_spawn_points = spawn_points_copy[:args.num_scenarios]

            print(f"[INFO] Will generate {args.num_scenarios} scenarios with {args.num_frames} frames each")

            # Track all scenario metadata
            all_scenarios_meta = []

            for scenario_idx in range(1, args.num_scenarios + 1):
                print(f"\n{'='*60}")
                print(f"[INFO] STARTING SCENARIO {scenario_idx}/{args.num_scenarios}")
                print(f"{'='*60}")

                # Get camera spawn point for this scenario
                camera_spawn = camera_spawn_points[scenario_idx - 1]

                # Create scenario-specific output directory
                scenario_dir = os.path.join(args.out_dir, f"scenario_{scenario_idx:03d}")
                os.makedirs(scenario_dir, exist_ok=True)

                # Use different seed for each scenario (deterministic but varied)
                scenario_seed = args.seed + scenario_idx * 1000

                # Run the scenario (camera will be pointed at target after spawning)
                scenario_meta = run_scenario(
                    client=client,
                    world=world,
                    bp_lib=bp_lib,
                    spawn_points=spawn_points,
                    scenario_idx=scenario_idx,
                    scenario_dir=scenario_dir,
                    num_frames=args.num_frames,
                    num_distractors=args.num_distractors,
                    seed=scenario_seed,
                    camera_spawn_point=camera_spawn,
                    tm_port=tm_port,
                    follow_mode=args.follow_mode
                )

                all_scenarios_meta.append(scenario_meta)

                # Brief pause between scenarios
                for _ in range(10):
                    world.tick()

            # Save master index
            master_index = {
                "num_scenarios": args.num_scenarios,
                "frames_per_scenario": args.num_frames,
                "base_seed": args.seed,
                "map": map_name,
                "scenarios": all_scenarios_meta
            }

            master_path = os.path.join(args.out_dir, "master_index.json")
            with open(master_path, 'w') as f:
                json.dump(master_index, f, indent=2)

            print(f"\n{'='*60}")
            print(f"[INFO] ALL SCENARIOS COMPLETE!")
            print(f"[INFO] Master index saved to {master_path}")
            print(f"{'='*60}")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # =================================================================
        # CLEANUP - Restore async mode
        # =================================================================
        print("[INFO] Final cleanup...")

        if original_settings is not None:
            try:
                world = client.get_world()
                original_settings.synchronous_mode = False
                original_settings.fixed_delta_seconds = None
                world.apply_settings(original_settings)

                tm = client.get_trafficmanager(args.tm_port)
                tm.set_synchronous_mode(False)

                print("[INFO] Restored async mode")
            except:
                pass

        print("[INFO] Done!")


if __name__ == "__main__":
    main()
