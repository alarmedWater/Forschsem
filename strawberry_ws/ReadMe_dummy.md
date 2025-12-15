# Dummy pipeline documentation

This “dummy pipeline” is a ROS 2 (rclpy) demo system that **replays stored RGB + depth images from disk**, runs **YOLOv8 instance segmentation**, uses the predicted instance masks to **mask the depth image**, and then computes **simple 3D features + point clouds** per detected strawberry instance. Additionally, it publishes a visualization where **one selected instance is highlighted**.

It is called “dummy” because it does not use a live camera driver yet; instead it simulates a camera by reading image files from folders.

## Overview:
(Folders on disk)
  RGB PNG/JPGs                         Depth PNGs (16UC1 mm)
  rgb_dir/color_*.png                  depth_dir/depth_*.png
        |                                      |
        v                                      v
  camera_folder_node (strawberry_camera/camera_folder_node.py)
        | publishes:
        |   /camera/color/image_raw                     (rgb8)
        |   /camera/aligned_depth_to_color/image_raw    (16UC1, mm)
        |   /camera/color/camera_info                   (intrinsics K)
        |
        |-------------------------------> seg_ultra_node (strawberry_segmentation/seg_ultra_node.py)
        |                                   subscribes: /camera/color/image_raw
        |                                   publishes:
        |                                      /seg/label_image      (mono16 instance IDs: 0=bg, 1..N)
        |                                      /seg/label_image_vis  (mono8 debug 0/255)
        |                                      /seg/overlay          (rgb8 YOLO overlay, optional)
        |
        v
  depth_mask_node (strawberry_segmentation/depth_mask_node.py)
        subscribes (sync):
          /camera/aligned_depth_to_color/image_raw  +  /seg/label_image
        publishes:
          /seg/depth_masked   (same encoding as depth input; background -> 0)

(camera intrinsics)
  /camera/color/camera_info -------------------------------> strawberry_features_node
                                                            (strawberry_segmentation/strawberry_features_node.py)
                                                            subscribes:
                                                              (sync) /seg/depth_masked + /seg/label_image
                                                              (async) /camera/color/camera_info
                                                            computes per instance:
                                                              centroid + AABB extent + box volume
                                                            publishes (optional):
                                                              /seg/strawberry_cloud            (PointCloud2 all instances)
                                                              /seg/strawberry_cloud_selected   (PointCloud2 selected instance)

(selected instance visualization)
  /camera/color/image_raw  +  /seg/label_image  -----> strawberry_selected_overlay_node
                                                       (strawberry_segmentation/strawberry_selected_overlay_node.py)
                                                       publishes:
                                                         /seg/selected_overlay (rgb8, background darkened, bbox)
camera_folder_node
  - reads next RGB frame -> converts BGR->RGB -> publishes /camera/color/image_raw
  - reads next depth frame -> ensures uint16 (mm) -> resize to RGB size -> publishes /camera/aligned_depth_to_color/image_raw
  - publishes CameraInfo K=[fx,0,cx; 0,fy,cy; 0,0,1] on /camera/color/camera_info

seg_ultra_node (YOLOv8)
  - receives RGB frame
  - runs YOLO segmentation -> gets masks
  - filters tiny masks (min_mask_area_px)
  - writes instance-id image (mono16): each pixel stores instance number (1..N), background=0
  - publishes:
      /seg/label_image (mono16)
      /seg/overlay (optional)
      /seg/label_image_vis (debug)

depth_mask_node
  - synchronizes depth + label (ApproximateTimeSynchronizer)
  - creates mask = (label > 0)
  - masked_depth = depth, but background pixels set to 0
  - publishes /seg/depth_masked

strawberry_features_node
  - waits for CameraInfo (fx,fy,cx,cy)
  - synchronizes masked depth + label
  - backprojects pixels to 3D using pinhole model:
      z = depth (m), x=(u-cx)*z/fx, y=(v-cy)*z/fy
  - per instance:
      collects 3D points -> filters if too few points
      computes centroid + AABB extent + AABB volume
  - optionally publishes point clouds:
      /seg/strawberry_cloud (all points)
      /seg/strawberry_cloud_selected (selected instance only)

strawberry_selected_overlay_node
  - synchronizes RGB + label
  - selects one instance id (parameter selected_instance_id)
  - darkens background, keeps selected instance bright
  - draws bbox + id text
  - publishes /seg/selected_overlay


## High-level dataflow

**Core idea:** Everything communicates via ROS topics.

1) `camera_folder_node` publishes:
- RGB image
- aligned depth image
- CameraInfo (intrinsics)

2) `seg_ultra_node` subscribes to RGB, publishes:
- instance label image (mono16 IDs)

3) `depth_mask_node` subscribes to depth + label, publishes:
- masked depth image

4) `strawberry_features_node` subscribes to masked depth + label + CameraInfo, publishes:
- point clouds and logs per-instance 3D features

5) `strawberry_selected_overlay_node` subscribes to RGB + label, publishes:
- overlay image showing only the selected instance highlighted

### Topic-level pipeline diagram (logical)

```
(color images from disk)
  /camera/color/image_raw  ------------------>  seg_ultra_node
            |                                      |
            |                                      v
            |                                /seg/label_image  (mono16 instance IDs)
            |
            v
(depth images from disk)
  /camera/aligned_depth_to_color/image_raw  -----> depth_mask_node -----> /seg/depth_masked
                                                     ^
                                                     |
                                            /seg/label_image

(camera intrinsics)
  /camera/color/camera_info -------------------------------> strawberry_features_node
                                                            ^              ^
                                                            |              |
                                                    /seg/depth_masked  /seg/label_image

(selected instance visualization)
  /camera/color/image_raw + /seg/label_image  -----> strawberry_selected_overlay_node
                                                 -> /seg/selected_overlay
```

---

## Nodes and their responsibilities

### 1) `strawberry_camera/camera_folder_node.py` (`camera_folder`)
**Role:** “Fake camera” that plays back frames from folders at a configurable FPS.

**Publishes**
- `/camera/color/image_raw` (`sensor_msgs/Image`, `rgb8`)
- `/camera/aligned_depth_to_color/image_raw` (`sensor_msgs/Image`, `16UC1`, millimeters)
- `/camera/color/camera_info` (`sensor_msgs/CameraInfo`)

**Important behavior**
- RGB is loaded with OpenCV as BGR and converted to **RGB** before publishing.
- Depth is loaded as-is (`cv2.IMREAD_UNCHANGED`). If not `uint16`, it is converted to mm.
- Depth is resized to match RGB if needed (nearest-neighbor), assuming depth is “aligned to color”.
- Camera intrinsics are currently passed via parameters (`fx, fy, cx, cy`) with placeholder defaults.

**Key parameters**
- `rgb_dir`, `rgb_pattern`
- `depth_dir`, `depth_pattern`
- `fps`, `loop`, `publish_depth`
- intrinsics: `fx, fy, cx, cy`
- frame ids: `frame_color`, `frame_depth`

---

### 2) `strawberry_segmentation/seg_ultra_node.py` (`strawberry_seg_ultra`)
**Role:** Runs **Ultralytics YOLOv8 segmentation** on incoming RGB images.

**Subscribes**
- `/camera/color/image_raw` (`sensor_msgs/Image`, `rgb8`) by default (`topic_in`)

**Publishes**
- `/seg/label_image` (`sensor_msgs/Image`, `mono16`): **instance ID map**
  - `0` = background
  - `1..N` = per-instance IDs assigned per frame
- `/seg/label_image_vis` (`sensor_msgs/Image`, `mono8`): debug mask (0/255)
- `/seg/overlay` (`sensor_msgs/Image`, `rgb8`): YOLO overlay visualization (optional)

**Important behavior**
- Converts incoming RGB → BGR for Ultralytics.
- Builds a **label image** by taking YOLO masks, optionally filtering small masks (`min_mask_area_px`).
- Instance IDs are assigned per frame (not persistent across frames).

**Key parameters**
- `model_path` (empty → uses package share `.../models/best.pt`)
- `device` (`auto`, `cpu`, `cuda:0`)
- `imgsz`, `conf_thres`, `iou_thres`, `max_det`
- `min_mask_area_px`
- `publish_overlay`, `profile`

---

### 3) `strawberry_segmentation/depth_mask_node.py` (`strawberry_depth_mask`)
**Role:** Combines depth + instance labels to keep only strawberry pixels in depth.

**Subscribes (synchronized)**
- depth: `/camera/aligned_depth_to_color/image_raw`
- labels: `/seg/label_image`

**Publishes**
- `/seg/depth_masked` (`sensor_msgs/Image`, same encoding as input depth)

**Important behavior**
- Uses approximate time synchronization (slop ~ 50 ms).
- Creates a mask: `label > 0`
- Background depth is set to `0` by default (`zero_background=True`)

**Key parameters**
- `depth_topic`, `label_topic`, `output_topic`
- `zero_background`, `profile`

---

### 4) `strawberry_segmentation/strawberry_features_node.py` (`strawberry_features`)
**Role:** Computes per-instance **3D point clouds** and simple **3D features**.

**Subscribes**
- `/seg/depth_masked` and `/seg/label_image` (synchronized)
- `/camera/color/camera_info` (intrinsics, not synchronized)

**Publishes**
- `/seg/strawberry_cloud` (`sensor_msgs/PointCloud2`): all instances combined (optional)
- `/seg/strawberry_cloud_selected` (`sensor_msgs/PointCloud2`): one selected instance (optional)

**Computed features (logged)**
Per instance (for instances with enough points):
- number of points `N`
- centroid `(x, y, z)` in meters (camera frame)
- axis-aligned bounding box extent `(dx, dy, dz)`
- AABB volume (`dx*dy*dz`)

**Important behavior**
- Requires `CameraInfo` first (fx, fy, cx, cy).
- Converts depth:
  - `uint16` assumed mm → meters
  - otherwise assumed already meters
- Back-projection using pinhole model:
  - `x = (u - cx) * z / fx`
  - `y = (v - cy) * z / fy`
  - `z = depth`
- Optional downsampling (`downsample_step`) for speed.

**Key parameters**
- `downsample_step`, `min_points`
- `selected_instance_id`
- `publish_all_cloud`, `publish_selected_cloud`
- metadata: `plant_id`, `view_id` (currently only used in logs)

---

### 5) `strawberry_segmentation/strawberry_selected_overlay_node.py` (`strawberry_selected_overlay`)
**Role:** Visualizes only one selected strawberry instance on the RGB image.

**Subscribes (synchronized)**
- RGB: `/camera/color/image_raw`
- labels: `/seg/label_image`

**Publishes**
- `/seg/selected_overlay` (`sensor_msgs/Image`, `rgb8`)

**Important behavior**
- Reads `selected_instance_id` at runtime (you can change it via `ros2 param set ...`)
- Darkens background and keeps selected instance bright
- Draws bounding box + instance ID text
- If the instance is not present (or too small), it publishes the original RGB frame.

**Key parameters**
- `selected_instance_id`, `min_pixels`, `darken_factor`
- `image_topic`, `label_topic`, `output_topic`

---

## Launch file: `dummy_system.launch.py`

The launch file starts the whole dummy pipeline:
- `camera_folder`
- `strawberry_seg_ultra`
- `strawberry_depth_mask`
- `strawberry_features`
- `strawberry_selected_overlay`

It also defines launch arguments:
- `rgb_dir`
- `depth_dir`
- `fps`
- `loop`
- `model_path`

---

## How to run

### Build
From your workspace:
```bash
cd ~/Forschsemrep/strawberry_ws
colcon build --symlink-install
source install/setup.bash
```

### Launch the dummy pipeline
```bash
ros2 launch strawberry_bringup dummy_system.launch.py
```

Override inputs if needed:
```bash
ros2 launch strawberry_bringup dummy_system.launch.py \
  rgb_dir:=/path/to/rgb \
  depth_dir:=/path/to/depth \
  fps:=5.0 \
  loop:=true \
  model_path:=/path/to/best.pt
```

---

## Debugging and visualization

### See images
- RGB: `/camera/color/image_raw`
- YOLO overlay: `/seg/overlay`
- label visualization: `/seg/label_image_vis`
- selected overlay: `/seg/selected_overlay`

Example:
```bash
ros2 run rqt_image_view rqt_image_view
```

### Inspect topics and rates
```bash
ros2 topic list
ros2 topic hz /camera/color/image_raw
ros2 topic echo /camera/color/camera_info --once
```

### View point clouds (RViz2)
- `/seg/strawberry_cloud`
- `/seg/strawberry_cloud_selected`

In RViz2:
```bash
ros2 run rviz2 rviz2
```
- Add → “PointCloud2”
- Set fixed frame to your camera frame id (from CameraInfo header)

---

## Runtime controls (useful knobs)

Change the selected instance being highlighted / published:
```bash
ros2 param set /strawberry_features selected_instance_id 2
ros2 param set /strawberry_selected_overlay selected_instance_id 2
```

Tune segmentation strictness:
```bash
ros2 param set /strawberry_seg_ultra conf_thres 0.7
ros2 param set /strawberry_seg_ultra min_mask_area_px 2000
```

---

## Current limitations (why it’s “dummy”)

- No real camera driver yet (images are replayed from disk).
- Depth and RGB alignment is assumed and enforced by resizing; real calibration/alignment is not handled here.
- Instance IDs from YOLO are **per-frame** only (not persistent across time).
- 3D features are simple (centroid + AABB), not full shape estimation.
