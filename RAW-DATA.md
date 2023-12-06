# LaMAR Raw Data

We release the raw data in 3 Capture directories, where each session corresponds to a recording of one of the devices. This data includes additional sensor modalities that were not used for the benchmark, such as depth, IMU, or GPS. We also release the 3D laser scans (as point cloud and mesh) obtained by the NavVis scanner and used to estimate the ground truth poses.

Each scene directory contains files `metadata_{phone,hololens}.json` that indicate, for each recording, its duration (in seconds) as well as whether some sensor modalities are missing (GPS and depth for phones).

## NavVis sessions

We imported the output of the NavVis processing software into Capture with `scantools.run_navvis_to_capture`. The resulting data includes:
- Images captures by the 4 on-device cameras, undistorted.
- A point cloud `raw_data/pointcloud.ply` in PLY format, aggregated from the measurements of the VLP-16 LiDAR sensors. The processing software removes dynamic objects and downsamples the points with a voxel size of 1cm.
- A mesh that we computed from the point cloud. We include high-resolution and simplified variants as `proc/meshes/{mesh,mesh_simplified}.ply`, in PLY format
- The rigid transformation between the NavVis session and the coordinate frame of the map in the benchmark data, as `proc/alignment_global.txt`. This allows rendering depth maps or images for any mapping or validation image, as in [`scantools.run_sequence_rerendering`](./scantools/run_sequence_rerendering.py):
```python
from scantools.proc.rendering import Renderer
from scantools.utils.io import read_mesh

w_t_camera: scantools.capture.Pose
camera: scantools.capture.Camera
session_navvis: scantools.capture.Session

# Transform the camera pose from world to NavVis coordinate frame
w_t_navvis = session_navvis.proc.alignment_global.get_abs_pose('pose_graph_optimized')
navvis_t_camera = w_t_navvis.inv() * w_t_camera

# Load the mesh with key "mesh" or "mesh_simplified"
mesh = read_mesh(capture.proc_path(session_navvis.id) / session_navvis.proc.meshes['mesh'])
renderer = Renderer(mesh)

# Render!
rgb, depth, normals = renderer.render_from_capture(navvis_t_camera, camera, with_normals=True)
```

## Phone sessions

We imported the output of the ScanCapture app into Capture by calling `scantools.run_phone_to_capture` with `downsample_framerate=None` (full framerate, approximately 10 FPS) and `split_sequence_on_failure=False` (entire sequences). In addition to the files present in the original release and described in the [capture format overview](./CAPTURE.md), each session in the full release contains several new files that are not part of the Capture API:

- `accelerometer.txt`: Raw high-frequency accelerometer measurements.
- `gyroscope.txt`: Raw high-frequency gyroscope measurements.
- `magnetometer.txt`: Raw high-frequency magnetometer measurements.
- `fused_imu.txt`: The data fused and processed by [CoreMotion](https://developer.apple.com/documentation/coremotion). It includes the device-specific acceleration (without gravity), the unbiased rotation rate, the magnetic field vector, the gravity vector, and the heading angle with respect to the North.
- `location.txt`: The geographic location reported by [CoreLocation](https://developer.apple.com/documentation/corelocation) in the highest accuracy mode.
- `depths.txt`: A list of the [depth maps captured by the on-device LiDAR scanner](https://developer.apple.com/documentation/arkit/ardepthdata), with associated confidence maps of per-pixel values within three levels (high, medium, low).

Please refer to the [ScanCapture iOS app](apps/ScanCapture_iOS) for more details on the content of these files.

## HoloLens sessions

All data was captured and processed with an experimental tool running on HoloLens 2 devices. In addition to the files present in the original release and described in the [capture format overview](./CAPTURE.md), each session in the full release contains several new files that are not part of the Capture API:

- `gravity.txt`: this file contains gravity estimates for each timestamp in the
  format "down vector in keyframe coordinate system / in rig frame".
- `components.txt`: the on-device SLAM system sometimes yields multiple sub-maps
  for a single recording. This file contains a sub-map component index for each
  timestamp. _Note that the estimates of odometry in `trajectories.txt` and
  gravity in `gravity.txt` are only consistent within a sub-map._
- `imu.txt`: contains the raw measurements from all the IMUs present on the
  device (accelerometer, gyroscope, magnetometer) in format x, y, z. _Note that
  there is no magnetometer external calibration and we do not plan to release
  the internal calibrations of IMUs._
- `info.txt`: contains summary information about the device and the data
  post-processing.
- `depth_LUT.npy`: this file contains a look-up table in the [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
  which can be used for undistorting the released depthmaps. The LUT is an array
  with size HxWx2 mapping (row, column) coordinates in the image to (x, y)
  coordinates on the normalized image plane. An example on how to use this LUT
  in Python can be found below.

### Depth map demo script

This short python snippet below shows how to load and visualize the depthmaps.
The only requirements are numpy, matplotlib, and opencv-python.

```python
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2

DEPTHMAP_SCALING_FACTOR = 1000  # mm to m

# Define the paths.
session_path = Path("path/to/session")
lut_path = session_path / "depth_LUT.npy"
lut = np.load(lut_path)

# Read the depthmap - note that this is in fact the length / norm of each ray, and not the z coordinate .
depthmap_paths = sorted((session_path / "raw_data" / "depths").glob("*.png"))
depthmap_path = depthmap_paths[0]
print("Processing depthmap at", depthmap_path)
# Alternatively, you can use scantools.utils.io.read_depth(depthmap_path).
depthmap = np.array(
  cv2.imread(str(depthmap_path), cv2.IMREAD_ANYDEPTH),
  dtype=float
)
depthmap /= DEPTHMAP_SCALING_FACTOR
is_valid = np.logical_not(depthmap == 0)
print(
  "Depthmap stats min/max (in meters)",
  np.min(depthmap[is_valid]),
  np.max(depthmap[is_valid])
)
## Plot raw depthmap.
plt.imshow(depthmap, cmap="inferno")
plt.colorbar()
plt.axis('off')
plt.show()

## Backproject to 3D.
num_valid_pixels = np.sum(is_valid)
valid_depths = depthmap[is_valid]
normalized_pixels = lut[is_valid]
normalized_pixels_hom = np.hstack([normalized_pixels, np.ones([num_valid_pixels, 1])])
points_3d = normalized_pixels_hom * (
  valid_depths[:, np.newaxis] / np.linalg.norm(normalized_pixels_hom, axis=-1, keepdims=True)
)
## Plot the 3D pointcloud.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], marker="X", s=1, c=valid_depths, cmap="inferno")
plt.show()
```
