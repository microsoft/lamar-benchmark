# Capture data format

The format is inspired by the [Kapture format from Naver Labs](https://github.com/naver/kapture/blob/main/kapture_format.adoc) but improves on the following points:

- it can handle multiple sessions per location
- it supports other assets: meshes, depth renderings, transformations between sessions, etc.
- image matching is delegated to [hloc](https://github.com/cvg/Hierarchical-Localization)
- the API is optimized for each of use and a smooth learning curve

## Overview

Here is an example of file structure for a capture at `location1`:

```
location1/                                  # a Capture directory
├── sessions/                               # a collection of Sessions 
│   ├── navvis1/                            # NavVis Session #1
│   │   ├── sensors.txt                     # list of all sensors with specs
│   │   ├── rigs.txt                        # rigid geometric relationship between sensors
│   │   ├── trajectories.txt                # pose for each (timestamp, sensor)
│   │   ├── images.txt                      # list of images with their paths
│   │   ├── pointclouds.txt                 # list of point clouds with their paths
│   │   ├── raw_data/                       # root path of images, point clouds, etc.
│   │   │   ├── images_undistorted/
│   │   │   └── pointcloud.ply
│   │   └── proc/                           # root path of processed assets
│   │       ├── meshes/                     # a collections of meshes
│   │       ├── depth_renderings.txt        # a list of rendered depth maps, one per image
│   │       ├── depth_renderings/           # root path for the depth maps
│   │       ├── alignment_global.txt        # global transforms between sessions
│   │       ├── alignment_trajectories.txt  # transform of each pose to a global reference
│   │       └── overlaps.h5                 # overlap matrix from this session to others
│   ├── hololens1/
│   │   ├── sensors.txt
│   │   ├── rigs.txt
│   │   ├── trajectories.txt
│   │   ├── images.txt
│   │   ├── depths.txt                      # list of depth maps with their paths
│   │   ├── bluetooth.txt                   # list of bluetooth measurements
│   │   ├── wifi.txt                        # list of wifi measurements
│   │   ├── raw_data/
│   │   │   ├── images/
│   │   │   └── depths/
│   │   └── proc/
│   │       └── alignment/
│   └── phone1/
│       └── ...
├── registration/                           # the data generated during alignment
│   ├── navvis2/
│   │   └── navvis1/                        # alignment of navvis2 w.r.t navvis1
│   │       └─ ...                          # intermediate data for matching/registration
│   └── hololens1/
│   │   └── navvis1/
│   └── phone1/
│       └── navvis2/
└── visualization/                          # root path of visualization dumps
    └─ ...                                  # all the data dumped during processing (TBD)
```

A Python interface that mirrors this file structure is available in `scantools.capture`. It can be created from an existing Capture file tree, but also create such tree from scratch and update it as it is modified.

## File formats

**Text files `.txt`** contain utf-8 lines that are comma separated values (CSV) or comments starting with `#`.

**Point clouds and meshes** are in `.ply` PLY format, preferably in binary with float values.

**Images** are in PNG or JPEG formats.

**Depth maps** are 16-bit PNG images scaled by `scantools.utils.io.DEPTH_SCALE`.

**Image features and matches** are handled by [hloc](https://github.com/cvg/Hierarchical-Localization) (currently HDF5 files).

## Details

### 1. Session data

- `sensors.txt`, `rigs.txt`, `trajectories.txt` follow [the Kapture format](https://github.com/naver/kapture/blob/main/kapture_format.adoc#2--sensors). However, the pose convention is reverted: `rigs.txt` contains camera-to-rig transformations and `trajectories.txt` contains sensor-to-world transformations.
- `images.txt`, `pointclouds.txt`, `depths.txt`, `wifi.txt`, and `bt.txt` follow the specifications of their corresponding `records_*.txt` in Kapture.

### 2. Processed files

- `proc/meshes/` contains all the meshes generated from the point cloud(s).
- `proc/depth_renderings.txt` is a list of depth maps and their relative paths in `proc/depth_renderings/`
- `proc/alignment_global.txt` is a list of global transforms w.r.t. other sessions with additional info (e.g. error statistics)
- `proc/alignment_trajectories.txt` is a list of transforms w.r.t an absolute reference, following the format of `trajectories.txt`
- `qrcodes/qr_map.txt` contains all detected QR codes.
  `qrcodes/qr_map_filtered_by_area.txt` is a list of QR codes filtered by area.
  If there are multiple measurements for one QR code, we select the one with
  largest area in the image.

### 3. Registration

The data generated for the pairwise and global alignment of all sessions is stored in `capture/registration/`. This includes images features and matches, registration confidences, errors statistics, and some visualizations. Each folder `registration/session1/session2/` contains the alignment of `session1` w.r.t `session2`.

### 4. Visualization

This directory contains images, meshes, and other assets useful to verify the ground truthing process.
