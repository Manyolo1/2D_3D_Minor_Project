# Data Setup Guide

## Required Folder Structure

The code expects data organized in one of these locations:
- `Data/benchmark/` - For ModelNet40 or similar benchmark datasets
- `Data/nonbenchmark/` - For custom datasets

## Data Format Options

### Option 1: Using ZIP files (--zip 1 --from-scratch 1)

Place ZIP files directly in `Data/nonbenchmark/`:
```
Data/nonbenchmark/
  ├── chair.zip
  ├── table.zip
  └── sofa.zip
```

Each ZIP should contain a folder with naming pattern `<classname>_depth_rgb/` or `<classname>_depth_float/`:
- For `--raw-data-type int`: folders named `*_depth_rgb/` containing `.png` depth images
- For `--raw-data-type float`: folders named `*_depth_float/` containing `.txt` depth files

### Option 2: Using extracted folders (--zip 0 --from-scratch 1)

Place extracted folders directly in `Data/nonbenchmark/`:
```
Data/nonbenchmark/
  ├── chair_depth_rgb/
  │   ├── chair_model1_Cam_0.png
  │   ├── chair_model1_Cam_1.png
  │   ├── ...
  │   └── chair_model1_Cam_19.png
  ├── table_depth_rgb/
  └── sofa_depth_rgb/
```

## File Naming Convention

Multi-view depth files must follow one of these patterns:
- `<modelname>_Cam_<viewpoint>.{png|txt}` - e.g., `chair_001_Cam_0.png` ... `chair_001_Cam_19.png`
- `<modelname>_<viewpoint>.{png|txt}` - e.g., `chair_001_0.png` ... `chair_001_19.png`

The code auto-detects the number of viewpoints from the first object.

## Depth File Formats

### PNG format (--raw-data-type int)
- Grayscale images where pixel intensity = depth value (0-255 or 0-65535)
- More compact, commonly used in public datasets

### TXT format (--raw-data-type float)
- Space-separated floating-point values arranged in rows
- Each row = one line of the depth map
- More precise depth measurements
- Example 3x3 depth map:
  ```
  0.5 0.6 0.7
  0.5 0.65 0.75
  0.6 0.7 0.8
  ```

## Quick Start

1. Place your data in `Data/nonbenchmark/` using one of the formats above
2. Run preprocessing:
   ```powershell
   python Main_Py\main_py.py --from-scratch 1 --zip 1 --raw-data-type int --conditional 1 --model-dir-name results_run1
   ```
3. After processing, you'll see:
   ```
   Data/nonbenchmark/Datasets/
     ├── train/data_0.data
     ├── validation/data_0.data
     └── test/data_0.data
   ```
4. Then run training:
   ```powershell
   python Main_Py\main_py.py --train 1 --conditional 1 --single-vp-net 1 --benchmark 0 --model-dir-name results_run1
   ```

## Using Your Existing Files

You have these zips in your project root:
- `renderDepth_new.zip`
- `zip-depth_render_reconstruction.zip`
- `zip-ExtraData.zip`

Move or copy the relevant class-specific depth zips to `Data/nonbenchmark/` or extract them there.
