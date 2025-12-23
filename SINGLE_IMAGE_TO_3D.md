# Single Image to 3D Scene Reconstruction

Convert any 2D depth image or RGB image into a complete 3D mesh with a single command!

## Quick Start

### Basic Usage

```powershell
# Reconstruct 3D scene from a single depth image
python Main_Py\single_image_to_3d.py --input path\to\your\image.png

# Reconstruct from RGB image
python Main_Py\single_image_to_3d.py --input path\to\photo.jpg --is-depth 0
```

### Output

The system generates:
- ✓ **Multi-view depth maps** (20 viewpoints)
- ✓ **Multi-view silhouettes** (20 viewpoints)
- ✓ **3D point cloud** (`pointcloud_fused.ply`)
- ✓ **3D mesh with faces** (`mesh_fused.ply`)

All files are saved to `input_to_3d/<your_image_name>/`

## Examples

### Example 1: Test with existing data

```powershell
# Use a depth image from your test data
python Main_Py\single_image_to_3d.py --input Data\nonbenchmark\cube_depth_rgb\cube_depth_0.png --category 0

# Use a sphere depth image
python Main_Py\single_image_to_3d.py --input Data\nonbenchmark\sphere_depth_rgb\sphere_depth_0.png --category 1
```

### Example 2: Custom image

```powershell
python Main_Py\single_image_to_3d.py ^
    --input my_image.png ^
    --model-dir results_run1 ^
    --epoch 80 ^
    --category 0 ^
    --focal-length 300 ^
    --output-dir my_3d_output
```

### Example 3: RGB photo (converts to depth estimation)

```powershell
python Main_Py\single_image_to_3d.py ^
    --input photo.jpg ^
    --is-depth 0 ^
    --category 0
```

## View Your 3D Results

### Option 1: Windows 3D Viewer (Easiest)
```powershell
start input_to_3d\<your_image>\mesh_fused.ply
```

### Option 2: Python + Open3D
```powershell
python -c "import open3d as o3d; mesh = o3d.io.read_triangle_mesh('input_to_3d/<your_image>/mesh_fused.ply'); mesh.compute_vertex_normals(); o3d.visualization.draw_geometries([mesh])"
```

### Option 3: Online Viewer
Upload `mesh_fused.ply` to: https://3dviewer.net

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | *required* | Path to input 2D image (PNG/JPG) |
| `--model-dir` | `results_run1` | Directory with trained model |
| `--epoch` | `80` | Model checkpoint epoch to use |
| `--output-dir` | `input_to_3d/<name>` | Where to save output files |
| `--category` | `0` | Object category (0=cube, 1=sphere) |
| `--is-depth` | `1` | 1=depth map, 0=RGB image |
| `--focal-length` | `300` | Camera focal length (pixels) |
| `--skip-mesh` | *flag* | Only generate images, skip 3D mesh |

## How It Works

```
Input 2D Image
     ↓
[VAE Encoder] → Latent Space (40D vector)
     ↓
[VAE Decoder] → 20 Viewpoint Predictions (depth + silhouette)
     ↓
[Back-projection] → 3D Points from each viewpoint
     ↓
[Multi-view Fusion] → Merged 3D Point Cloud
     ↓
[Surface Reconstruction] → 3D Mesh with Faces
     ↓
Output: mesh_fused.ply
```

## Pipeline Steps

1. **Preprocessing**: Resizes input to 224×224, normalizes to [0,1]
2. **Encoding**: VAE encoder maps 2D image → 40D latent vector
3. **Decoding**: VAE decoder generates all 20 viewpoints from latent code
4. **Back-projection**: Each depth map → 3D points using pinhole camera model
5. **Fusion**: Merge overlapping points from all views (voxel downsampling)
6. **Meshing**: Alpha shape or convex hull to create surface with faces

## Troubleshooting

### "Checkpoint not found"
Make sure you've trained the model first:
```powershell
python Main_Py\main_py.py --train 1 --conditional 1 --single-vp-net 1 --benchmark 0 --model-dir-name results_run1 --max-epochs 80
```

### "Module not found"
Install required packages:
```powershell
pip install torch torchvision pillow numpy imageio open3d tqdm
```

### Mesh looks wrong
Try adjusting:
- `--category`: Use correct object type (0=cube, 1=sphere)
- `--focal-length`: Try 200-500 range
- Train longer (more epochs = better quality)

### Want higher quality?
- Train with more epochs: `--max-epochs 200`
- Use larger images: Modify `img_size` in code
- Add more training data

## Advanced: Batch Processing

Process multiple images:
```powershell
# Process all PNG files in a directory
Get-ChildItem .\my_images\*.png | ForEach-Object {
    python Main_Py\single_image_to_3d.py --input $_.FullName
}
```

## Next Steps

1. **Test the system**: Use a sample depth image from your data
2. **Try your own images**: Any depth map or photo works!
3. **View results**: Open the generated `.ply` mesh file
4. **Iterate**: Adjust category/focal-length for better results

## Technical Details

- **Model**: Variational Autoencoder (VAE) trained on multi-view depth data
- **Input**: Single 2D depth/RGB image (any size, auto-resized to 224×224)
- **Latent**: 40-dimensional continuous representation
- **Output**: 20 viewpoints × 2 channels (depth + silhouette)
- **3D**: Pinhole camera projection + multi-view fusion
- **Mesh**: Alpha shape (α=0.05) or convex hull fallback

## Citation

If you use this system, please cite your training data source and mention:
- PyTorch (model framework)
- Open3D (3D processing)
- VAE architecture for 3D reconstruction
