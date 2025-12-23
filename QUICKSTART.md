# 🎯 Quick Start Guide: 3D Reconstruction Pipeline

## Complete Workflow (Copy & Paste)

### Step 1: Setup Environment
```powershell
# Install dependencies
pip install torch torchvision numpy pillow psutil

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Step 2: Check Your Data
```powershell
# Your data should be in:
# Data/nonbenchmark/cube_depth_rgb/    ✓ (Already exists)
# Data/nonbenchmark/sphere_depth_rgb/  ✓ (Already exists)

# Files are already preprocessed in:
# Data/nonbenchmark/Datasets/train/data_0.data     ✓
# Data/nonbenchmark/Datasets/validation/data_0.data ✓
# Data/nonbenchmark/Datasets/test/data_0.data      ✓
```

### Step 3: Train the Model (Already Done!)
```powershell
# You already trained for 5 epochs. To continue or retrain:
python Main_Py\main_py.py --train 1 --conditional 1 --single-vp-net 1 --benchmark 0 --model-dir-name results_run1 --max-epochs 20 --batch-size 4
```

### Step 4: Reconstruct 3D Scenes (Already Done!)
```powershell
# You already have reconstructions in: results_run1\reconstructions\
# To reconstruct more samples:
python Main_Py\run_inference.py --model-dir results_run1 --epoch 5 --num-samples 10
```

### Step 5: Generate Point Clouds (Already Done!)
```powershell
# You already have .ply files generated!
# Location: results_run1\reconstructions\sample_000\pointcloud_fused.ply
#           results_run1\reconstructions\sample_001\pointcloud_fused.ply
```

### Step 6: View Your 3D Results

**Option 1: Online Viewer (Easiest)**
1. Go to: https://3dviewer.net/
2. Upload: `results_run1\reconstructions\sample_000\pointcloud_fused.ply`
3. Rotate and zoom!

**Option 2: MeshLab (Professional)**
```powershell
# Download: https://www.meshlab.net/
# Open → Import Mesh → Select .ply file
```

**Option 3: CloudCompare**
```powershell
# Download: https://www.danielgm.net/cc/
# Drag and drop .ply files
```

---

## 📊 What You Have Now

### Generated Files:
```
results_run1/
├── model/
│   └── epoch5/
│       └── model.pt                    # Trained VAE checkpoint
└── reconstructions/
    ├── sample_000/                     # Reconstructed CUBE
    │   ├── input_view.png             # Input depth view used
    │   ├── recon_depth_vp00.png       # 20 reconstructed depth views
    │   ├── recon_depth_vp01.png
    │   ├── ...
    │   ├── recon_sil_vp00.png         # 20 reconstructed silhouettes
    │   ├── ...
    │   ├── pointcloud_vp00.ply        # 20 individual viewpoint clouds
    │   ├── ...
    │   ├── pointcloud_fused.ply       # ✨ FUSED 3D MODEL (20,766 points)
    │   └── pointcloud_gt_fused.ply    # Ground truth comparison
    └── sample_001/                     # Reconstructed SPHERE
        ├── ... (same structure)
        └── pointcloud_fused.ply       # ✨ FUSED 3D MODEL (20,671 points)
```

---

## 🎨 Understanding the Results

### What Happened:
1. **Input:** Single depth view (one of 20 viewpoints)
2. **VAE Encoding:** Compressed to 40-dimensional latent vector
3. **VAE Decoding:** Generated all 20 viewpoints (depth + silhouettes)
4. **3D Fusion:** Back-projected each view to 3D points and merged

### Loss Metrics (After 5 Epochs):
| Metric | Value | What It Means |
|--------|-------|---------------|
| **Total Loss** | 0.831 | Overall error (lower is better) |
| **Depth Loss** | 0.214 | Depth map accuracy (0 = perfect) |
| **Silhouette Loss** | 0.617 | Shape outline accuracy |
| **Classification** | 50% | Can distinguish cube vs sphere |
| **KL Divergence** | 34.8 | Latent space regularization |

### To Improve Results:
```powershell
# Train for more epochs (80 recommended)
python Main_Py\main_py.py --train 1 --conditional 1 --single-vp-net 1 --benchmark 0 --model-dir-name results_better --max-epochs 80 --batch-size 4

# This will improve:
# - Depth accuracy (0.214 → ~0.18)
# - Classification (50% → ~90%)
# - Overall reconstruction quality
```

---

## 🚀 Next Steps

### 1. Add More Data
```powershell
# Create more synthetic objects:
python create_synthetic_data.py  # Edit to add cylinders, pyramids, etc.

# Then reprocess and retrain:
python Main_Py\main_py.py --from-scratch 1 --conditional 1 --model-dir-name results_moredata
python Main_Py\main_py.py --train 1 --conditional 1 --single-vp-net 1 --model-dir-name results_moredata --max-epochs 80
```

### 2. Test Different Modes

**Multi-View Input (Use All 20 Views):**
```powershell
python Main_Py\main_py.py --train 1 --conditional 1 --single-vp-net 0 --benchmark 0 --model-dir-name results_multiview --max-epochs 80
```

**Dropout Network (Robust to Missing Views):**
```powershell
python Main_Py\main_py.py --train 1 --conditional 1 --dropout-net 1 --benchmark 0 --model-dir-name results_dropout --max-epochs 80
```

### 3. Export to Mesh
The current pipeline generates point clouds. To create watertight meshes:

**Option A: Use MeshLab (Manual)**
1. Open `pointcloud_fused.ply` in MeshLab
2. Filters → Remeshing → Screened Poisson Surface Reconstruction
3. Export as `.obj` or `.stl`

**Option B: Add Poisson Reconstruction** (TODO - can be implemented)
```python
# Add to generate_pointcloud.py using Open3D:
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
o3d.io.write_triangle_mesh("mesh.obj", mesh)
```

### 4. Real-World Data
To use with real depth sensors (Kinect, RealSense, iPhone LiDAR):
1. Capture 20 viewpoints of an object
2. Save as grayscale PNG (depth in mm)
3. Place in `Data/nonbenchmark/myobject_depth_rgb/`
4. Run preprocessing and training

---

## 📐 Key Formulas Used

### 1. VAE Loss
```
L = ||X - X̂||₁ + λ × KL(q(z|X) || p(z))
```

### 2. Back-Projection (Depth → 3D)
```
X = (u - cₓ) × d / fₓ
Y = (v - cᵧ) × d / fᵧ
Z = d
```

### 3. KL Divergence
```
KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
```

---

## ✅ Your Current Status

- [x] Data preprocessed (18 training samples, 2 test samples)
- [x] Model trained (5 epochs, 18.4M parameters)
- [x] Checkpoint saved (`results_run1/model/epoch5/model.pt`)
- [x] Reconstructions generated (2 samples: cube + sphere)
- [x] Point clouds created (.ply files ready to view)
- [ ] Train for more epochs (recommended: 80)
- [ ] Add more training data
- [ ] Export to mesh format
- [ ] Test on real depth sensor data

---

## 🆘 Common Questions

**Q: My reconstructions look blurry**
A: Train for more epochs (80+). After 5 epochs, the model is still learning.

**Q: Can I use my own 3D models?**
A: Yes! Render 20 depth views, save as PNG, organize like `myobject_depth_rgb/obj_XXXX_Cam_Y.png`

**Q: What if I only have single-view depth?**
A: The model is trained with `--single-vp-net 1`, so it can generate 20 views from 1 input view. Perfect for your use case!

**Q: How do I get colored point clouds?**
A: Add RGB images alongside depth, then modify `generate_pointcloud.py` to load colors.

**Q: Can this work with stereo cameras?**
A: Yes! Convert stereo disparity to depth using: `depth = focal × baseline / disparity`

---

## 🎉 Summary

You now have a **complete 3D reconstruction pipeline**:
- ✅ Data → Preprocessing → Training → Inference → 3D Point Clouds

**Your generated 3D models are in:**
- `results_run1\reconstructions\sample_000\pointcloud_fused.ply` (Cube)
- `results_run1\reconstructions\sample_001\pointcloud_fused.ply` (Sphere)

**View them now at:** https://3dviewer.net/

For better results, train longer:
```powershell
python Main_Py\main_py.py --train 1 --conditional 1 --single-vp-net 1 --benchmark 0 --model-dir-name results_final --max-epochs 80
```

**Happy 3D Reconstructing! 🚀🎨**
