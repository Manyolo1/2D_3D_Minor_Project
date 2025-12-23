"""
Convert reconstructed depth maps to 3D point clouds and meshes.
Generates .ply files (point clouds) and optionally .obj files (meshes with faces).
"""
import argparse
import os
import numpy as np
from PIL import Image

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def parse_args():
    parser = argparse.ArgumentParser(description='Generate 3D point clouds from depth maps')
    
    parser.add_argument('--recon-dir', type=str, required=True,
                       help='Directory containing reconstructed samples')
    parser.add_argument('--focal-length', type=float, default=300.0,
                       help='Camera focal length in pixels')
    parser.add_argument('--baseline', type=float, default=0.1,
                       help='Camera baseline spacing (for multi-view)')
    parser.add_argument('--mesh', type=int, default=1, choices=[0, 1],
                       help='Generate mesh with faces using Poisson reconstruction')
    parser.add_argument('--single-sample', action='store_true',
                       help='Process recon-dir as a single sample (not containing sample_* subdirs)')
    
    return parser.parse_args()


def depth_to_point_cloud(depth_map, focal_length, cx=None, cy=None):
    """
    Back-project depth map to 3D points using pinhole camera model.
    
    Formula:
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
    
    Args:
        depth_map: [H, W] numpy array of depth values
        focal_length: camera focal length in pixels
        cx, cy: principal point (default: image center)
    
    Returns:
        points: [N, 3] array of (X, Y, Z) coordinates
    """
    h, w = depth_map.shape
    
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0
    
    # Create pixel coordinate meshgrid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Normalize depth to reasonable range (0 to 1)
    depth_normalized = depth_map / 255.0
    
    # Back-projection formulas
    X = (u - cx) * depth_normalized / focal_length
    Y = (v - cy) * depth_normalized / focal_length
    Z = depth_normalized
    
    # Filter out zero-depth points
    valid_mask = depth_normalized > 0.01
    
    points = np.stack([X[valid_mask], Y[valid_mask], Z[valid_mask]], axis=-1)
    
    return points


def save_ply(points, filename, colors=None):
    """
    Save point cloud as .ply file.
    
    Args:
        points: [N, 3] array of (X, Y, Z)
        filename: output .ply path
        colors: [N, 3] array of (R, G, B) in [0, 255], optional
    """
    with open(filename, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        # Write vertex data
        for i in range(len(points)):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i].astype(int)
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
            else:
                f.write(f"{x} {y} {z}\n")


def reconstruct_mesh_poisson(points, filename, depth=9):
    """
    Reconstruct mesh from point cloud using Ball Pivoting Algorithm (more robust than Poisson).
    Requires Open3D library.
    
    Args:
        points: [N, 3] array of (X, Y, Z)
        filename: output .ply path
        depth: Not used for Ball Pivoting
    
    Returns:
        True if successful, False otherwise
    """
    if not HAS_OPEN3D:
        print("  Warning: Open3D not installed. Skipping mesh generation.")
        print("  Install with: pip install open3d")
        return False
    
    try:
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals (required for Ball Pivoting)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))
        
        # Ball Pivoting Algorithm for surface reconstruction
        # Compute radii for the ball - important parameter!
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist, avg_dist * 2]
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        
        if mesh is None or len(mesh.triangles) == 0:
            print(f"    Surface reconstruction failed (no triangles generated)")
            return False
        
        # Clean up mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Save mesh
        o3d.io.write_triangle_mesh(filename, mesh)
        n_vertices = len(mesh.vertices)
        n_triangles = len(mesh.triangles)
        return True
    
    except Exception as e:
        print(f"    Error during mesh reconstruction: {type(e).__name__}: {e}")
        return False


def fuse_multi_view_point_clouds(points_list, radius=0.05):
    """
    Fuse multiple point clouds from different viewpoints.
    Simple version: concatenate and optionally downsample.
    
    Args:
        points_list: list of [N_i, 3] point arrays
        radius: voxel size for downsampling (0 = no downsampling)
    
    Returns:
        fused_points: [M, 3] fused point cloud
    """
    if not points_list:
        return np.array([])
    
    # Concatenate all points
    all_points = np.vstack(points_list)
    
    if radius > 0:
        # Simple voxel-based downsampling
        voxel_indices = np.floor(all_points / radius).astype(int)
        unique_voxels, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        all_points = all_points[unique_indices]
    
    return all_points


def main():
    args = parse_args()
    
    print("=" * 80)
    print("3D Point Cloud Generation from Depth Maps")
    print("=" * 80)
    
    if not os.path.exists(args.recon_dir):
        print(f"Error: Directory not found: {args.recon_dir}")
        return
    
    # Process either single sample or multiple samples
    if args.single_sample:
        # Process recon_dir directly as a single sample
        sample_dirs = [('single_sample', args.recon_dir)]
    else:
        # Process each sample directory
        sample_dir_names = sorted([d for d in os.listdir(args.recon_dir) 
                             if os.path.isdir(os.path.join(args.recon_dir, d))])
        sample_dirs = [(name, os.path.join(args.recon_dir, name)) for name in sample_dir_names]
    
    if not sample_dirs:
        print(f"No sample directories found in {args.recon_dir}")
        return
    
    print(f"Found {len(sample_dirs)} samples to process")
    
    for sample_dir_name, sample_path in sample_dirs:
        
        # Find all reconstructed depth viewpoints
        # Support both naming patterns: recon_depth_vp* and *_depth_vp*
        depth_files = sorted([f for f in os.listdir(sample_path) 
                            if (f.startswith('recon_depth_vp') or '_depth_vp' in f) and f.endswith('.png')])
        
        if not depth_files:
            print(f"  {sample_dir_name}: No depth files found, skipping")
            continue
        
        print(f"\n  Processing {sample_dir_name}:")
        print(f"    Found {len(depth_files)} viewpoints")
        
        # Generate point cloud for each viewpoint
        viewpoint_clouds = []
        
        for depth_file in depth_files:
            depth_path = os.path.join(sample_path, depth_file)
            depth_img = Image.open(depth_path).convert('L')
            depth_array = np.array(depth_img)
            
            # Back-project to 3D
            points = depth_to_point_cloud(depth_array, args.focal_length)
            viewpoint_clouds.append(points)
        
        # Save individual viewpoint clouds
        for i, points in enumerate(viewpoint_clouds):
            vp_ply = os.path.join(sample_path, f'pointcloud_vp{i:02d}.ply')
            save_ply(points, vp_ply)
        
        # Fuse all viewpoints into single cloud
        fused_points = fuse_multi_view_point_clouds(viewpoint_clouds, radius=0.02)
        fused_ply = os.path.join(sample_path, 'pointcloud_fused.ply')
        save_ply(fused_points, fused_ply)
        
        print(f"    Generated {len(viewpoint_clouds)} individual point clouds")
        print(f"    Fused cloud: {len(fused_points)} points -> {fused_ply}")
        
        # Generate mesh with faces if Open3D is available
        if args.mesh:
            fused_ply_mesh = os.path.join(sample_path, 'mesh_fused.ply')
            if reconstruct_mesh_poisson(fused_points, fused_ply_mesh, depth=9):
                print(f"    Mesh with faces: {fused_ply_mesh} [OK]")
        
        # Also process ground truth if available
        gt_files = sorted([f for f in os.listdir(sample_path) 
                          if f.startswith('gt_depth_vp') and f.endswith('.png')])
        
        if gt_files:
            gt_clouds = []
            for gt_file in gt_files:
                gt_path = os.path.join(sample_path, gt_file)
                gt_img = Image.open(gt_path).convert('L')
                gt_array = np.array(gt_img)
                points = depth_to_point_cloud(gt_array, args.focal_length)
                gt_clouds.append(points)
            
            gt_fused = fuse_multi_view_point_clouds(gt_clouds, radius=0.02)
            gt_ply = os.path.join(sample_path, 'pointcloud_gt_fused.ply')
            save_ply(gt_fused, gt_ply)
            print(f"    Ground truth fused: {len(gt_fused)} points -> {gt_ply}")
    
    print("\n" + "=" * 80)
    print("Point cloud generation complete!")
    print("View .ply files in MeshLab, CloudCompare, or online viewers")
    print("=" * 80)


if __name__ == '__main__':
    main()
