"""
Quick script to add faces to point clouds using Open3D.
Tries alpha-shape first; falls back to convex hull.
"""
import os
import numpy as np
import open3d as o3d

recon_dir = r"results_run1\reconstructions"

for sample in ["sample_000", "sample_001"]:
    sample_path = os.path.join(recon_dir, sample)
    ply_file = os.path.join(sample_path, "pointcloud_fused.ply")
    mesh_file = os.path.join(sample_path, "mesh_fused.ply")
    
    print(f"Processing {sample}...")
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    print(f"  Loaded {len(pcd.points)} points")
    
    # Downsample for faster processing
    pcd = pcd.voxel_down_sample(voxel_size=0.03)
    print(f"  Downsampled to {len(pcd.points)} points")

    mesh = None
    # Try alpha-shape reconstruction (better surfaces than hull if alpha chosen well)
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.05)
        if mesh is None or len(mesh.triangles) == 0:
            raise RuntimeError("alpha shape produced empty mesh")
        print("  Alpha-shape reconstruction succeeded")
    except Exception as e:
        print(f"  Alpha-shape failed: {e}; falling back to convex hull")
        # Fallback: convex hull
        try:
            mesh, _ = pcd.compute_convex_hull()
            print("  Convex hull reconstruction succeeded")
        except Exception as e2:
            print(f"  Convex hull failed: {e2}")
            continue

    # Clean up mesh (optional)
    try:
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
    except Exception:
        pass

    # Save
    o3d.io.write_triangle_mesh(mesh_file, mesh)
    print(f"  Saved mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")
    print(f"  → {mesh_file}\n")

print("✓ Done! Mesh files with faces generated.")
