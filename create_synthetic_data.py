"""
Create synthetic multi-view depth data for testing the VAE pipeline.
Generates simple geometric shapes (sphere, cube) with 20 viewpoints each.
"""
import os
import numpy as np
from PIL import Image


def generate_sphere_depth(size=224, radius=80, center_x=112, center_y=112, angle=0):
    """Generate a depth map of a sphere from a specific viewpoint."""
    depth = np.zeros((size, size), dtype=np.uint8)
    
    for i in range(size):
        for j in range(size):
            dx = j - center_x
            dy = i - center_y
            
            # Rotate point
            dx_rot = dx * np.cos(angle) - dy * np.sin(angle)
            dy_rot = dx * np.sin(angle) + dy * np.cos(angle)
            
            dist_sq = dx_rot**2 + dy_rot**2
            
            if dist_sq < radius**2:
                # Calculate depth (z-coordinate on sphere)
                z = np.sqrt(radius**2 - dist_sq)
                # Normalize to 0-255 range
                depth[i, j] = int((z / radius) * 200 + 55)
    
    return depth


def generate_cube_depth(size=224, cube_size=100, center_x=112, center_y=112, angle=0):
    """Generate a depth map of a cube from a specific viewpoint."""
    depth = np.zeros((size, size), dtype=np.uint8)
    
    half_size = cube_size // 2
    
    for i in range(size):
        for j in range(size):
            dx = j - center_x
            dy = i - center_y
            
            # Rotate point
            dx_rot = dx * np.cos(angle) - dy * np.sin(angle)
            dy_rot = dx * np.sin(angle) + dy * np.cos(angle)
            
            if abs(dx_rot) < half_size and abs(dy_rot) < half_size:
                # Simple flat face with slight gradient
                depth_val = 150 + int((dx_rot / half_size) * 30)
                depth[i, j] = np.clip(depth_val, 0, 255)
    
    return depth


def create_multiview_dataset(output_dir, num_objects=10, num_viewpoints=20):
    """Create a synthetic multi-view dataset."""
    
    # Create class folders
    classes = {
        'sphere': generate_sphere_depth,
        'cube': generate_cube_depth
    }
    
    for class_name, generator_func in classes.items():
        class_folder = os.path.join(output_dir, f'{class_name}_depth_rgb')
        os.makedirs(class_folder, exist_ok=True)
        
        print(f"Generating {class_name} data...")
        
        for obj_id in range(num_objects):
            for vp_id in range(num_viewpoints):
                # Calculate viewpoint angle (evenly distributed around object)
                angle = (vp_id / num_viewpoints) * 2 * np.pi
                
                # Generate depth map
                if class_name == 'sphere':
                    depth_map = generator_func(angle=angle)
                else:
                    depth_map = generator_func(angle=angle)
                
                # Add some noise for variety
                noise = np.random.randint(-10, 10, depth_map.shape, dtype=np.int16)
                depth_map = np.clip(depth_map.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # Save as PNG
                filename = f'{class_name}_{obj_id:04d}_Cam_{vp_id}.png'
                filepath = os.path.join(class_folder, filename)
                
                img = Image.fromarray(depth_map, mode='L')
                img.save(filepath)
        
        print(f"  Created {num_objects} objects with {num_viewpoints} views each in {class_folder}")


if __name__ == '__main__':
    output_directory = 'Data/nonbenchmark'
    
    print("=" * 80)
    print("Creating Synthetic Multi-View Depth Dataset")
    print("=" * 80)
    print(f"Output directory: {output_directory}")
    print()
    
    create_multiview_dataset(
        output_dir=output_directory,
        num_objects=10,  # 10 objects per class
        num_viewpoints=20  # 20 viewpoints per object
    )
    
    print()
    print("=" * 80)
    print("Dataset creation complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Run preprocessing:")
    print("   python Main_Py\\main_py.py --from-scratch 1 --raw-data-type int --conditional 1 --model-dir-name results_run1")
    print()
    print("2. Run training:")
    print("   python Main_Py\\main_py.py --train 1 --conditional 1 --single-vp-net 1 --benchmark 0 --model-dir-name results_run1")
