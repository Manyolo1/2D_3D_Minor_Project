#!/usr/bin/env python3
"""
Single Image to 3D Scene Reconstruction
Input: One 2D depth image or RGB image
Output: Full 3D mesh reconstruction
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import subprocess
import tempfile

# Lazy imports for depth estimation
try:
    import torch.hub  # noqa: F401
except Exception:
    pass

# Import existing modules
from vae_model import VAE, Sampler


class SingleImageReconstructor:
    """Reconstruct 3D scene from a single 2D input image"""
    
    def __init__(self, model_dir, epoch, img_size=224, num_viewpoints=20, latent_dim=40):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.num_viewpoints = num_viewpoints
        self.latent_dim = latent_dim
        
        # Load trained model
        checkpoint_path = os.path.join(model_dir, 'model', f'epoch{epoch}', 'model.pt')
        print(f"Loading checkpoint: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model parameters
        self.conditional = checkpoint.get('conditional', True)
        self.single_vp = checkpoint.get('single_vp', True)
        self.num_classes = checkpoint.get('num_classes', 2)
        self.tanh = checkpoint.get('tanh', 0)
        feature_maps = checkpoint.get('feature_maps', 74)
        
        # Model params tuple: (n_input_ch, n_ch, n_latents, tanh, single_vp_net, conditional, num_cats, benchmark, dropout_net)
        model_params = (
            self.num_viewpoints,  # n_input_ch
            feature_maps,         # n_ch (base channels)
            self.latent_dim,      # n_latents
            self.tanh,            # tanh
            self.single_vp,       # single_vp_net
            self.conditional,     # conditional
            self.num_classes,     # num_cats
            False,                # benchmark
            False                 # dropout_net
        )
        
        # Initialize encoder and decoder
        self.encoder = VAE.get_encoder(model_params)
        self.decoder = VAE.get_decoder(model_params)
        
        # Load state dicts (checkpoint stores encoder/decoder separately)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.encoder.eval()
        self.decoder.eval()
        
        self.sampler = Sampler()
        print(f"Model loaded successfully on {self.device}")
    
    def preprocess_image(self, image_path, is_depth=True):
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to input image (PNG/JPG)
            is_depth: If True, treat as depth map; if False, convert RGB to grayscale
        
        Returns:
            Preprocessed tensor [1, 1, H, W]
        """
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale if RGB
        if img.mode == 'RGB' or img.mode == 'RGBA':
            if is_depth:
                # For depth, take first channel or convert to grayscale
                img = img.convert('L')
            else:
                # For RGB, convert to grayscale depth estimation (simple approach)
                img = img.convert('L')
        
        # Resize to model input size
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        # Convert to tensor [1, 1, H, W]
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        
        return img_tensor.to(self.device)

    def estimate_depth_from_rgb(self, image_path, model_name='midas-small', save_path=None):
        """Estimate a depth map from an RGB image using MiDaS variants.

        Args:
            image_path: input RGB image path
            model_name: 'midas-small' (fast) or 'midas-large' (better quality)
            save_path: where to save the estimated depth PNG (grayscale [0,255])

        Returns: path to saved depth map (PNG)
        """
        try:
            midas_model = 'MiDaS_small' if model_name == 'midas-small' else 'DPT_Large'
            transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            midas = torch.hub.load('intel-isl/MiDaS', midas_model)
            midas.to(self.device)
            midas.eval()

            if model_name == 'midas-small':
                transform = transforms.small_transform
            else:
                transform = transforms.dpt_transform

            img = Image.open(image_path).convert('RGB')
            orig_w, orig_h = img.size
            input_batch = transform(img).to(self.device)

            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(orig_h, orig_w),
                    mode='bicubic',
                    align_corners=False,
                ).squeeze()

            depth = prediction.cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_img = (depth * 255).astype(np.uint8)

            if save_path is None:
                fd, save_path = tempfile.mkstemp(suffix='.png')
                os.close(fd)

            Image.fromarray(depth_img, mode='L').save(save_path)
            print(f"  ✓ Estimated depth saved to {save_path}")
            return save_path
        except Exception as e:
            raise RuntimeError(f"Depth estimation failed: {e}")
    
    def reconstruct_multiview(self, input_tensor, category=0):
        """
        Reconstruct all viewpoints from a single input view
        
        Args:
            input_tensor: Preprocessed input tensor [1, 1, H, W]
            category: Category index for conditional model (0=cube, 1=sphere, etc.)
        
        Returns:
            Dictionary with depth and silhouette predictions for all viewpoints
        """
        with torch.no_grad():
            # Encode single view to latent space  
            mu, log_var, class_logits = self.encoder(input_tensor)
            
            # Sample latent vector
            z = self.sampler(mu, log_var)
            
            # Prepare category tensor
            batch_size = z.shape[0]
            categories = torch.tensor([category] * batch_size, dtype=torch.long).to(self.device)
            
            # Decode to all viewpoints
            # Convert category to one-hot if conditional
            if self.conditional:
                cat_onehot = F.one_hot(categories, num_classes=self.num_classes).float()
                depth_channels, silhouette_channels = self.decoder((z, cat_onehot))
            else:
                depth_channels, silhouette_channels = self.decoder(z)
            
            return {
                'depth': depth_channels.cpu(),
                'silhouette': silhouette_channels.cpu(),
                'latent': z.cpu(),
                'mu': mu.cpu(),
                'log_var': log_var.cpu()
            }
    
    def save_reconstructions(self, predictions, output_dir, input_name):
        """Save all reconstructed viewpoints as images"""
        os.makedirs(output_dir, exist_ok=True)
        
        depth = predictions['depth'][0]  # [num_vps, H, W]
        silhouette = predictions['silhouette'][0]  # [num_vps, H, W]
        
        print(f"\nSaving reconstructions to: {output_dir}")
        
        for vp_idx in range(self.num_viewpoints):
            # Save depth map
            depth_img = (depth[vp_idx].numpy() * 255).astype(np.uint8)
            depth_path = os.path.join(output_dir, f'{input_name}_depth_vp{vp_idx:02d}.png')
            Image.fromarray(depth_img, mode='L').save(depth_path)
            
            # Save silhouette
            sil_img = (silhouette[vp_idx].numpy() * 255).astype(np.uint8)
            sil_path = os.path.join(output_dir, f'{input_name}_silhouette_vp{vp_idx:02d}.png')
            Image.fromarray(sil_img, mode='L').save(sil_path)
        
        print(f"  ✓ Saved {self.num_viewpoints} depth maps")
        print(f"  ✓ Saved {self.num_viewpoints} silhouettes")
        
        return output_dir
    
    def generate_3d_mesh(self, recon_dir, focal_length=300):
        """Generate 3D point cloud and mesh from reconstructed views"""
        print("\n" + "="*80)
        print("Generating 3D mesh from reconstructed views...")
        print("="*80)
        
        # Call generate_pointcloud.py
        script_path = os.path.join(os.path.dirname(__file__), 'generate_pointcloud.py')
        
        cmd = [
            sys.executable,
            script_path,
            '--recon-dir', recon_dir,
            '--focal-length', str(focal_length),
            '--single-sample'  # Process only this sample
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            print("\n✓ 3D mesh generation complete!")
            
            # Find generated mesh file
            mesh_path = os.path.join(recon_dir, 'mesh_fused.ply')
            if os.path.exists(mesh_path):
                print(f"\n📦 3D Mesh saved to: {mesh_path}")
                return mesh_path
        else:
            print("Error generating 3D mesh:")
            print(result.stderr)
            return None


def main():
    parser = argparse.ArgumentParser(description='Single Image to 3D Scene Reconstruction')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input 2D image (depth map or RGB)')
    parser.add_argument('--model-dir', type=str, default='results_run1',
                        help='Directory containing trained model')
    parser.add_argument('--epoch', type=int, default=80,
                        help='Model checkpoint epoch to use')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: input_to_3d/)')
    parser.add_argument('--category', type=int, default=0,
                        help='Object category: 0=cube, 1=sphere, etc.')
    parser.add_argument('--is-depth', type=int, default=1,
                        help='1 if input is depth map, 0 if RGB image')
    parser.add_argument('--depth-from-rgb', action='store_true',
                        help='If set, run monocular depth estimation on RGB before reconstruction (implies is-depth=0)')
    parser.add_argument('--rgb-depth-model', type=str, default='midas-small', choices=['midas-small', 'midas-large'],
                        help='Model to use for RGB->depth (midas-small faster, midas-large higher quality)')
    parser.add_argument('--focal-length', type=float, default=300,
                        help='Camera focal length for 3D projection')
    parser.add_argument('--skip-mesh', action='store_true',
                        help='Skip 3D mesh generation (only save multi-view images)')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        args.output_dir = os.path.join('input_to_3d', input_basename)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Single Image to 3D Scene Reconstruction")
    print("="*80)
    print(f"Input image: {args.input}")
    print(f"Model: {args.model_dir} (epoch {args.epoch})")
    print(f"Output: {args.output_dir}")
    print(f"Category: {args.category}")
    print(f"Input type: {'Depth map' if args.is_depth else 'RGB image'}")
    if args.depth_from_rgb:
        print("Depth estimation: enabled (monocular RGB -> depth)")
    print("="*80)
    
    # Initialize reconstructor
    reconstructor = SingleImageReconstructor(
        model_dir=args.model_dir,
        epoch=args.epoch
    )

    # Load and preprocess input image (with optional depth estimation)
    input_path_for_recon = args.input
    effective_is_depth = bool(args.is_depth)

    if args.depth_from_rgb or not args.is_depth:
        print("\n[1/4] Estimating depth from RGB...")
        depth_path = os.path.join(args.output_dir, 'estimated_depth.png')
        input_path_for_recon = reconstructor.estimate_depth_from_rgb(
            args.input,
            model_name=args.rgb_depth_model,
            save_path=depth_path,
        )
        effective_is_depth = True
        print("  ✓ Depth estimated from RGB")
    else:
        print("\n[1/4] Loading input depth image...")

    input_tensor = reconstructor.preprocess_image(input_path_for_recon, is_depth=effective_is_depth)
    print(f"  ✓ Image loaded and preprocessed: {input_tensor.shape}")
    
    # Reconstruct all viewpoints
    print("\n[2/4] Reconstructing multi-view scene...")
    predictions = reconstructor.reconstruct_multiview(input_tensor, category=args.category)
    print(f"  ✓ Generated {reconstructor.num_viewpoints} viewpoints")
    print(f"  ✓ Latent representation: {predictions['latent'].shape}")
    
    # Save reconstructed views
    print("\n[3/4] Saving reconstructed views...")
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    recon_dir = reconstructor.save_reconstructions(predictions, args.output_dir, input_name)
    
    # Generate 3D mesh
    if not args.skip_mesh:
        print("\n[4/4] Generating 3D mesh...")
        mesh_path = reconstructor.generate_3d_mesh(recon_dir, focal_length=args.focal_length)
        
        if mesh_path:
            print("\n" + "="*80)
            print("✓ SUCCESS! 3D reconstruction complete!")
            print("="*80)
            print(f"\nGenerated files in: {args.output_dir}")
            print(f"  • Multi-view depth maps: {input_name}_depth_vp*.png")
            print(f"  • Multi-view silhouettes: {input_name}_silhouette_vp*.png")
            print(f"  • 3D Mesh: mesh_fused.ply")
            print(f"  • 3D Point Cloud: pointcloud_fused.ply")
            print("\nView 3D mesh:")
            print(f"  • Windows: start {mesh_path}")
            print(f"  • Python: python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_triangle_mesh('{mesh_path}')])")
            print(f"  • Online: Upload {mesh_path} to https://3dviewer.net")
    else:
        print("\n" + "="*80)
        print("✓ Multi-view reconstruction complete!")
        print("="*80)
        print(f"\nGenerated files in: {args.output_dir}")
        print(f"  • Multi-view depth maps: {input_name}_depth_vp*.png")
        print(f"  • Multi-view silhouettes: {input_name}_silhouette_vp*.png")


if __name__ == '__main__':
    main()
