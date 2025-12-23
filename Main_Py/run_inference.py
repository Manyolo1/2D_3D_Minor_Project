"""
Inference script to load trained VAE and reconstruct 3D views from test data.
Saves reconstructed depth maps and silhouettes as images.
"""
import argparse
import torch
import os
from PIL import Image
import numpy as np
from vae_model import VAE, Sampler
from common_funcs import CommonFuncs


def parse_args():
    parser = argparse.ArgumentParser(description='VAE Inference - Reconstruct 3D scenes')
    
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Model directory containing checkpoints')
    parser.add_argument('--epoch', type=int, default=5,
                       help='Epoch number to load')
    parser.add_argument('--benchmark', type=int, default=0, choices=[0, 1],
                       help='Use benchmark dataset')
    parser.add_argument('--tanh', type=int, default=0, choices=[0, 1],
                       help='Data normalized to [-1,1]')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of test samples to reconstruct')
    parser.add_argument('--output-dir', type=str, default='',
                       help='Output directory (default: model_dir/reconstructions)')
    
    return parser.parse_args()


def save_depth_image(tensor, path):
    """Save depth tensor [H, W] as grayscale image"""
    # Normalize to [0, 255]
    t = tensor.cpu().numpy()
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    img = (t * 255).astype(np.uint8)
    Image.fromarray(img, mode='L').save(path)


def main():
    args = parse_args()
    common_funcs = CommonFuncs()
    
    print("=" * 80)
    print("VAE Inference - 3D Scene Reconstruction")
    print("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = os.path.join(
        args.model_dir, 'model', f'epoch{args.epoch}', 'model.pt'
    )
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Available epochs:")
        model_dir = os.path.join(args.model_dir, 'model')
        if os.path.exists(model_dir):
            for item in os.listdir(model_dir):
                print(f"  {item}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get test data
    _, _, test_files = common_funcs.obtain_data_path(
        bool(args.benchmark), test_phase=False, lowest_size=False
    )
    
    if not test_files:
        print("Error: No test data found!")
        return
    
    print(f"Loading test data: {test_files[0]}")
    data = torch.load(test_files[0])
    
    if bool(args.tanh):
        data['dataset'] = common_funcs.normalize_minus_one_to_one(data['dataset'])
    
    # Infer model parameters from checkpoint and data
    num_samples = min(args.num_samples, data['dataset'].size(0))
    num_vps = data['dataset'].size(1)
    img_size = data['dataset'].size(2)
    num_categories = len(data['category'])
    
    # Reconstruct model architecture
    # We need to guess parameters; typically stored in opt during training
    # For now, use defaults matching training command
    model_params = [
        num_vps,      # n_input_ch (20)
        74,           # n_ch (default)
        40,           # n_latents (conditional default)
        args.tanh,    # tanh
        True,         # single_vp_net (from training command)
        True,         # conditional (from training command)
        num_categories,
        args.benchmark,
        False         # dropout_net
    ]
    
    encoder = VAE.get_encoder(model_params).to(device)
    decoder = VAE.get_decoder(model_params).to(device)
    sampler = Sampler().to(device)
    
    # Load weights
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    encoder.eval()
    decoder.eval()
    
    # Setup output directory
    if not args.output_dir:
        args.output_dir = os.path.join(args.model_dir, 'reconstructions')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nReconstructing {num_samples} samples...")
    print(f"Output directory: {args.output_dir}")
    
    with torch.no_grad():
        for sample_idx in range(num_samples):
            batch_data = data['dataset'][sample_idx:sample_idx+1].to(device)
            batch_label = data['labels'][sample_idx:sample_idx+1]
            
            # Prepare input (single VP)
            model_input = common_funcs.drop_input_vps(
                batch_data,
                mark_input=False,
                dropout_net=False,
                single_vp_net=True
            )
            
            if not isinstance(model_input, torch.Tensor):
                model_input = model_input[0]
            
            model_input = model_input.to(device)
            
            # Encode
            mean, logvar, class_scores = encoder(model_input)
            
            # Sample latent
            z = sampler(mean, logvar)
            
            # Decode with class conditioning
            class_onehot = torch.zeros(1, num_categories).to(device)
            class_onehot[0, batch_label[0] - 1] = 1
            
            recon_depth, recon_sil = decoder([z, class_onehot])
            
            # Denormalize if needed
            if bool(args.tanh):
                recon_depth = common_funcs.normalize_back_to_zero_to_one(recon_depth)
                recon_sil = common_funcs.normalize_back_to_zero_to_one(recon_sil)
            
            # Save reconstructions
            sample_dir = os.path.join(args.output_dir, f'sample_{sample_idx:03d}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save input view (the one used for reconstruction)
            input_vp = model_input[0, 0].cpu()
            save_depth_image(input_vp, os.path.join(sample_dir, 'input_view.png'))
            
            # Save all reconstructed viewpoints
            for vp_idx in range(num_vps):
                depth_vp = recon_depth[0, vp_idx]
                sil_vp = recon_sil[0, vp_idx]
                
                save_depth_image(depth_vp, os.path.join(sample_dir, f'recon_depth_vp{vp_idx:02d}.png'))
                save_depth_image(sil_vp, os.path.join(sample_dir, f'recon_sil_vp{vp_idx:02d}.png'))
            
            # Also save ground truth for comparison
            for vp_idx in range(num_vps):
                gt_depth = batch_data[0, vp_idx]
                save_depth_image(gt_depth, os.path.join(sample_dir, f'gt_depth_vp{vp_idx:02d}.png'))
            
            print(f"  Sample {sample_idx}: {data['category'][batch_label[0]-1]} -> {sample_dir}")
    
    print("\n" + "=" * 80)
    print(f"Reconstruction complete! {num_samples} samples saved to:")
    print(f"  {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
