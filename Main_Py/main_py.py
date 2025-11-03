import argparse
import torch
import os
from train_vae import VAETrainer
from data_loader import DataLoader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='3D Shape VAE Training')
    
    # Global options
    parser.add_argument('--global-data-type', type=str, default='float',
                       choices=['float', 'double'],
                       help='Default tensor data type')
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed (0 for random)')
    parser.add_argument('--test-phase', type=int, default=0, choices=[0, 1],
                       help='Run in test mode')
    parser.add_argument('--model-dir-name', type=str, default='',
                       help='Directory name for saving models and results')
    parser.add_argument('--benchmark', type=int, default=0, choices=[0, 1],
                       help='Use benchmark dataset (ModelNet40)')
    
    # Data options
    parser.add_argument('--max-memory', type=int, default=3000,
                       help='Maximum memory in MB for data processing')
    parser.add_argument('--zip', type=int, default=0, choices=[0, 1],
                       help='Extract zip files')
    parser.add_argument('--from-scratch', type=int, default=0, choices=[0, 1],
                       help='Reprocess data from scratch')
    parser.add_argument('--raw-data-type', type=str, default='int',
                       choices=['int', 'float'],
                       help='Raw data file type (int=png, float=txt)')
    parser.add_argument('--p-train', type=float, default=0.925,
                       help='Training set proportion')
    parser.add_argument('--p-valid', type=float, default=0.045,
                       help='Validation set proportion')
    parser.add_argument('--p-test', type=float, default=0.03,
                       help='Test set proportion')
    parser.add_argument('--resize-scale', type=float, default=1.0,
                       help='Image resize scale (0, 1]')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--num-vps', type=int, default=20,
                       help='Number of viewpoints')
    parser.add_argument('--train', type=int, default=1, choices=[0, 1],
                       help='Start training')
    
    # Model options
    parser.add_argument('--n-ch', type=int, default=74,
                       help='Base number of feature maps')
    parser.add_argument('--n-latents', type=int, default=0,
                       help='Number of latent dimensions (0=auto)')
    parser.add_argument('--dropout-net', type=int, default=0, choices=[0, 1],
                       help='Drop 15-18 views during training')
    parser.add_argument('--silhouette-input', type=int, default=0, choices=[0, 1],
                       help='Use only silhouettes for training')
    parser.add_argument('--single-vp-net', type=int, default=1, choices=[0, 1],
                       help='Train with single random viewpoint')
    parser.add_argument('--conditional', type=int, default=-1, choices=[-1, 0, 1],
                       help='Train conditional model (-1=must specify)')
    parser.add_argument('--kld', type=float, default=0,
                       help='KLD loss coefficient (0=auto)')
    
    # Training options
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--batch-size-change-epoch', type=int, default=20,
                       help='Change batch size every N epochs')
    parser.add_argument('--batch-size-change', type=int, default=2,
                       help='Batch size increment')
    parser.add_argument('--target-batch-size', type=int, default=8,
                       help='Target batch size')
    parser.add_argument('--initial-lr', type=float, default=0.000002,
                       help='Initial learning rate')
    parser.add_argument('--lr', type=float, default=0.000085,
                       help='Learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.98,
                       help='Learning rate decay')
    parser.add_argument('--max-epochs', type=int, default=80,
                       help='Maximum training epochs')
    parser.add_argument('--tanh', type=int, default=0, choices=[0, 1],
                       help='Use tanh activation (normalize to [-1, 1])')
    
    # Experiment options
    parser.add_argument('--canvas-hw', type=int, default=5,
                       help='Canvas height/width for samples')
    parser.add_argument('--n-samples', type=int, default=2,
                       help='Number of sample sets')
    parser.add_argument('--manifold-exp', type=str, default='randomSampling',
                       choices=['randomSampling', 'interpolation'],
                       help='Manifold experiment type')
    parser.add_argument('--mean', type=float, default=0,
                       help='Z vector mean')
    parser.add_argument('--var', type=float, default=1,
                       help='Z vector variance')
    parser.add_argument('--n-reconstructions', type=int, default=50,
                       help='Number of reconstructions')
    parser.add_argument('--experiment', type=int, default=0, choices=[0, 1],
                       help='Run experiments on pretrained model')
    parser.add_argument('--exp-type', type=str, default='sample',
                       help='Experiment type')
    parser.add_argument('--from-epoch', type=int, default=0,
                       help='Load model from epoch')
    parser.add_argument('--sample-category', type=str, default='',
                       help='Categories for conditional sampling')
    parser.add_argument('--vp-to-keep', type=int, default=100,
                       help='Viewpoint index to keep')
    
    return parser.parse_args()


def validate_args(opt):
    """Validate and process arguments"""
    # Convert binary flags
    opt.zip = bool(opt.zip)
    opt.from_scratch = bool(opt.from_scratch)
    opt.test_phase = bool(opt.test_phase)
    opt.benchmark = bool(opt.benchmark)
    opt.tanh = bool(opt.tanh)
    opt.dropout_net = bool(opt.dropout_net)
    opt.silhouette_input = bool(opt.silhouette_input)
    opt.single_vp_net = bool(opt.single_vp_net)
    opt.train = bool(opt.train)
    opt.experiment = bool(opt.experiment)
    
    # Validate conditional flag
    if opt.conditional == -1:
        raise ValueError("Must specify --conditional 0 or 1")
    opt.conditional = bool(opt.conditional)
    
    # Set auto values
    if opt.n_latents == 0:
        opt.n_latents = 40 if opt.conditional else 100
    
    if opt.kld == 0:
        if opt.benchmark:
            opt.kld = 200 if not opt.dropout_net and not opt.single_vp_net else \
                     150 if opt.dropout_net else 120
        else:
            opt.kld = 240 if not opt.dropout_net and not opt.single_vp_net else \
                     180 if opt.dropout_net else 150
    
    # Adjust parameters for benchmark
    if opt.benchmark:
        opt.n_ch = 70
        opt.max_epochs = 105
        opt.lr = 0.000092
    
    # Validate batch size
    if opt.batch_size < 2:
        print("Batch size must be at least 2, setting to 2")
        opt.batch_size = 2
    
    # Validate resize scale
    if opt.resize_scale <= 0 or opt.resize_scale > 1:
        opt.resize_scale = 1.0
    
    # Generate model directory name if not provided
    if not opt.model_dir_name:
        opt.model_dir_name = f'exp_{torch.rand(1).item():.4f}'
    
    # Process sample categories
    if opt.sample_category:
        opt.sample_category = [c.strip() for c in opt.sample_category.split(',')]
    
    return opt


def main():
    """Main entry point"""
    print("=" * 80)
    print("3D Shape VAE - PyTorch Implementation")
    print("=" * 80)
    
    # Parse arguments
    opt = parse_args()
    opt = validate_args(opt)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Model directory: {opt.model_dir_name}")
    print(f"  Benchmark mode: {opt.benchmark}")
    print(f"  Test phase: {opt.test_phase}")
    print(f"  Image size: {opt.img_size}")
    print(f"  Viewpoints: {opt.num_vps}")
    print(f"  Latent dimensions: {opt.n_latents}")
    print(f"  Conditional: {opt.conditional}")
    print(f"  Single VP: {opt.single_vp_net}")
    print(f"  Dropout Net: {opt.dropout_net}")
    
    # Set default tensor type
    if opt.global_data_type == 'float':
        torch.set_default_dtype(torch.float32)
    elif opt.global_data_type == 'double':
        torch.set_default_dtype(torch.float64)
    
    # Process data if needed
    if not opt.experiment and (opt.zip or opt.from_scratch):
        print("\nProcessing data...")
        data_loader = DataLoader(opt)
        data_loader.process_and_save_data()
    
    # Train model
    if not opt.experiment and opt.train:
        print("\nStarting training...")
        trainer = VAETrainer(opt)
        trainer.train()
    
    # Run experiments
    if opt.experiment:
        print("\nRunning experiments...")
        print("Experiment functionality not yet implemented")
        # Would call experiment runner here
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
