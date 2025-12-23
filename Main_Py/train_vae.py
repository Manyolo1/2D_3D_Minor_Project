import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import time
import numpy as np
from vae_model import VAE, Sampler
from kld_criterion import KLDCriterion
from common_funcs import CommonFuncs


class VAETrainer:
    """Trainer class for Variational Autoencoder"""
    
    def __init__(self, opt):
        self.opt = opt
        self.common_funcs = CommonFuncs()
        
        # Set random seed
        if opt.seed > 0:
            torch.manual_seed(opt.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(opt.seed)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Get data paths
        self.train_files, self.valid_files, self.test_files = \
            self.common_funcs.obtain_data_path(opt.benchmark, opt.test_phase, True)
        
        # Load first training file to get dataset info
        print("Loading first training file...")
        data = torch.load(self.train_files[0])
        
        if opt.tanh:
            data['dataset'] = self.common_funcs.normalize_minus_one_to_one(data['dataset'])
        
        self.num_categories = len(data['category'])
        self.categories = data['category']
        
        # Model parameters
        model_params = [
            opt.num_vps,  # n_input_ch
            opt.n_ch,     # n_output_ch
            opt.n_latents, # n_latents
            opt.tanh,      # tanh
            opt.single_vp_net,  # single_vp_net
            opt.conditional,    # conditional
            self.num_categories if opt.conditional else 0,  # num_cats
            opt.benchmark,      # benchmark
            opt.dropout_net     # dropout_net
        ]
        
        # Build model
        self.encoder = VAE.get_encoder(model_params).to(self.device)
        self.decoder = VAE.get_decoder(model_params).to(self.device)
        self.sampler = Sampler().to(self.device)
        
        # Loss functions
        self.reconstruction_criterion = nn.L1Loss(reduction='sum')
        self.kld_criterion = KLDCriterion(opt.kld).to(self.device)
        
        if opt.conditional:
            self.classification_criterion = nn.CrossEntropyLoss(reduction='sum')
        
        # Optimizer
        self.parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(
            self.parameters,
            lr=opt.initial_lr,
            betas=(0.9, 0.999)
        )
        
        self.current_lr = opt.initial_lr
        
        # Training state
        self.epoch = 1
        self.batch_size = opt.batch_size
        
        print(f"\nModel Configuration:")
        print(f"  Latent dimensions: {opt.n_latents}")
        print(f"  Batch size: {opt.batch_size}")
        print(f"  Feature maps: {opt.n_ch}")
        print(f"  Learning rate: {opt.lr}")
        print(f"  KLD coefficient: {opt.kld}")
        print(f"  Conditional: {opt.conditional}")
        print(f"  Single VP: {opt.single_vp_net}")
        print(f"  Dropout Net: {opt.dropout_net}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters)
        print(f"  Total parameters: {total_params:,}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.encoder.train()
        self.decoder.train()
        
        epoch_loss = 0
        epoch_kld = 0
        epoch_recon_depth = 0
        epoch_recon_sil = 0
        epoch_class_loss = 0
        epoch_class_acc = 0
        num_samples = 0
        
        print(f"\nEpoch {self.epoch}: Training...")
        start_time = time.time()
        
        # Train on all data files
        for file_idx, train_file in enumerate(self.train_files):
            data = torch.load(train_file)
            
            if self.opt.tanh:
                data['dataset'] = self.common_funcs.normalize_minus_one_to_one(
                    data['dataset']
                )
            
            # Generate batch indices
            indices = self.common_funcs.generate_batch_indices(
                data['dataset'].size(0),
                self.batch_size
            )
            
            for batch_idx, idx in enumerate(indices):
                batch_data = data['dataset'][idx].to(self.device)
                batch_labels = data['labels'][idx] if self.opt.conditional else None
                
                # Create silhouettes
                silhouettes = batch_data.clone()
                if self.opt.tanh:
                    silhouettes[silhouettes > -1] = 1
                    silhouettes[silhouettes == -1] = 0
                else:
                    silhouettes[silhouettes > 0] = 1
                
                # Drop viewpoints if needed
                model_input = self.common_funcs.drop_input_vps(
                    batch_data if not self.opt.silhouette_input else silhouettes,
                    False,
                    self.opt.dropout_net,
                    None, None,
                    self.opt.single_vp_net
                )
                
                if not isinstance(model_input, torch.Tensor):
                    model_input = model_input[0]
                
                model_input = model_input.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if self.opt.conditional:
                    mean, logvar, class_scores = self.encoder(model_input)
                    
                    # Classification loss
                    class_loss = self.classification_criterion(
                        class_scores, 
                        batch_labels.to(self.device) - 1  # Convert to 0-indexed
                    )
                    epoch_class_loss += class_loss.item()
                    
                    # Classification accuracy
                    _, predicted = torch.max(class_scores, 1)
                    epoch_class_acc += (predicted == batch_labels.to(self.device) - 1).sum().item()
                    
                    # Create one-hot vectors
                    class_onehot = torch.zeros(
                        batch_data.size(0), 
                        self.num_categories
                    ).to(self.device)
                    class_onehot.scatter_(1, (batch_labels - 1).unsqueeze(1).to(self.device), 1)
                    
                    z = self.sampler(mean, logvar)
                    recon = self.decoder([z, class_onehot])
                else:
                    mean, logvar = self.encoder(model_input)
                    z = self.sampler(mean, logvar)
                    recon = self.decoder(z)
                
                # Reconstruction loss
                recon_depth, recon_sil = recon
                
                depth_loss = self.reconstruction_criterion(recon_depth, batch_data)
                sil_loss = self.reconstruction_criterion(recon_sil, silhouettes)
                recon_loss = depth_loss + sil_loss
                
                # KLD loss
                kld_loss = self.kld_criterion(mean, logvar)
                
                # Total loss
                total_loss = recon_loss + kld_loss
                if self.opt.conditional:
                    total_loss += class_loss
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                # Accumulate statistics
                batch_samples = batch_data.size(0)
                num_samples += batch_samples
                epoch_loss += total_loss.item()
                epoch_kld += kld_loss.item()
                epoch_recon_depth += depth_loss.item()
                epoch_recon_sil += sil_loss.item()
        
        # Compute epoch statistics
        img_size = self.opt.img_size
        num_vps = self.opt.num_vps
        
        epoch_loss /= (num_samples * img_size * img_size * num_vps)
        epoch_kld /= num_samples
        epoch_recon_depth /= (num_samples * img_size * img_size * num_vps)
        epoch_recon_sil /= (num_samples * img_size * img_size * num_vps)
        
        if self.opt.conditional:
            epoch_class_loss /= num_samples
            epoch_class_acc /= num_samples
        
        elapsed = time.time() - start_time
        
        # Print statistics
        if self.opt.conditional:
            print(f"Epoch {self.epoch}: "
                  f"Loss={epoch_loss:.4f}, KLD={epoch_kld:.1f}, "
                  f"Depth={epoch_recon_depth:.4f}, Sil={epoch_recon_sil:.4f}, "
                  f"ClassLoss={epoch_class_loss:.3f}, Acc={epoch_class_acc:.3f}, "
                  f"Samples={num_samples}, Time={elapsed/60:.1f}min")
        else:
            print(f"Epoch {self.epoch}: "
                  f"Loss={epoch_loss:.4f}, KLD={epoch_kld:.1f}, "
                  f"Depth={epoch_recon_depth:.4f}, Sil={epoch_recon_sil:.4f}, "
                  f"Samples={num_samples}, Time={elapsed/60:.1f}min")
        
        return {
            'loss': epoch_loss,
            'kld': epoch_kld,
            'depth': epoch_recon_depth,
            'sil': epoch_recon_sil,
            'class_loss': epoch_class_loss if self.opt.conditional else 0,
            'class_acc': epoch_class_acc if self.opt.conditional else 0
        }
    
    def validate(self):
        """Validate the model"""
        # Skip validation if no validation files
        if not self.valid_files or len(self.valid_files) == 0:
            print(f"Epoch {self.epoch}: No validation data, skipping validation")
            return 0.0
        
        self.encoder.eval()
        self.decoder.eval()
        
        valid_loss = 0
        valid_kld = 0
        valid_depth = 0
        valid_sil = 0
        valid_class_loss = 0
        valid_class_acc = 0
        num_samples = 0
        
        print(f"Epoch {self.epoch}: Validating...")
        
        with torch.no_grad():
            for valid_file in self.valid_files:
                data = torch.load(valid_file)
                
                if self.opt.tanh:
                    data['dataset'] = self.common_funcs.normalize_minus_one_to_one(
                        data['dataset']
                    )
                
                indices = self.common_funcs.generate_batch_indices(
                    data['dataset'].size(0),
                    self.batch_size
                )
                
                for idx in indices:
                    batch_data = data['dataset'][idx].to(self.device)
                    batch_labels = data['labels'][idx] if self.opt.conditional else None
                    
                    silhouettes = batch_data.clone()
                    if self.opt.tanh:
                        silhouettes[silhouettes > -1] = 1
                        silhouettes[silhouettes == -1] = 0
                    else:
                        silhouettes[silhouettes > 0] = 1
                    
                    model_input = self.common_funcs.drop_input_vps(
                        batch_data if not self.opt.silhouette_input else silhouettes,
                        False,
                        self.opt.dropout_net,
                        None, None,
                        self.opt.single_vp_net
                    )
                    
                    if not isinstance(model_input, torch.Tensor):
                        model_input = model_input[0]
                    
                    model_input = model_input.to(self.device)
                    
                    # Forward pass
                    if self.opt.conditional:
                        mean, logvar, class_scores = self.encoder(model_input)
                        
                        class_loss = self.classification_criterion(
                            class_scores, 
                            batch_labels.to(self.device) - 1
                        )
                        valid_class_loss += class_loss.item()
                        
                        _, predicted = torch.max(class_scores, 1)
                        valid_class_acc += (predicted == batch_labels.to(self.device) - 1).sum().item()
                        
                        class_onehot = torch.zeros(
                            batch_data.size(0), 
                            self.num_categories
                        ).to(self.device)
                        class_onehot.scatter_(1, (batch_labels - 1).unsqueeze(1).to(self.device), 1)
                        
                        z = self.sampler(mean, logvar)
                        recon = self.decoder([z, class_onehot])
                    else:
                        mean, logvar = self.encoder(model_input)
                        z = self.sampler(mean, logvar)
                        recon = self.decoder(z)
                    
                    recon_depth, recon_sil = recon
                    
                    depth_loss = self.reconstruction_criterion(recon_depth, batch_data)
                    sil_loss = self.reconstruction_criterion(recon_sil, silhouettes)
                    kld_loss = self.kld_criterion(mean, logvar)
                    
                    batch_samples = batch_data.size(0)
                    num_samples += batch_samples
                    valid_depth += depth_loss.item()
                    valid_sil += sil_loss.item()
                    valid_kld += kld_loss.item()
                    valid_loss += (depth_loss + sil_loss + kld_loss).item()
        
        # Compute statistics
        img_size = self.opt.img_size
        num_vps = self.opt.num_vps
        
        if num_samples == 0:
            print("Warning: No validation samples processed")
            return 0.0
        
        valid_loss /= (num_samples * img_size * img_size * num_vps)
        valid_kld /= num_samples
        valid_depth /= (num_samples * img_size * img_size * num_vps)
        valid_sil /= (num_samples * img_size * img_size * num_vps)
        
        if self.opt.conditional:
            valid_class_loss /= num_samples
            valid_class_acc /= num_samples
            print(f"Validation: Loss={valid_loss:.4f}, KLD={valid_kld:.1f}, "
                  f"Depth={valid_depth:.4f}, Sil={valid_sil:.4f}, "
                  f"ClassLoss={valid_class_loss:.3f}, Acc={valid_class_acc:.3f}")
        else:
            print(f"Validation: Loss={valid_loss:.4f}, KLD={valid_kld:.1f}, "
                  f"Depth={valid_depth:.4f}, Sil={valid_sil:.4f}")
        
        return valid_loss
    
    def save_model(self):
        """Save model checkpoint"""
        save_dir = os.path.join(self.opt.model_dir_name, 'model', f'epoch{self.epoch}')
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }, os.path.join(save_dir, 'model.pt'))
        
        print(f"Model saved to {save_dir}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 80)
        print("Starting Training")
        print("=" * 80)
        
        while self.epoch <= self.opt.max_epochs:
            # Train
            train_stats = self.train_epoch()
            
            # Validate
            valid_loss = self.validate()
            
            # Save model periodically (save final epoch and every 2 epochs after 18)
            if self.epoch == self.opt.max_epochs or (self.epoch >= 18 and self.epoch % 2 == 0):
                self.save_model()
            
            # Update learning rate
            self.update_learning_rate()
            
            # Update batch size
            if self.epoch % self.opt.batch_size_change_epoch == 0:
                self.update_batch_size()
            
            self.epoch += 1
        
        print("\nTraining completed!")
    
    def update_learning_rate(self):
        """Update learning rate based on schedule"""
        if self.epoch <= 10:
            # Warm-up phase
            target_epoch = 26
            lr_schedule = torch.linspace(
                self.opt.initial_lr,
                self.opt.lr,
                target_epoch
            )
            new_lr = lr_schedule[min(self.epoch * 2, target_epoch - 1)].item()
        elif self.epoch == 11:
            new_lr = self.opt.lr
        elif 12 <= self.epoch < 37:
            new_lr = self.current_lr * self.opt.lr_decay
        else:
            new_lr = self.current_lr
        
        if new_lr != self.current_lr:
            self.current_lr = new_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate updated to {new_lr:.6f}")
    
    def update_batch_size(self):
        """Update batch size"""
        if self.batch_size < self.opt.target_batch_size:
            old_bs = self.batch_size
            self.batch_size = min(
                self.batch_size + self.opt.batch_size_change,
                self.opt.target_batch_size
            )
            print(f"Batch size updated from {old_bs} to {self.batch_size}")
