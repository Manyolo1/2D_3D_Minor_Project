import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt


class CommonFuncs:
    """Frequently-used utility functions"""
    
    @staticmethod
    def num_tensor_elements(tensor_size):
        """Computes the number of elements in a tensor when flattened"""
        tensor_elements = 1
        for i in range(len(tensor_size)):
            tensor_elements *= tensor_size[i]
        return tensor_elements
    
    @staticmethod
    def table_concat(t1, t2):
        """Concatenates two lists"""
        return t1 + t2
    
    @staticmethod
    def find_files(folder_path, file_type):
        """
        Returns all file paths with specified extension along with file names
        
        Args:
            folder_path: The path where to find files
            file_type: File extension to look for (e.g. 'zip', 'txt')
            
        Returns:
            files_path: List of full file paths
            file_names: List of file names without extension
        """
        file_names = []
        files_path = []
        
        pattern = os.path.join(folder_path, f'*.{file_type}')
        for file_path in glob.glob(pattern):
            files_path.append(file_path)
            # Extract filename without extension
            base_name = os.path.basename(file_path)
            name_without_ext = os.path.splitext(base_name)[0]
            file_names.append(name_without_ext)
            
        return files_path, file_names
    
    @staticmethod
    def rand_perm_table_elements(table):
        """Randomly permutes the elements of a list"""
        if len(table) > 1:
            indices = torch.randperm(len(table))
            temp_table = [table[i] for i in indices]
            table = temp_table
        return table
    
    @staticmethod
    def memory_per_sample_image(img_size, data_type_num_bytes):
        """Computes memory (in MBs) required for one image"""
        total_size_on_mem = 0  # In MBs
        if len(img_size) == 3:
            total_size_on_mem = img_size[1] * img_size[2] * data_type_num_bytes / 1024 / 1024
        elif len(img_size) == 2:
            total_size_on_mem = img_size[0] * img_size[1] * data_type_num_bytes / 1024 / 1024
        else:
            total_size_on_mem = img_size * data_type_num_bytes / 1024 / 1024
        return total_size_on_mem
    
    @staticmethod
    def get_free_memory(ratio, max_memory=3000):
        """
        Calculates the current amount of free memory
        
        Args:
            ratio: Percentage [0-1] of free memory to reserve
            max_memory: Maximum memory to use in MB
            
        Returns:
            Free memory minus reserved amount in MBs
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
            free_mem = mem.available / 1024 / 1024  # Convert to MB
            leave_free_mem = free_mem * ratio
            return min(free_mem - leave_free_mem, max_memory)
        except ImportError:
            print("psutil not installed, returning max_memory")
            return max_memory
    
    @staticmethod
    def get_gpu_mem():
        """Get GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            cached = torch.cuda.memory_reserved() / 1024 / 1024
            return allocated, cached
        return 0, 0
    
    @staticmethod
    def obtain_data_path(benchmark, test_phase, lowest_size=False):
        """
        Returns paths to train, validation and test datasets
        
        Args:
            benchmark: Whether using benchmark dataset
            test_phase: Whether in test phase
            lowest_size: If True, return file with lowest size
            
        Returns:
            Paths to train, validation and test data
        """
        data_folder_path = os.path.join(
            os.getcwd(),
            'Data/benchmark/Datasets' if benchmark else 'Data/nonbenchmark/Datasets'
        )
        
        training_data_path = os.path.join(data_folder_path, 'train')
        validation_data_path = os.path.join(data_folder_path, 'validation')
        test_data_path = os.path.join(
            data_folder_path, 
            'validation' if benchmark else 'test'
        )
        
        train_data_files, _ = CommonFuncs.find_files(training_data_path, 'data')
        validation_data_files, _ = CommonFuncs.find_files(validation_data_path, 'data')
        test_data_files, _ = CommonFuncs.find_files(test_data_path, 'data')
        
        if test_phase and lowest_size:
            smallest_file = None
            smallest_size = float('inf')
            for file_path in test_data_files:
                size = os.path.getsize(file_path) / (1024**3)  # Size in GB
                if size < smallest_size:
                    smallest_size = size
                    smallest_file = file_path
            if smallest_file:
                test_data_files = [smallest_file]
                train_data_files = test_data_files
                validation_data_files = test_data_files
        
        return train_data_files, validation_data_files, test_data_files
    
    @staticmethod
    def sample_diagonal_mvn(mean, log_var, num_vectors):
        """
        Generates samples from diagonal multivariate normal distribution
        
        Args:
            mean: Mean vector [1 x num_dim]
            log_var: Log variance vector [1 x num_dim]
            num_vectors: Number of samples to generate
            
        Returns:
            Tensor of samples [num_vectors x num_dim]
        """
        if isinstance(mean, list):
            n_latents = mean[0].size(1)
        else:
            n_latents = mean.size(1)
        
        samples = torch.zeros(num_vectors, n_latents)
        
        for i in range(num_vectors):
            if isinstance(log_var, list):
                log_var_sample = log_var[0] + torch.randn(n_latents) * torch.exp(log_var[1] * 0.5)
            else:
                log_var_sample = log_var
            
            if isinstance(mean, list):
                mu = mean[0] + torch.randn(n_latents) * torch.exp(mean[1] * 0.5)
            else:
                mu = mean
            
            samples[i] = mu + torch.randn(n_latents) * torch.exp(log_var_sample * 0.5)
        
        return samples
    
    @staticmethod
    def interpolate_z_vectors(z_vector, target_z_vector, num_vectors):
        """
        Interpolates between two z vectors
        
        Args:
            z_vector: Starting vector [1 x num_dim]
            target_z_vector: Target vector [1 x num_dim]
            num_vectors: Number of interpolation steps
            
        Returns:
            Interpolated vectors [num_vectors x num_dim]
        """
        n_latents = z_vector.size(1)
        interpolated_z_vectors = torch.zeros(num_vectors, n_latents)
        
        for i in range(num_vectors):
            for j in range(n_latents):
                interpolated_z_vectors[i, j] = torch.linspace(
                    z_vector[0, j], target_z_vector[0, j], num_vectors
                )[i]
        
        return interpolated_z_vectors
    
    @staticmethod
    def normalize_minus_one_to_one(data, in_place=False):
        """Normalize data from [0, 1] to [-1, 1]"""
        if not in_place:
            data_temp = data.clone()
        else:
            data_temp = data
        data_temp = data_temp * 255 / 127 - 1
        return data_temp
    
    @staticmethod
    def normalize_back_to_zero_to_one(data, in_place=False):
        """Normalize data from [-1, 1] to [0, 1]"""
        if not in_place:
            data_temp = data.clone()
        else:
            data_temp = data
        data_temp = (data_temp + 1) * 127 / 255
        return data_temp
    
    @staticmethod
    def drop_input_vps(input_tensor, mark_input, dropout_net, num_drop_vps=None, 
                       drop_indices=None, single_vp_net=False, picked_vps=None, 
                       condition_hot_vec=None):
        """
        Randomly zeros-out viewpoints for DropoutNet or all except one for SingleVPNet
        """
        if isinstance(input_tensor, list):
            dropped_depth_tensor = input_tensor[0].clone()
            dropped_silhouettes_tensor = input_tensor[1].clone()
        else:
            dropped_depth_tensor = input_tensor.clone()
            dropped_silhouettes_tensor = None
        
        num_vps = dropped_depth_tensor.size(1)
        
        if dropout_net:
            for i in range(dropped_depth_tensor.size(0)):
                if num_drop_vps is None or num_drop_vps[0] == 0:
                    num_drop_vps = torch.tensor([random.randint(num_vps-5, num_vps-2)])
                if drop_indices is None:
                    drop_indices = torch.randperm(num_vps)
                
                for j in range(int(num_drop_vps[0])):
                    dropped_depth_tensor[i, int(drop_indices[j])] = 0
                    if dropped_silhouettes_tensor is not None:
                        dropped_silhouettes_tensor[i, int(drop_indices[j])] = 0
                    
                    if mark_input:
                        if isinstance(input_tensor, list):
                            input_tensor[0][i, int(drop_indices[j]), :20, :20] = 1
                            input_tensor[1][i, int(drop_indices[j]), :20, :20] = 1
                        else:
                            input_tensor[i, int(drop_indices[j]), :20, :20] = 1
        
        elif single_vp_net:
            if picked_vps is None:
                picked_vps = torch.tensor([random.randint(0, num_vps-1)])
            
            temp_depth_vp = torch.zeros(
                dropped_depth_tensor.size(0), 1, 
                dropped_depth_tensor.size(2), 
                dropped_depth_tensor.size(3)
            ).type(dropped_depth_tensor.type())
            
            for i in range(dropped_depth_tensor.size(0)):
                pick_vp = int(picked_vps[0 if len(picked_vps) == 1 else i])
                temp_depth_vp[i, 0] = dropped_depth_tensor[i, pick_vp].clone()
            
            dropped_depth_tensor = temp_depth_vp
            
            if dropped_silhouettes_tensor is not None:
                temp_mask_vp = torch.zeros_like(dropped_depth_tensor)
                for i in range(temp_mask_vp.size(0)):
                    pick_vp = int(picked_vps[0 if len(picked_vps) == 1 else i])
                    temp_mask_vp[i, 0] = dropped_silhouettes_tensor[i, pick_vp].clone()
                dropped_silhouettes_tensor = temp_mask_vp
        
        if isinstance(input_tensor, list):
            if condition_hot_vec is not None:
                return [dropped_depth_tensor, dropped_silhouettes_tensor, condition_hot_vec]
            else:
                return [dropped_depth_tensor, dropped_silhouettes_tensor]
        else:
            if condition_hot_vec is not None:
                return [dropped_depth_tensor, condition_hot_vec]
            else:
                return dropped_depth_tensor
    
    @staticmethod
    def compute_classification_accuracy(predicted_scores, target_class_vec=None, 
                                       return_hot_vec=False, num_cats=None):
        """Computes classification accuracy"""
        pred_scores = predicted_scores.clone().float()
        
        softmax = nn.Softmax(dim=1)
        output = softmax(pred_scores)
        _, idx = output.topk(1, dim=1)
        
        if not return_hot_vec and target_class_vec is not None:
            target_class = target_class_vec.clone().float()
            return (idx.float().view(-1) == target_class.float()).sum().item()
        else:
            idx = idx.view(pred_scores.size(0))
            target_class_hot_vec = torch.zeros(pred_scores.size(0), num_cats)
            for i in range(pred_scores.size(0)):
                target_class_hot_vec[i, idx[i]] = 1
            return target_class_hot_vec.cuda() if torch.cuda.is_available() else target_class_hot_vec
    
    @staticmethod
    def generate_batch_indices(num_data_points, batch_size):
        """Creates batch indices for extracting batches of data"""
        perm = torch.randperm(num_data_points).long()
        indices = torch.split(perm, batch_size)
        
        if len(indices) > 1:
            last_batch_size = indices[-1].size(0)
            if last_batch_size > 1:
                indices = list(indices)
            else:
                indices = list(indices[:-1])
        
        return indices
