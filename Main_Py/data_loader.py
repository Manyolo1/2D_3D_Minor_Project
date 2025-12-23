import os
import zipfile
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from common_funcs import CommonFuncs


class DataLoader:
    """
    Data loader for 3D shape datasets
    Processes raw data from zip files and creates PyTorch tensors
    """
    
    def __init__(self, opt):
        self.opt = opt
        self.common_funcs = CommonFuncs()
        
        # Set default tensor type
        if opt.global_data_type == 'float':
            torch.set_default_dtype(torch.float32)
            self.data_type_num_bytes = 4
        elif opt.global_data_type == 'double':
            torch.set_default_dtype(torch.float64)
            self.data_type_num_bytes = 8
        
        # Validate resize scale
        if opt.resize_scale < 0 or opt.resize_scale > 1:
            opt.resize_scale = 1
        
        # Normalize train/valid/test splits
        p_sum = opt.p_train + opt.p_valid + opt.p_test
        if p_sum != 1:
            opt.p_train /= p_sum
            opt.p_valid /= p_sum
            opt.p_test /= p_sum
        
        # Determine file extension and lookup word
        if opt.raw_data_type == 'float':
            self.folder_lookup_word = 'depth_float'
            self.file_extension = '.txt'
        elif opt.raw_data_type == 'int':
            self.folder_lookup_word = 'depth_rgb'
            self.file_extension = '.png'
        else:
            raise ValueError("raw_data_type must be 'float' or 'int'")
    
    def get_file_names_from_zip(self, zip_file_path):
        """Returns all file names in a .zip file without uncompressing"""
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            return zip_ref.namelist()
    
    def separate_file_and_folder_names(self, all_file_names):
        """Separates file and folder names from zip contents"""
        files = []
        folders = []
        
        for name in all_file_names:
            if name.endswith('/'):
                folders.append(name)
            elif self.file_extension in name:
                files.append(os.path.basename(name))
        
        return folders, files
    
    def unzip_folder(self, zip_file_path, folder_names, data_folder_path):
        """Extracts specific folder from zip file"""
        flag = False
        unzipped_path = None
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for folder_name in folder_names:
                if self.folder_lookup_word in folder_name:
                    target_path = os.path.join(data_folder_path, folder_name)
                    
                    if not os.path.exists(target_path):
                        print(f"Extracting {folder_name} from {zip_file_path}...")
                        zip_ref.extractall(data_folder_path, 
                                         [m for m in zip_ref.namelist() 
                                          if m.startswith(folder_name)])
                    
                    flag = True
                    unzipped_path = target_path
                    break
        
        return flag, unzipped_path
    
    def divide_data_for_viewpoints(self, data, p_train, p_valid, p_test):
        """
        Divides data into train/validation/test sets and organizes by viewpoints
        
        Args:
            data: List of dicts with 'path', 'files', 'label' keys
            p_train, p_valid, p_test: Split ratios
            
        Returns:
            viewpoints_data_path: Paths organized by split and viewpoint
            labels: Corresponding labels
        """
        # Determine number of viewpoints
        first_file = data[0]['files'][0]
        num_viewpoints = 1
        base_name = first_file.split('_Cam_')[0] if '_Cam_' in first_file else first_file.split('_')[0]
        
        for i, fname in enumerate(data[0]['files'][1:], 1):
            current_base = fname.split('_Cam_')[0] if '_Cam_' in fname else fname.split('_')[0]
            if current_base == base_name:
                num_viewpoints += 1
            else:
                break
        
        # Create viewpoint patterns
        viewpoint_patterns = []
        viewpoints_data_path = [[] for _ in range(num_viewpoints)]
        labels = [[] for _ in range(num_viewpoints)]
        
        for i in range(num_viewpoints):
            viewpoint_patterns.append(f'_Cam_{i}' if '_Cam_' in first_file else f'_{i}')
        
        # Organize files by viewpoint
        for class_idx, class_data in enumerate(data):
            for vp_idx, pattern in enumerate(viewpoint_patterns):
                for fname in class_data['files']:
                    if pattern in fname:
                        full_path = os.path.join(class_data['path'], fname)
                        viewpoints_data_path[vp_idx].append(full_path)
                        labels[vp_idx].append(class_idx + 1)  # 1-indexed labels
        
        # Random permutation
        num_images = len(viewpoints_data_path[0])
        perm_indices = torch.randperm(num_images).tolist()
        
        temp_vp_data = [[] for _ in range(num_viewpoints)]
        temp_labels = [[] for _ in range(num_viewpoints)]
        
        for vp_idx in range(num_viewpoints):
            for perm_idx in perm_indices:
                temp_vp_data[vp_idx].append(viewpoints_data_path[vp_idx][perm_idx])
                temp_labels[vp_idx].append(labels[vp_idx][perm_idx])
        
        viewpoints_data_path = temp_vp_data
        labels = temp_labels
        
        # Split into train/valid/test
        num_samples = len(viewpoints_data_path[0])
        num_train = int(num_samples * (p_valid if p_valid == 1 else p_train))
        num_valid = int(num_samples * (0 if p_valid == 1 else p_valid))
        num_test = num_samples - num_train - num_valid
        
        splits = [num_train, num_valid, num_test]
        split_data = [[] for _ in range(3 if p_test != 0 else 1)]
        split_labels = [[] for _ in range(3 if p_test != 0 else 1)]
        
        start_idx = 0
        for split_idx, split_size in enumerate(splits[:len(split_data)]):
            end_idx = start_idx + split_size
            
            for vp_idx in range(num_viewpoints):
                vp_split_data = []
                vp_split_labels = []
                
                for sample_idx in range(start_idx, end_idx):
                    vp_split_data.append(viewpoints_data_path[vp_idx][sample_idx])
                    vp_split_labels.append(labels[vp_idx][sample_idx])
                
                split_data[split_idx].append(vp_split_data)
                split_labels[split_idx].append(vp_split_labels)
            
            start_idx = end_idx
        
        # Reorganize by 3D model (all viewpoints together)
        final_data = []
        final_labels = []
        
        for split_idx in range(len(split_data)):
            num_models = len(split_data[split_idx][0])
            
            split_models = []
            split_model_labels = []
            
            for model_idx in range(num_models):
                model_viewpoints = []
                for vp_idx in range(len(split_data[split_idx])):
                    model_viewpoints.append(split_data[split_idx][vp_idx][model_idx])
                
                split_models.append(model_viewpoints)
                split_model_labels.append(split_labels[split_idx][0][model_idx])
            
            final_data.append(split_models)
            final_labels.append(split_model_labels)
        
        return final_data, final_labels
    
    def load_txt_into_tensor(self, path):
        """Loads a .txt file into a PyTorch tensor"""
        with open(path, 'r') as f:
            lines = f.readlines()
        
        height = len(lines)
        width = len(lines[0].strip().split())
        
        tensor = torch.zeros(1, height, width)
        
        for i, line in enumerate(lines):
            values = [float(v) for v in line.strip().split()]
            tensor[0, i, :len(values)] = torch.tensor(values)
        
        return tensor
    
    def load_image(self, path, channels=1):
        """Load image as tensor"""
        img = Image.open(path)
        if channels == 1:
            img = img.convert('L')
        
        transform = transforms.ToTensor()
        return transform(img).unsqueeze(0)
    
    def process_and_save_data(self):
        """Main method to process raw data and save as PyTorch tensors"""
        print("=" * 80)
        print("Loading Data Into Memory and Storing on Disk for Training")
        print("=" * 80)
        
        raw_data_folder = 'Data/benchmark' if self.opt.benchmark else 'Data/nonbenchmark'
        data_folder_path = os.path.join(os.getcwd(), raw_data_folder)
        
        # Find and process zip files or existing folders
        data = []
        
        if self.opt.from_scratch and self.opt.zip:
            zip_files, class_labels = self.common_funcs.find_files(data_folder_path, 'zip')
            
            for i, zip_file in enumerate(zip_files):
                print(f"Processing {i+1}/{len(zip_files)}: {zip_file}")
                
                all_file_names = self.get_file_names_from_zip(zip_file)
                folders, files = self.separate_file_and_folder_names(all_file_names)
                
                flag, unzipped_path = self.unzip_folder(zip_file, folders, data_folder_path)
                
                if flag:
                    data.append({
                        'path': unzipped_path,
                        'files': files,
                        'label': class_labels[i]
                    })
        else:
            # Look for already extracted folders
            for item in os.listdir(data_folder_path):
                if self.folder_lookup_word in item:
                    item_path = os.path.join(data_folder_path, item)
                    if os.path.isdir(item_path):
                        files = [f for f in os.listdir(item_path) 
                                if self.file_extension in f]
                        label = item.split('_')[0]
                        
                        data.append({
                            'path': item_path,
                            'files': files,
                            'label': label
                        })
        
        if not data:
            print("No data found to process!")
            return
        
        print(f"Found {len(data)} dataset(s) to process")
        
        # Get category names
        object_categories = [d['label'] for d in data]
        
        # Divide data
        viewpoints_data, labels = self.divide_data_for_viewpoints(
            data, self.opt.p_train, self.opt.p_valid, self.opt.p_test
        )
        
        # Save processed data
        self._save_processed_data(viewpoints_data, labels, object_categories, data_folder_path)
    
    def _save_processed_data(self, viewpoints_data, labels, categories, storage_path):
        """Save processed tensors to disk.

        Expects:
            viewpoints_data: list over splits [split][model][vp] -> filepath
            labels: list over splits [split][model] -> int label (1-indexed)
            categories: list of class names (strings)
            storage_path: root data folder (e.g., Data/benchmark or Data/nonbenchmark)

        Saves for each split a single .data file with keys:
            'dataset': FloatTensor [N, num_vps, H, W]
            'labels': LongTensor [N]
            'category': list[str]
        into: <storage_path>/Datasets/<split>/data_0.data
        """
        print("Saving processed data...")
        import torchvision.transforms as T
        import torch.nn.functional as F

        splits_names = ["train", "validation", "test"]

        # Create destination directories
        datasets_root = os.path.join(storage_path, "Datasets")
        os.makedirs(datasets_root, exist_ok=True)
        for split_name in splits_names[: len(viewpoints_data)]:
            os.makedirs(os.path.join(datasets_root, split_name), exist_ok=True)

        # Helper: load one depth map as tensor [1, H, W] and resize to img_size
        def load_depth(path):
            if self.file_extension == ".png":
                img = Image.open(path).convert("L")
                resize = T.Resize((self.opt.img_size, self.opt.img_size), interpolation=Image.BILINEAR)
                tensor = T.ToTensor()(resize(img))  # [1, H, W] in [0,1]
                return tensor
            else:  # .txt float grid
                t = self.load_txt_into_tensor(path)  # [1, h, w]
                # Resize with bilinear to (img_size, img_size)
                t = F.interpolate(t.unsqueeze(0), size=(self.opt.img_size, self.opt.img_size), mode="bilinear", align_corners=False).squeeze(0)
                return t

        for split_idx, split_models in enumerate(viewpoints_data):
            split_name = splits_names[split_idx]
            split_labels = labels[split_idx]

            if len(split_models) == 0:
                # Nothing to save for this split
                continue

            num_models = len(split_models)
            num_vps = len(split_models[0])  # number of viewpoints per model

            # Allocate tensor [N, V, H, W]
            dataset = torch.zeros(
                (num_models, num_vps, self.opt.img_size, self.opt.img_size), dtype=torch.float32
            )
            label_tensor = torch.zeros((num_models,), dtype=torch.long)

            for m_idx, vp_paths in enumerate(split_models):
                # vp_paths: list of file paths for each viewpoint for this model
                for v_idx, fp in enumerate(vp_paths):
                    dataset[m_idx, v_idx] = load_depth(fp)
                label_tensor[m_idx] = int(split_labels[m_idx])  # keep 1-indexed as upstream code expects

            save_dict = {
                "dataset": dataset,
                "labels": label_tensor,
                "category": categories,
            }

            out_dir = os.path.join(datasets_root, split_name)
            out_path = os.path.join(out_dir, "data_0.data")
            torch.save(save_dict, out_path)
            print(f"Saved {split_name}: {num_models} samples, {num_vps} vps -> {out_path}")

  
