import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize

class PMDataset(Dataset):
    def __init__(self, data_path, transform=None, dataset_type='train', limit_samples=None):
        super(PMDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform
        self.image_files = []
        self.label_files = []
        self.limit_samples = limit_samples

        if dataset_type == 'train':
            image_dir = 'Train/Data'
            label_dir = 'Train/Mask'
        elif dataset_type == 'test':
            image_dir = 'Test/Data'
            label_dir = 'Test/Mask'
        else:
            raise ValueError(f'Invalid dataset type: {dataset_type}')

        print(os.path.join(data_path, image_dir))
        for root, dirs, files in os.walk(os.path.join(data_path, image_dir)):
            for file in sorted(files):
                if file.endswith('.npy'):
                    self.image_files.append(os.path.join(root, file))

        for root, dirs, files in os.walk(os.path.join(data_path, label_dir)):
            for file in sorted(files):
                if file.endswith('.npy'):
                    self.label_files.append(os.path.join(root, file))

        assert(len(self.image_files) == len(self.label_files))

        print(f"The total number of images in {dataset_type}: {len(self.image_files)}")
        print(f"The total number of labels in {dataset_type}: {len(self.label_files)}")

    def __len__(self):
        if self.limit_samples is not None:
            return min(self.limit_samples, len(self.image_files))
        return len(self.image_files)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = os.path.join(self.image_files[idx])
        label_path = os.path.join(self.label_files[idx])

        image = np.load(image_path)
        label = np.load(label_path)

        assert not np.any(np.isnan(image))
        assert not np.any(np.isnan(label))
        

        image = resize(image, (512, 512), mode='reflect', anti_aliasing=True)

        label = np.where(label == 100, 1, 0).astype(np.float32)

        label = resize(label, (512, 512), mode='reflect', order=0, anti_aliasing=False)
        
        #label = np.where(label == 100, 1, 0).astype(np.float32)
        

        image = np.repeat(image[..., np.newaxis], 3, axis=-1)
        #m, s = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))

        if self.transform:
            

            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def get_path(self, index: int):
        return self.image_files[index], self.label_files[index]