import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class EfficientPhysDataset(Dataset):
    def __init__(self, video_data, label_data):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.video_data = video_data  # N, H, W, C
        self.label_data = label_data.reshape(-1, 1)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        video_data = torch.tensor(np.transpose(self.video_data[index], (2, 0, 1)), dtype=torch.float32)
        label_data = torch.tensor(self.label_data[index], dtype=torch.float32)

        if torch.cuda.is_available():
            video_data = video_data.to('cuda')
            label_data = label_data.to('cuda')

        return video_data, label_data

    def __len__(self):
        return len(self.label_data)
