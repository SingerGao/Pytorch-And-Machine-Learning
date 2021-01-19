import os
import random
import torch
import numpy as np  
import matplotlib.pyplot as plt
from PIL import Image  
from torch.utils.data import Dataset,DataLoader 
from torchvision import transforms, utils


class OXFlowerDataset(Dataset):
    """OX Flower 17 dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.len = len(open(os.path.join(root_dir, 'files.txt'),'r').readlines())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        label = random.randint(0, 16)
        imgs_folder = os.path.join(self.root_dir,
                                '{}'.format(label))
        img_list = os.listdir(imgs_folder)
        img_name = random.choice(img_list)
        img_path = os.path.join(imgs_folder, img_name)
        img = Image.open(img_path)
        img = img.resize((128, 128))

        if self.transform:
            t_img = self.transform(img)
        else:
            np_img = np.array(img)/256.0
            np_img = np_img.transpose((2,0,1))
            t_img = torch.tensor(np_img, dtype=torch.float)
        img.close()
        return (t_img, label)


if __name__ == '__main__':
    dataset_path='./flower_dataset/'
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), #将图片转换为Tensor
        #transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])
    ox_flower_dataset = OXFlowerDataset(root_dir='./flower_dataset', 
                                    transform=transform)

    dataloader = DataLoader(ox_flower_dataset, 
                            batch_size=64,
                            shuffle=True, 
                            num_workers=8)

    for idx, (imgs, labels) in enumerate(dataloader):
        print(imgs, labels)
