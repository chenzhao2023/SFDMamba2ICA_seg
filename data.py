import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

def resize_image(path, size=(512, 512)):
    img = Image.open(path).convert('L')
    img = img.resize(size)
    return img

class MyDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.names = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.img_dir, name)
        label_path = os.path.join(self.label_dir, name)

        img = resize_image(img_path)
        label = resize_image(label_path)

        img = transform(img)
        label = transform(label)

        return img, label, name

class TargetDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.names = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.img_dir, name)
        label_path = os.path.join(self.label_dir, name)

        img = resize_image(img_path)
        label = resize_image(label_path)

        img = transform(img)
        label = transform(label)

        return img, label, name

class PairedDataset(Dataset):
    def __init__(self, source_img_dir, source_label_dir,
                 target_img_dir, target_label_dir):
        self.source_dataset = MyDataset(source_img_dir, source_label_dir)
        self.target_dataset = MyDataset(target_img_dir, target_label_dir)

    def __len__(self):
        return min(len(self.source_dataset), len(self.target_dataset))

    def __getitem__(self, idx):
        img_s, mask_s, name_s = self.source_dataset[idx]
        img_t, mask_t, name_t = self.target_dataset[idx]
        return img_s, mask_s, name_s, img_t, mask_t, name_t

if __name__ == '__main__':
    source_dataset = MyDataset("dataset/source/images", "dataset/source/masks")
    print("Source Dataset Length:", len(source_dataset))
    img_s, mask_s, name_s = source_dataset[0]
    print("Source:", name_s, img_s.shape, mask_s.shape)

    target_dataset = TargetDataset("dataset/test/images", "dataset/test/masks")
    print("Target Dataset Length:", len(target_dataset))
    img_t, mask_t, name_t = target_dataset[0]
    print("Target:", name_t, img_t.shape, mask_t.shape)

    paired_dataset = PairedDataset("dataset/source/images", "dataset/source/masks",
                                   "dataset/target/images", "dataset/target/masks")
    print("Paired Dataset Length:", len(paired_dataset))
    img_s, mask_s, name_s, img_t, mask_t, name_t = paired_dataset[0]
    print("Paired Source:", name_s, img_s.shape, mask_s.shape)
    print("Paired Target:", name_t, img_t.shape, mask_t.shape)
