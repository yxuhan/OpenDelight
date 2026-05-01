'''
mix FaceOLAT and Syhthetic dataset
'''


import os
from PIL import Image
import glob
import torch
from torchvision import transforms
import random
from torchvision.utils import save_image
import random


class MixDataset(torch.utils.data.Dataset):
    def __init__(self, opt, split="train"):
        self.syn_data_root = opt.syn_data_root
        self.real_data_root = opt.real_data_root
        self.syn_prob = opt.syn_prob
        self.split = split

        self.real_dataset = FaceOLATDataset(
            data_root=self.real_data_root, opt=opt, split=split
        )
        self.syn_dataset = SyntheticDataset(
            data_root=self.syn_data_root, opt=opt, split=split
        )

    def __len__(self,):
        return len(self.real_dataset) + len(self.syn_dataset)

    def __getitem__(self, idx):
        if self.split == "train":
            if random.random() < self.syn_prob:
                return self.syn_dataset[random.randint(0, len(self.syn_dataset) - 1)]
            else:
                return self.real_dataset[random.randint(0, len(self.real_dataset) - 1)]
        else:
            if idx < len(self.real_dataset):
                return self.real_dataset[idx]
            else:
                return self.syn_dataset[idx - len(self.real_dataset)]


class FaceOLATDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, opt, split="train"):
        self.data_root = os.path.join(data_root, split)
        self.img_pth_list = sorted(glob.glob(
            os.path.join(self.data_root, "align_image", "*", "*", "*.png")
        ))
        self.split = split
        self.use_white_bg_color = opt.white_bg_color == 1

    def __len__(self):
        return len(self.img_pth_list)

    def load_image(self, pth):
        img = transforms.ToTensor()(
            Image.open(pth)
        )
        return img

    def __getitem__(self, idx):
        img_pth = self.img_pth_list[idx]
        
        img_root = os.path.dirname(img_pth)
        cam_id = os.path.basename(img_root)
        img_root = os.path.dirname(img_root)
        sub_id = os.path.basename(img_root)

        mask_pth = os.path.join(self.data_root, "align_mask_face", "%s_%s.png" % (sub_id, cam_id))
        diff_pth = os.path.join(self.data_root, "align_diffuse", "%s_%s.png" % (sub_id, cam_id))

        img = self.load_image(img_pth)
        mask = self.load_image(mask_pth)
        gt = self.load_image(diff_pth)

        if self.use_white_bg_color:
            img = img * mask + (1 - mask)
            gt = gt * mask + (1 - mask)
        else:
            img = img * mask
            gt = gt * mask

        if random.random() > 0.5 and self.split == "train":
            img = torch.flip(img, [-1])
            gt = torch.flip(gt, [-1])
            mask = torch.flip(mask, [-1])

        return {
            "img": img,
            "gt": gt,
            "mask": mask,
            "data_source": 1,
        }


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, opt, split="train"):
        self.data_root = os.path.join(data_root, split)
        self.img_pth_list = sorted(glob.glob(
            os.path.join(self.data_root, "*", "align_image", "*.png")
        ))
        self.split = split
        self.use_white_bg_color = opt.white_bg_color == 1
        self.use_aug_nosie = opt.aug_noise == 1

    def __len__(self):
        return len(self.img_pth_list)

    def load_image(self, pth):
        if os.path.exists(pth):
            img = transforms.ToTensor()(
                Image.open(pth)
            )
        else:
            return torch.zeros(3, 512, 512)
        return img

    def __getitem__(self, idx):
        img_pth = self.img_pth_list[idx]
        mask_pth = img_pth.replace("align_image", "mask")
        diff_pth = img_pth.replace("align_image", "align_diffuse")
        # shadow_mask_pth = img_pth.replace("align_image", "align_shadow_mask")
        # normal_pth = img_pth.replace("align_image", "align_normal")

        img = self.load_image(img_pth)
        # # add noise augmentation
        if self.use_aug_nosie == 1:
            if self.split == "train" or idx % 2 == 0:
                noise_scale = random.uniform(0.0, 0.03)
                noise = torch.randn_like(img) * noise_scale
                img = img + noise
        
        mask = self.load_image(mask_pth)
        gt = self.load_image(diff_pth)
        # shadow_mask = self.load_image(shadow_mask_pth)
        # normal = self.load_image(normal_pth)

        if random.random() > 0.5 and self.split == "train":
            img = torch.flip(img, [-1])
            gt = torch.flip(gt, [-1])
            mask = torch.flip(mask, [-1])
            # shadow_mask = torch.flip(shadow_mask, [-1])
            # normal = torch.flip(normal, [-1])
            # normal[0] = 1 - normal[0]

        if self.use_white_bg_color:
            img = img * mask + (1 - mask)
            gt = gt * mask + (1 - mask)
            # shadow_mask = shadow_mask * mask + (1 - mask)
            # normal = normal * mask + (1 - mask)
        else:
            img = img * mask
            gt = gt * mask
            # shadow_mask = shadow_mask * mask
            # normal = normal * mask

        return {
            "img": img,
            "gt": gt,
            "mask": mask,
            "data_source": 0,
            # "shadow_mask": shadow_mask[:1],
            # "normal": normal,
        }
