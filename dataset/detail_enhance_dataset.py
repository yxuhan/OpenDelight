import os
from PIL import Image
import glob
import torch
from torchvision import transforms
import random
from torchvision.utils import save_image
import kornia


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
        shadow_mask_pth = img_pth.replace("align_image", "align_shadow_mask")
        normal_pth = img_pth.replace("align_image", "align_normal")

        img = self.load_image(img_pth)
        
        mask = self.load_image(mask_pth)
        gt = self.load_image(diff_pth)
        shadow_mask = self.load_image(shadow_mask_pth)
        normal = self.load_image(normal_pth)

        if random.random() > 0.5 and self.split == "train":
            img = torch.flip(img, [-1])
            gt = torch.flip(gt, [-1])
            mask = torch.flip(mask, [-1])
            shadow_mask = torch.flip(shadow_mask, [-1])
            normal = torch.flip(normal, [-1])
            normal[0] = 1 - normal[0]
        
        if True:
            degrade_img = torch.clone(gt)
            if random.random() > 0.5:  # gaussian blur degradation
                ks = random.choice([3., 5.])
                std = random.choice([1/3, 1/4, 1/5])
                degrade_img = kornia.filters.gaussian_blur2d(
                    degrade_img[None, ...], (int(ks), int(ks)), (ks*std, ks*std)
                )[0]
            else:
                scale = random.choice([2, 4])
                degrade_img = torch.nn.functional.interpolate(
                    degrade_img[None, ...], 
                    scale_factor=1/scale, 
                    mode='bilinear', 
                    align_corners=False,
                    antialias=True,
                )[0]
                degrade_img = torch.nn.functional.interpolate(
                    degrade_img[None, ...], 
                    size=(512, 512), 
                    mode='bilinear', 
                    align_corners=False,
                    antialias=True,
                )[0]

            # add noise
            noise_scale = random.uniform(0.0, 0.03)
            noise = torch.randn_like(degrade_img) * noise_scale
            degrade_img = degrade_img + noise

        if self.use_white_bg_color:
            img = img * mask + (1 - mask)
            gt = gt * mask + (1 - mask)
            shadow_mask = shadow_mask * mask + (1 - mask)
            normal = normal * mask + (1 - mask)
            degrade_img = degrade_img * mask + (1 - mask)
        else:
            img = img * mask
            gt = gt * mask
            shadow_mask = shadow_mask * mask
            normal = normal * mask
            degrade_img = degrade_img * mask

        return {
            "img": img,
            "gt": gt,
            "mask": mask,
            "shadow_mask": shadow_mask[:1],
            "normal": normal,
            "degrade_img": degrade_img,
        }
