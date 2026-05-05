import yaml
import os
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log_dir', type=str, default="workspace/mae_mask-loss")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--eval_freq', type=int, default=2000)
parser.add_argument('--ckpt_freq', type=int, default=10000)

parser.add_argument('--white_bg_color', type=int, default=0)
parser.add_argument('--mask_pho_loss', type=int, default=0)  # compute loss only in masked region
parser.add_argument('--aug_noise', type=int, default=0)  # data augmentation with noise added

parser.add_argument('--data_root', type=str, default="data/render_demo_split")

parser.add_argument('--use_normal', type=int, default=0)
parser.add_argument('--w_normal_loss', type=float, default=0.1)

parser.add_argument('--use_shadow_mask', type=int, default=0)
parser.add_argument('--w_shadow_mask', type=float, default=0.5)

parser.add_argument('--use_conf', type=int, default=0)
parser.add_argument('--conf_alpha', type=float, default=0.2)

parser.add_argument('--ddp_port', type=str, default="12355")
# parser.add_argument('--use_mask_loss', type=int, default=0)
opt, _ = parser.parse_known_args()


import random
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import lpips
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt


def conf_to_tensor(conf):
    '''
    input: [b,1,h,w]
    output: [b,3,h,w]
    '''
    device = conf.device
    conf = conf.detach().cpu().permute(0, 2, 3, 1).numpy()[..., 0]
    confs = [conf[i] for i in range(conf.shape[0])]
    cmap = plt.get_cmap('jet')
    # confs_max = max([d.max() for d in confs])
    confs = [cmap(d / d.max()) for d in confs]
    conf_torch = []
    for c in confs:
        conf_torch.append(
            torch.from_numpy(c).float()
        )
    conf_torch = torch.stack(conf_torch, dim=0).to(device)
    conf_torch = conf_torch.permute(0, 3, 1, 2)
    return conf_torch[:, :3]


class Trainer:
    def __init__(self, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.batch_size = opt.batch_size
        # self.use_mask_loss = opt.use_mask_loss == 1
        self.use_conf = opt.use_conf == 1
        self.use_shadow_mask = opt.use_shadow_mask == 1
        self.use_normal = opt.use_normal == 1

        self.is_main_process = self.rank == 0

        self.create_dataset()
        self.create_model()
        self.create_optimizer()
        self.scaler = GradScaler()

        self.lpips_loss = lpips.LPIPS(net="vgg").to(self.device)
        self.lpips_loss.requires_grad_(False)
        
        # initialize logger (created only in main process)
        self.log_dir = opt.log_dir
        if self.is_main_process:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)
        
            args_dict = vars(opt)
            with open(os.path.join(self.log_dir, "opt.txt"), "w") as f:
                f.write("python ttrain_enhancer.py \\\n")
                for k in args_dict.keys():
                    f.write(f"    --{k} {args_dict[k]} \\\n")
    
    def create_dataset(self):
        data_root = opt.data_root
        from dataset.detail_enhance_dataset import SyntheticDataset
        self.train_dataset = SyntheticDataset(
            data_root=data_root,
            split="train",
            opt=opt,
        )
        self.val_dataset = SyntheticDataset(
            data_root=data_root,
            split="val",
            opt=opt,
        )
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            pin_memory=True,
        )

        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            pin_memory=True,
        )

    def create_model(self):
        from model.detail_enhance_unet import UNet_Enhancer

        self.model = UNet_Enhancer(
            n_channels=6,
            n_classes=3,
            output_confidence=self.use_conf,
            output_shadow_mask=self.use_shadow_mask,
            output_normal=self.use_normal,
        ).to(self.device)
        self.model = DDP(self.model, device_ids=[self.device], find_unused_parameters=False)

    def create_optimizer(self):
        params = [
            {"params": self.model.module.parameters(), "lr": 1e-4},
        ]
        self.optimizer = torch.optim.AdamW(params)
    
    def train(self):
        num_epoch = 1000
        num_iter = 0
        eval_freq = opt.eval_freq
        
        for epoch in range(num_epoch):
            # set sampler epoch to ensure different shuffling per epoch
            self.train_sampler.set_epoch(epoch)
            
            for data in self.train_loader:
                num_iter += 1
                imgs = data["img"].to(self.device)
                gts = data["gt"].to(self.device)
                masks = data["mask"].to(self.device)
                normals = data["normal"].to(self.device)
                degrade_imgs = data["degrade_img"].to(self.device)

                with autocast():
                    out_dict = self.model(
                        torch.cat([degrade_imgs, imgs], dim=1)
                    )

                    preds = out_dict["diffuse"]
                    loss_l1 = F.l1_loss(preds, gts)
                    
                    # loss_l1 = F.l1_loss(preds, gts)
                    loss_lpips = self.lpips_loss(preds, gts, normalize=True).mean()
                    loss = loss_l1 + 0.1 * loss_lpips

                    loss_dict = {}
                    loss_dict["lpips"] = loss_lpips.item()
                    loss_dict["l1"] = loss_l1.item()

                if self.is_main_process:
                    for k in loss_dict.keys():
                        self.writer.add_scalar('Train/Loss_' + k, loss_dict[k], num_iter)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
                # checkpointing and evaluation
                if num_iter % opt.ckpt_freq == 0:
                    if self.is_main_process:
                        torch.save(self.model.module.state_dict(), os.path.join(self.log_dir, "iter_%08d.pth" % num_iter))
                
                if num_iter % eval_freq == 0:
                    # ensure save directory exists
                    save_dir = os.path.join(self.log_dir, "iter_%08d" % num_iter)
                    if self.is_main_process:
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(self.model.module.state_dict(), os.path.join(self.log_dir, "latest.pth"))
                    dist.barrier()
                    
                    loss_total_l1 = 0
                    loss_total_lpips = 0

                    self.val_sampler.set_epoch(0)

                    self.model.eval()

                    # TODO need to refactor the visualization code
                    for val_iter, data in enumerate(self.val_loader):
                        imgs = data["img"].to(self.device)
                        gts = data["gt"].to(self.device)
                        masks = data["mask"].to(self.device)
                        normals = data["normal"].to(self.device)
                        degrade_imgs = data["degrade_img"].to(self.device)

                        with autocast():
                            with torch.no_grad():
                                out_dict = self.model(
                                    torch.cat([degrade_imgs, imgs], dim=1)
                                )
                                # preds = preds * masks
                                preds = out_dict["diffuse"]
                                if self.use_conf:
                                    conf = out_dict["confidence"] * masks
                                    conf = conf_to_tensor(conf)

                                loss_l1 = F.l1_loss(preds, gts)
                                loss_lpips = self.lpips_loss(preds, gts, normalize=True).mean()
                                loss_total_l1 += loss_l1.item() * imgs.shape[0]
                                loss_total_lpips += loss_lpips.item() * imgs.shape[0]
                        
                        vis = torch.cat([imgs, degrade_imgs, preds, gts], dim=0)
                                                
                        save_image(
                            vis,
                            os.path.join(save_dir, "rank_%d_%05d.png" % (self.rank, val_iter)),
                            nrow=imgs.shape[0],
                        )

                    self.model.train()

                    total_val_samples = len(self.val_dataset)

                    loss_total_l1_tensor = torch.tensor(loss_total_l1, device=self.device)
                    loss_total_lpips_tensor = torch.tensor(loss_total_lpips, device=self.device)

                    dist.all_reduce(loss_total_l1_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(loss_total_lpips_tensor, op=dist.ReduceOp.SUM)

                    loss_avg_l1 = loss_total_l1_tensor.item() / total_val_samples
                    loss_avg_lpips = loss_total_lpips_tensor.item() / total_val_samples

                    if self.is_main_process:
                        self.writer.add_scalar('Val/Loss L1', loss_avg_l1, num_iter)
                        self.writer.add_scalar('Val/Loss LPIPS', loss_avg_lpips, num_iter)

        # cleanup distributed environment
        dist.destroy_process_group()


def setup(rank, world_size):
    """Initialize distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = opt.ddp_port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size):
    """Main function executed by each process"""
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    trainer = Trainer(rank, world_size, device)
    trainer.train()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training")
    
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    