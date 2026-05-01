import yaml
import os
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config_path', type=str, default="config/delight_residual_mae_sapiens_512_mix.yaml")
parser.add_argument('--log_dir', type=str, default="workspace/debug")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--eval_freq', type=int, default=2000)
parser.add_argument('--ckpt_freq', type=int, default=10000)
parser.add_argument('--white_bg_color', type=int, default=0)
parser.add_argument('--aug_noise', type=int, default=0)
parser.add_argument('--syn_data_root', type=str, default="/root/autodl-tmp/OpenDelight/Network/data/merge_dataset_split")
parser.add_argument('--real_data_root', type=str, default="/root/autodl-tmp/OpenDelight/FaceOLAT/data/processed_14view_12light")
parser.add_argument('--syn_prob', type=float, default=0.5)
parser.add_argument('--ddp_port', type=str, default="12355")
parser.add_argument('--init_ckpt_path', type=str, default="xxx")
parser.add_argument('--grad_ckpt', type=float, default=1)
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


class Trainer:
    def __init__(self, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.batch_size = opt.batch_size
        # self.use_mask_loss = opt.use_mask_loss == 1

        # Only the main process records logs and saves
        self.is_main_process = self.rank == 0

        self.create_dataset()
        self.create_model()
        self.create_optimizer()
        self.scaler = GradScaler()

        self.lpips_loss = lpips.LPIPS(net="vgg").to(self.device)
        self.lpips_loss.requires_grad_(False)
        
        # Initialize logger (created only in main process)
        self.log_dir = opt.log_dir
        if self.is_main_process:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)
            
            args_dict = vars(opt)
            with open(os.path.join(self.log_dir, "opt.txt"), "w") as f:
                f.write("python train.py \\\n")
                for k in args_dict.keys():
                    f.write(f"    --{k} {args_dict[k]} \\\n")
        
            # Save current experiment code
            from utils import copy_code_folder
            copy_code_folder(output_dir=os.path.join(self.log_dir, "code"))

            os.makedirs(os.path.join(self.log_dir, "ckpt"), exist_ok=True)
    
    def create_dataset(self):
        from dataset.mix_dataset import MixDataset as SyntheticDataset
        self.train_dataset = SyntheticDataset(
            split="train",
            opt=opt,
        )
        self.val_dataset = SyntheticDataset(
            split="val",
            opt=opt,
        )
        
        # Do not pass world_size and rank directly to DistributedSampler
        self.train_sampler = DistributedSampler(self.train_dataset, 
                                               shuffle=True)
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            pin_memory=True,
            num_workers=4,
        )
        
        # Validation also uses DistributedSampler
        self.val_sampler = DistributedSampler(self.val_dataset,
                                             shuffle=False)
        
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            pin_memory=True,
            num_workers=4,
        )

    def create_model(self):
        from model.delight_base_net import DelightBaseModel
        cfg_path = opt.config_path
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        cfg["encoder"]["grad_checkpointing"] = opt.grad_ckpt == 1
        cfg["decoder"]["grad_checkpointing"] = opt.grad_ckpt == 1

        self.model = DelightBaseModel(
            enc_cfg=cfg["encoder"],
            dec_cfg=cfg["decoder"],
            device=self.device
        ).to(self.device)

        self.encoder_type = cfg["encoder"]["type"]

        if os.path.isfile(opt.init_ckpt_path):
            print(f"Rank {self.rank}: Loading initial checkpoint from {opt.init_ckpt_path}")
            weight = torch.load(opt.init_ckpt_path, map_location=self.device)
            self.model.load_state_dict(weight, strict=True)
        else:
            print(f"Rank {self.rank}: No initial checkpoint found at {opt.init_ckpt_path}, training from scratch.")
        
        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[self.device], find_unused_parameters=False)

    def create_optimizer(self):
        # Note: use the ddp model's module to access original parameters
        params = [
            {"params": self.model.module.encoder.parameters(), "lr": 1e-5},
            {"params": self.model.module.decoder.parameters(), "lr": 1e-4},
        ]
        self.optimizer = torch.optim.AdamW(params)
    
    def train(self):
        num_epoch = 1000
        num_iter = 0
        eval_freq = opt.eval_freq
        
        for epoch in range(num_epoch):
            # Set sampler epoch to ensure different shuffle per epoch
            self.train_sampler.set_epoch(epoch)
            
            for data in self.train_loader:
                num_iter += 1
                imgs = data["img"].to(self.device)
                gts = data["gt"].to(self.device)
                masks = data["mask"].to(self.device)
                data_source = data["data_source"].to(self.device)
                
                if "mask_loss" in data:
                    mask_loss = data["mask_loss"].to(self.device)
                else:
                    mask_loss = 1.

                with autocast():
                    if self.encoder_type == "mae_mix":
                        preds = self.model(imgs, data_source)
                    else:
                        preds = self.model(imgs)
                    
                    preds = preds * mask_loss
                    gts = gts * mask_loss

                    loss_l1 = F.l1_loss(preds, gts)
                    loss_lpips = self.lpips_loss(preds, gts, normalize=True).mean()
                    loss = loss_l1 + 0.1 * loss_lpips
                
                # Only main process logs training metrics
                if self.is_main_process:
                    self.writer.add_scalar('Train/Loss L1', loss_l1.item(), num_iter)
                    self.writer.add_scalar('Train/Loss Lpips', loss_lpips.item(), num_iter)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
                # Validation and saving
                if num_iter % opt.ckpt_freq == 0:
                    if self.is_main_process:
                        torch.save(self.model.module.state_dict(), os.path.join(self.log_dir, "ckpt", "iter_%08d.pth" % num_iter))
                
                if num_iter % eval_freq == 0:
                    # All processes need to create the save directory
                    save_dir = os.path.join(self.log_dir, "vis", "iter_%08d" % num_iter)
                    if self.is_main_process:
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(self.model.module.state_dict(), os.path.join(self.log_dir, "latest.pth"))
                    dist.barrier()
                    
                    loss_total_l1 = 0
                    loss_total_lpips = 0
                    
                    # No shuffle during validation; set sampler epoch for each process
                    self.val_sampler.set_epoch(0)
                    
                    # Switch model to eval mode
                    self.model.eval()
                    
                    for val_iter, data in enumerate(self.val_loader):
                        imgs = data["img"].to(self.device)
                        gts = data["gt"].to(self.device)
                        masks = data["mask"].to(self.device)
                        data_source = data["data_source"].to(self.device)

                        with autocast():
                            with torch.no_grad():
                                if self.encoder_type == "mae_mix":
                                    preds = self.model(imgs, data_source)
                                else:
                                    preds = self.model(imgs)
                                # preds = preds * masks
                                loss_l1 = F.l1_loss(preds, gts)
                                loss_lpips = self.lpips_loss(preds, gts, normalize=True).mean()
                                loss_total_l1 += loss_l1.item() * imgs.shape[0]
                                loss_total_lpips += loss_lpips.item() * imgs.shape[0]
                        
                        vis = torch.cat([imgs, preds, gts], dim=0)
                        save_image(
                            vis,
                            os.path.join(save_dir, "rank_%d_%05d.png" % (self.rank, val_iter)),
                            nrow=imgs.shape[0],
                        )

                    # Switch back to train mode
                    self.model.train()
                    
                    # Compute average losses across processes
                    total_val_samples = len(self.val_dataset)
                    
                    # Convert losses to tensors for all_reduce
                    loss_total_l1_tensor = torch.tensor(loss_total_l1, device=self.device)
                    loss_total_lpips_tensor = torch.tensor(loss_total_lpips, device=self.device)
                    
                    # Aggregate losses across processes
                    dist.all_reduce(loss_total_l1_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(loss_total_lpips_tensor, op=dist.ReduceOp.SUM)
                    
                    # Compute averages
                    loss_avg_l1 = loss_total_l1_tensor.item() / total_val_samples
                    loss_avg_lpips = loss_total_lpips_tensor.item() / total_val_samples
                    
                    # Log only in the main process
                    if self.is_main_process:
                        self.writer.add_scalar('Val/Loss L1', loss_avg_l1, num_iter)
                        self.writer.add_scalar('Val/Loss LPIPS', loss_avg_lpips, num_iter)

        # Training finished: clean up distributed environment
        dist.destroy_process_group()


def setup(rank, world_size):
    """Initialize distributed environment"""
    # Set master address and port
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = opt.ddp_port  # can be changed to another free port

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size):
    """Main function for each process"""
    # Initialize distributed environment
    setup(rank, world_size)
    
    # Set current device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    # Create trainer and start training
    trainer = Trainer(rank, world_size, device)
    # trainer.train_pseudo()  # Use pseudo data for testing
    trainer.train()


if __name__ == "__main__":
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training")
    
    # Launch training with multiprocessing
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    