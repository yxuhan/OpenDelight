import os
import argparse
import random
from tqdm import tqdm


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str, default="workspace/ava256")
parser.add_argument('--save_root', type=str, default="workspace/ava256_split")
opt, _ = parser.parse_known_args()

id_name_list = sorted(os.listdir(opt.data_root))
random.shuffle(id_name_list)

val_id_name_list = id_name_list[:10]
train_id_name_list = id_name_list[10:]

train_save_root = os.path.join(opt.save_root, "train")
val_save_root = os.path.join(opt.save_root, "val")
os.makedirs(train_save_root, exist_ok=True)
os.makedirs(val_save_root, exist_ok=True)

for id_name in tqdm(train_id_name_list):
    src_dir = os.path.join(opt.data_root, id_name)
    dst_dir = os.path.join(train_save_root, id_name)
    os.system(f"cp -r {src_dir} {dst_dir}")

for id_name in tqdm(val_id_name_list):
    src_dir = os.path.join(opt.data_root, id_name)
    dst_dir = os.path.join(val_save_root, id_name)
    os.system(f"cp -r {src_dir} {dst_dir}")
