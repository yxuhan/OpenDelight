import os
import argparse


parser = argparse.ArgumentParser()

# DATA CONFIG
parser.add_argument("--data_root", type=str, default="misc/test_ffhq")
parser.add_argument("--save_root", type=str, default="workspace/test_ffhq_results")
parser.add_argument("--model_name", type=str, default="opendelight")

# MASK CONFIG
parser.add_argument('--skin_mask', type=int, default=1)

# ALIGN CONFIG
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--border_scale', type=float, default=0.85)

# MODEL CONFIG
parser.add_argument('--ckpt_path', type=str, default='pretrained/opendelight/base_delight_network.pth')
parser.add_argument('--cfg_path', type=str, default='config/delight_base_network.yaml')
parser.add_argument("--data_idx", type=int, default=0)

# DEVICE CONFIG
parser.add_argument("--device", type=str, default="0")

# ENHANCER CONFIG
parser.add_argument('--use_enhancer', type=int, default=1)
parser.add_argument('--enhancer_ckpt_path', type=str, default="pretrained/opendelight/unet_enhancer.pth")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device


import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import yaml
import torch
from torchvision import transforms
from torchvision.utils import save_image

from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor

from synthetic.align_utils import align_face
from synthetic.align_utils import inverse_align_face


def convert_to_absolute_path(input_path):
	if os.path.isabs(input_path):
		return input_path
	else:
		return os.path.abspath(input_path)


code_dir = os.path.dirname(os.path.abspath(__file__))
args.data_root = convert_to_absolute_path(args.data_root)
args.save_root = convert_to_absolute_path(args.save_root)


# config root
opendelight_root = os.path.join(args.save_root, args.model_name)
mat_root = os.path.join(opendelight_root, "mat")
os.makedirs(mat_root, exist_ok=True)
img_root = args.data_root
save_root = os.path.join(opendelight_root, "delight")
os.makedirs(save_root, exist_ok=True)


def mat_image():
	mat_code_root = os.path.join(code_dir, "matting")
	mat_python_path = "python"

	cmd = """
		%s run_matting.py \
		--input_root %s \
		--output_root %s
	"""

	os.chdir(mat_code_root)
	os.system(cmd % (mat_python_path, img_root, mat_root))
	os.chdir(code_dir)


def skin_mask_image(predictor):
	import facer
	# face_detector = facer.face_detector('retinaface/mobilenet', device="cuda")
	face_parser = facer.face_parser('farl/lapa/448', device="cuda") # optional "farl/celebm/448"
	for pth in tqdm(sorted(os.listdir(img_root))):
		img_path = os.path.join(img_root, pth)

		with torch.no_grad():
			image_cv2 = cv2.imread(img_path)
			face_lmk = predictor.detect(image_cv2)
			
			kps = np.concatenate([
				np.mean(face_lmk[0, 36:42], axis=0, keepdims=True),  # left eye
				np.mean(face_lmk[0, 42:48], axis=0, keepdims=True),  # right eye
				face_lmk[0, 30:31],
				face_lmk[0, 48:49],
				face_lmk[0, 54:55],
			], axis=0)

			kps = torch.from_numpy(kps).unsqueeze(0).to(device="cuda")

		image = facer.hwc2bchw(facer.read_hwc(img_path)).to(device="cuda")
		with torch.inference_mode():
			faces = {
				"points": kps,
				"image_ids": torch.tensor([0], device="cuda"),
			}
			faces = face_parser(image, faces)
		
		seg_logits = faces['seg']['logits']
		seg_probs = seg_logits.softmax(dim=1)[0]  # nclasses x h x w
		vis_seg_probs = seg_probs.argmax(dim=0)

		hair_mask = (vis_seg_probs == 10).float()
		# eye_mask = (vis_seg_probs == 4).float() + (vis_seg_probs == 5).float()
		face_mask = (vis_seg_probs >= 1).float()
		face_mask = face_mask - hair_mask
		
		mat = transforms.ToTensor()(Image.open(os.path.join(mat_root, pth))).to(face_mask.device)
		mat = mat * face_mask
		save_image(mat, os.path.join(mat_root, pth))


def align_image(img_path, mask_path, predictor):
	img = transforms.ToTensor()(Image.open(img_path))
	if img.shape[0] == 4:
		img = img[:3] * img[3:]
	
	img = img[[2, 1, 0]]
	img = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
	lm = predictor.detect(img)[0]
	
	img_float = img.astype('float32')
	aligned_face, H = align_face(
		img_float, 
		output_size=args.img_size, 
		lm=lm,
		border_scale=args.border_scale,
	)
	aligned_face = aligned_face.astype('uint8')
	
	mask = cv2.imread(mask_path)
	mask_float = mask.astype('float32')
	aligned_mask, _ = align_face(
		mask_float, 
		output_size=args.img_size, 
		lm=lm,
		border_scale=args.border_scale,
	)
	
	return aligned_face, aligned_mask, {"H": H, "orig_size": (img.shape[1], img.shape[0])}


def inv_align_ffhq(img, params):
	img_float = img.astype('float32')
	mask = np.ones_like(img_float)
	inv_aligned_face = inverse_align_face(img_float, params["H"], params["orig_size"])
	inv_aligned_face = inv_aligned_face.astype('uint8')
	inv_mask = inverse_align_face(mask, params["H"], params["orig_size"])
	inv_mask = (inv_mask[:, :, 0] > 0).astype('uint8') * 255
	return inv_aligned_face, inv_mask


class LandmarksDetectorIBug:
	def __init__(self, device):
		self.face_detector = RetinaFacePredictor(
			threshold=0.8, device=device,
			model=RetinaFacePredictor.get_model('resnet50')
		)

		# Create a facial landmark detector
		self.landmark_detector = FANPredictor(
			device=device, model=FANPredictor.get_model('2dfan2_alt')
		)
	
	def detect(self, images):
		# images should be in OPENCV format
		detected_faces = self.face_detector(images, rgb=False)
		landmarks, scores = self.landmark_detector(images, detected_faces, rgb=False)
		return landmarks


def cv2_to_torch(img_cv2, device):
	img = torch.from_numpy(img_cv2).permute(2, 0, 1).float() / 255.0
	if img.shape[0] == 3:
		return img[None, [2, 1, 0], ...].to(device)
	elif img.shape[0] == 1:
		return img[None, ...].to(device)
	else:
		raise NotImplementedError("CV2 to Torch only supports 1 or 3 channel images.")


def torch_to_cv2(img_tensor):
	img = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
	return img[..., [2, 1, 0]]


def create_model(device):
	from model.delight_base_net import DelightBaseModel
	cfg_path = args.cfg_path
	with open(cfg_path, "r") as f:
		cfg = yaml.safe_load(f)

	model = DelightBaseModel(
		enc_cfg=cfg["encoder"],
		dec_cfg=cfg["decoder"],
		device=device,
	).to(device)

	weight = torch.load(args.ckpt_path, map_location=device)
	model.load_state_dict(weight, strict=True)
	model.eval()

	return model, cfg["encoder"]["type"]


def create_enhancer(device):
	from model.detail_enhance_unet import UNet_Enhancer
	model = UNet_Enhancer(
		n_channels=6,
		n_classes=3,
		output_confidence=False,
		output_shadow_mask=False,
	).to(device)

	weight = torch.load(args.enhancer_ckpt_path, map_location=device)
	model.load_state_dict(weight, strict=True)
	model.eval()
	return model


if __name__ == "__main__":
	predictor = LandmarksDetectorIBug(device="cuda")
	model, enc_type = create_model(device="cuda")

	mat_image()
	if args.skin_mask == 1:
		skin_mask_image(predictor)
	if args.use_enhancer == 1:
		enhancer = create_enhancer(device="cuda")

	for pth in tqdm(sorted(os.listdir(img_root))):
		img_path = os.path.join(img_root, pth)
		mask_path = os.path.join(mat_root, pth)
		
		# align
		aligned_face, aligned_mask, params = align_image(img_path, mask_path, predictor)
		
		# infer
		aligned_face_torch = cv2_to_torch(aligned_face, device="cuda")
		aligned_mask_torch = cv2_to_torch(aligned_mask, device="cuda")
		input_face_torch = aligned_face_torch * aligned_mask_torch

		with torch.no_grad():
			input_face_torch_512 = torch.nn.functional.interpolate(
				input_face_torch, size=(512, 512), mode='bicubic'
			)
			
			if enc_type == "mae_mix":
				output_face_torch = model(input_face_torch_512, torch.tensor([args.data_idx]))
			else:
				output_face_torch = model(input_face_torch_512, None)
			
			output_face_torch = torch.nn.functional.interpolate(
				output_face_torch, size=(args.img_size, args.img_size), mode='bicubic'
			)

			if args.use_enhancer == 1:
				enhanced_torch = enhancer(
					torch.cat([output_face_torch, input_face_torch], dim=1)
				)["diffuse"]

		# align back
		output_face = torch_to_cv2(output_face_torch)
		res = inv_align_ffhq(output_face, params)[0]
		res_mask, inv_mask = inv_align_ffhq(aligned_mask, params)
		
		res_torch = cv2_to_torch(res, device="cuda")
		res_mask_torch = cv2_to_torch(res_mask, device="cuda")
		inv_mask_torch = cv2_to_torch(inv_mask[..., None], device="cuda")
		final_mask = res_mask_torch * inv_mask_torch

		if args.use_enhancer == 1:
			enhanced = torch_to_cv2(enhanced_torch)
			enh_res = inv_align_ffhq(enhanced, params)[0]
			enh_res_torch = cv2_to_torch(enh_res, device="cuda")
			save_image(
				torch.cat([enh_res_torch, final_mask[:, :1]], dim=1),
				os.path.join(save_root, pth),
			)
		else:
			save_image(
			torch.cat([res_torch, final_mask[:, :1]], dim=1),
			os.path.join(save_root, pth),
		)
