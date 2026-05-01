import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


def parse_args():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument(
		'--data_root',
		type=str,
		default='../data/ava256_4k',
		help='Root directory that contains one subdirectory per person.',
	)
	parser.add_argument(
		'--num_img',
		type=int,
		default=10,
		help='Number of views to render for each person.',
	)
	parser.add_argument(
		'--devices',
		type=int,
		nargs='+',
		default=[0, 1],
		help='GPU ids that can be used by blender_render_scan.py.',
	)
	parser.add_argument(
		'--save_root',
		type=str,
		default='../workspace/ava256_raw/fold0',
		help='Root directory for rendered results.',
	)
	return parser.parse_args()


def resolve_path(path_str, base_dir):
	path = Path(path_str)
	if path.is_absolute():
		return path
	return (base_dir / path).resolve()


def format_duration(seconds):
	seconds = max(0, int(seconds))
	hours, remainder = divmod(seconds, 3600)
	minutes, secs = divmod(remainder, 60)
	return f'{hours:02d}:{minutes:02d}:{secs:02d}'


def append_log(log_path, message):
	with open(log_path, 'a', encoding='utf-8') as file:
		file.write(message + '\n')


def run_render_for_person(person_dir, save_dir, num_img, devices, script_dir):
	blenderproc_exe = shutil.which('blenderproc')
	if blenderproc_exe is None:
		raise RuntimeError('Could not find `blenderproc` in PATH.')

	env = os.environ.copy()
	env['CUDA_VISIBLE_DEVICES'] = ','.join(str(device) for device in devices)

	cmd = [
		blenderproc_exe,
		'run',
		'blender_render_scan.py',
		'--',
		'--data_root',
		str(person_dir),
		'--save_root',
		str(save_dir),
		'--num_view',
		str(num_img),
	]

	print(
		f'[{person_dir.name}] start on GPUs {env["CUDA_VISIBLE_DEVICES"]}: '
		f'{person_dir} -> {save_dir}'
	)
	subprocess.run(cmd, cwd=script_dir, env=env, check=True)
	print(f'[{person_dir.name}] finished')


def main():
	opt = parse_args()

	script_dir = Path(__file__).resolve().parent

	data_root = resolve_path(opt.data_root, script_dir)
	save_root = resolve_path(opt.save_root, script_dir)

	if not data_root.exists():
		raise FileNotFoundError(f'data_root does not exist: {data_root}')

	save_root.mkdir(parents=True, exist_ok=True)
	log_path = save_root / 'log.txt'
	start_time = time.time()
	append_log(log_path, '=' * 80)
	append_log(log_path, f'start_time={time.strftime("%Y-%m-%d %H:%M:%S")}')

	person_dirs = sorted(
		[path for path in data_root.iterdir() if path.is_dir()],
		key=lambda path: path.name,
	)

	if not person_dirs:
		print(f'No person folders found under {data_root}')
		return

	devices = opt.devices
	if not devices:
		raise ValueError('`devices` must contain at least one GPU id.')

	total_count = len(person_dirs)
	for index, person_dir in enumerate(person_dirs, start=1):
		person_save_dir = save_root / person_dir.name
		person_save_dir.mkdir(parents=True, exist_ok=True)
		run_render_for_person(
			person_dir=person_dir,
			save_dir=person_save_dir,
			num_img=opt.num_img,
			devices=devices,
			script_dir=script_dir,
		)

		elapsed_seconds = time.time() - start_time
		remaining_count = total_count - index
		avg_seconds = elapsed_seconds / index
		eta_seconds = avg_seconds * remaining_count
		log_line = (
			f'[{index}/{total_count}] person={person_dir.name} '
			f'elapsed={format_duration(elapsed_seconds)} '
			f'eta={format_duration(eta_seconds)}'
		)
		append_log(log_path, log_line)
		print(log_line)

	append_log(log_path, f'finished_total={format_duration(time.time() - start_time)}')


if __name__ == '__main__':
	main()
