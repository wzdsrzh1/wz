import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as torch_ssim
import os
import csv
import matplotlib.pyplot as plt
from PIL import Image


def _to_gray_uint8(img):
	if img.ndim == 3 and img.shape[2] == 3:
		img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
	elif img.ndim == 3 and img.shape[2] == 1:
		img = img[:, :, 0]
	return np.clip(img, 0, 255).astype(np.uint8)


def entropy(img):
	gray = _to_gray_uint8(img)
	hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
	p = hist[hist > 0]
	return float(-np.sum(p * np.log2(p)))


def mutual_information(img1, img2, bins=256):
	x = _to_gray_uint8(img1)
	y = _to_gray_uint8(img2)
	joint_hist, _, _ = np.histogram2d(x.ravel(), y.ravel(), bins=bins, range=[[0, 255], [0, 255]], density=True)
	px = np.sum(joint_hist, axis=1, keepdims=True)
	py = np.sum(joint_hist, axis=0, keepdims=True)
	px_py = px @ py
	nz = joint_hist > 0
	mi = np.sum(joint_hist[nz] * np.log2(joint_hist[nz] / px_py[nz]))
	return float(mi)


def spatial_frequency(img):
	gray = _to_gray_uint8(img).astype(np.float32)
	row_diff = np.diff(gray, axis=1)
	col_diff = np.diff(gray, axis=0)
	rf = np.sqrt(np.mean(row_diff ** 2))
	cf = np.sqrt(np.mean(col_diff ** 2))
	return float(np.sqrt(rf ** 2 + cf ** 2))


def std_deviation(img):
	gray = _to_gray_uint8(img).astype(np.float32)
	return float(np.std(gray))


def q_abf(fused, a, b):
	# Xydeas & Petrovic edge-based Qabf
	fa = _to_gray_uint8(a).astype(np.float32)
	fb = _to_gray_uint8(b).astype(np.float32)
	ff = _to_gray_uint8(fused).astype(np.float32)

	def _gradients(x):
		kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
		ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
		gx = F.conv2d(torch.from_numpy(x)[None, None, :, :], torch.from_numpy(kx)[None, None, :, :], padding=1).numpy()[0, 0]
		gy = F.conv2d(torch.from_numpy(x)[None, None, :, :], torch.from_numpy(ky)[None, None, :, :], padding=1).numpy()[0, 0]
		mag = np.hypot(gx, gy)
		ang = np.arctan2(gy, gx)
		return mag, ang

	ma, anga = _gradients(fa)
	mb, angb = _gradients(fb)
	mf, angf = _gradients(ff)

	# Avoid zeros
	ma += 1e-8
	mb += 1e-8
	mf += 1e-8

	ga = 2 * ma * mf / (ma ** 2 + mf ** 2)
	gb = 2 * mb * mf / (mb ** 2 + mf ** 2)

	da = 1 - np.abs(np.cos(anga - angf))
	db = 1 - np.abs(np.cos(angb - angf))

	qa = ga * np.exp(-1 * da)
	qb = gb * np.exp(-1 * db)

	wa = ma / (ma + mb)
	wb = mb / (ma + mb)
	qabf = np.mean(wa * qa + wb * qb)
	return float(qabf)


def ssim_with_sources(fused, src):
	# Convert to torch [1,1,H,W] float in 0..255
	def _to_tensor(x):
		g = _to_gray_uint8(x).astype(np.float32)
		return torch.from_numpy(g)[None, None, :, :]
	f = _to_tensor(fused)
	s = _to_tensor(src)
	return float(torch_ssim(f, s, size_average=True, val_range=255))


def compute_metrics(fused, ir, vis):
	metrics = {}
	metrics['EN'] = entropy(fused)
	metrics['SF'] = spatial_frequency(fused)
	metrics['SD'] = std_deviation(fused)
	metrics['MI'] = mutual_information(fused, ir) + mutual_information(fused, vis)
	metrics['Qabf'] = q_abf(fused, ir, vis)
	metrics['SSIM_F_IR'] = ssim_with_sources(fused, ir)
	metrics['SSIM_F_VIS'] = ssim_with_sources(fused, vis)
	return metrics


def _load_img(path):
	img = Image.open(path)
	if img.mode not in ['L', 'RGB']:
		img = img.convert('RGB')
	return np.array(img)


def _match_indices(dir_path, prefix):
	files = os.listdir(dir_path)
	indices = sorted({
		f.split('.')[0][len(prefix):]
		for f in files
		if f.lower().startswith(prefix.lower()) and (f.lower().endswith('.png') or f.lower().endswith('.jpg') or f.lower().endswith('.jpeg'))
	})
	return files, indices


def evaluate_dirs(fused_dir, ir_dir=None, vis_dir=None, out_dir=None):
	if out_dir is None:
		out_dir = os.path.join(fused_dir, 'metrics')
	os.makedirs(out_dir, exist_ok=True)

	# gather fused files; try common prefixes
	fused_files, fused_idx = _match_indices(fused_dir, 'fusion')
	if not fused_idx:
		# fallback to F* naming
		fused_files, fused_idx = _match_indices(fused_dir, 'F')

	rows = []
	for idx in fused_idx:
		# locate fused path
		cands = [f for f in fused_files if f.split('.')[0].endswith(idx)]
		if not cands:
			continue
		fused_path = os.path.join(fused_dir, cands[0])
		fused = _load_img(fused_path)

		if ir_dir and vis_dir:
			ir_name_png = f'IR{idx}.png'
			ir_name_jpg = f'IR{idx}.jpg'
			vi_name_png = f'VIS{idx}.png'
			vi_name_jpg = f'VIS{idx}.jpg'
			ir_path = os.path.join(ir_dir, ir_name_png if os.path.exists(os.path.join(ir_dir, ir_name_png)) else ir_name_jpg)
			vi_path = os.path.join(vis_dir, vi_name_png if os.path.exists(os.path.join(vis_dir, vi_name_png)) else vi_name_jpg)
			if not (os.path.exists(ir_path) and os.path.exists(vi_path)):
				# skip if pair missing
				continue
			ir = _load_img(ir_path)
			vi = _load_img(vi_path)
			m = compute_metrics(fused, ir, vi)
		else:
			# compute single-image metrics
			m = {'EN': entropy(fused), 'SF': spatial_frequency(fused), 'SD': std_deviation(fused)}

		m_row = {'index': idx, **m}
		rows.append(m_row)

	# save csv
	if rows:
		csv_path = os.path.join(out_dir, 'metrics.csv')
		headers = list(rows[0].keys())
		with open(csv_path, 'w', newline='') as f:
			w = csv.DictWriter(f, fieldnames=headers)
			w.writeheader()
			for r in rows:
				w.writerow(r)

		# average bar chart
		avg = {}
		for k in headers:
			if k == 'index':
				continue
			avg[k] = float(np.mean([r[k] for r in rows]))
		plt.figure(figsize=(8, 4))
		plt.bar(list(avg.keys()), list(avg.values()))
		plt.xticks(rotation=30, ha='right')
		plt.tight_layout()
		plt.ylabel('value')
		plt.title('Average metrics')
		plt.savefig(os.path.join(out_dir, 'metrics_avg.png'))
		plt.close()

	return rows


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Compute fusion metrics from folder(s)')
	parser.add_argument('--fused_dir', type=str, required=True, help='包含融合图的文件夹')
	parser.add_argument('--ir_dir', type=str, default=None, help='红外图文件夹，可选')
	parser.add_argument('--vis_dir', type=str, default=None, help='可见光图文件夹，可选')
	parser.add_argument('--out_dir', type=str, default=None, help='输出指标文件夹，默认=fused_dir/metrics')
	args_cli = parser.parse_args()
	evaluate_dirs(args_cli.fused_dir, args_cli.ir_dir, args_cli.vis_dir, args_cli.out_dir)


