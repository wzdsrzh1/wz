# test phase
import torch
from torch.autograd import Variable
from net import DenseFuse_net
import utils
from args_fusion import args
import numpy as np
import time
import cv2
import os
import csv
import matplotlib.pyplot as plt
from metrics import compute_metrics


def load_model(path, input_nc, output_nc):

	nest_model = DenseFuse_net(input_nc, output_nc)
	nest_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
	nest_model.cuda()

	return nest_model


def _generate_fusion_image(model, strategy_type, img1, img2):
	# encoder
	# test = torch.unsqueeze(img_ir[:, i, :, :], 1)
	en_r = model.encoder(img1)
	# vision_features(en_r, 'ir')
	en_v = model.encoder(img2)
	# vision_features(en_v, 'vi')
	# fusion
	f = model.fusion(en_r, en_v, strategy_type=strategy_type)
	# f = en_v
	# decoder
	img_fusion = model.decoder(f)
	return img_fusion[0]


def run_demo(model, infrared_path, visible_path, output_path_root, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode):
	# if mode == 'L':
	ir_img = utils.get_test_images(infrared_path, height=None, width=None, mode=mode)
	vis_img = utils.get_test_images(visible_path, height=None, width=None, mode=mode)
	# else:
	# 	img_ir = utils.tensor_load_rgbimage(infrared_path)
	# 	img_ir = img_ir.unsqueeze(0).float()
	# 	img_vi = utils.tensor_load_rgbimage(visible_path)
	# 	img_vi = img_vi.unsqueeze(0).float()

	# dim = img_ir.shape
	if args.cuda:
		ir_img = ir_img.cuda()
		vis_img = vis_img.cuda()
	ir_img = Variable(ir_img, requires_grad=False)
	vis_img = Variable(vis_img, requires_grad=False)
	dimension = ir_img.size()

	img_fusion = _generate_fusion_image(model, strategy_type, ir_img, vis_img)
	############################ multi outputs ##############################################
	file_name = 'fusion_' + fusion_type + '_' + str(index) +  '_network_' + network_type + '_' + strategy_type + '_' + ssim_weight_str + '.png'
	output_path = output_path_root + file_name
	# # save images
	# utils.save_image_test(img_fusion, output_path)
	# utils.tensor_save_rgbimage(img_fusion, output_path)
	if args.cuda:
		img = img_fusion.cpu().clamp(0, 255).data[0].numpy()
	else:
		img = img_fusion.clamp(0, 255).data[0].numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	utils.save_images(output_path, img)

	# prepare numpy sources for metrics (HxW or HxWx1)
	if args.cuda:
		ir_np = ir_img.cpu().numpy()[0].transpose(1, 2, 0)
		vi_np = vis_img.cpu().numpy()[0].transpose(1, 2, 0)
	else:
		ir_np = ir_img.numpy()[0].transpose(1, 2, 0)
		vi_np = vis_img.numpy()[0].transpose(1, 2, 0)

	metrics = compute_metrics(img, ir_np, vi_np)
	print(output_path)
	return output_path, metrics


def vision_features(feature_maps, img_type):
	count = 0
	for features in feature_maps:
		count += 1
		for index in range(features.size(1)):
			file_name = 'feature_maps_' + img_type + '_level_' + str(count) + '_channel_' + str(index) + '.png'
			output_path = 'outputs/feature_maps/' + file_name
			map = features[:, index, :, :].view(1,1,features.size(2),features.size(3))
			map = map*255
			# save images
			utils.save_image_test(map, output_path)


def main():
	# run demo
	# test_path = "images/test-RGB/"
	test_path = "images/IV_images/IV_images/"
	network_type = 'densefuse'
	fusion_type = 'auto'  # auto, fusion_layer, fusion_all
	strategy_type_list = ['addition', 'attention_weight']  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask

	output_path = './outputs/'
	strategy_type = strategy_type_list[0]

	if os.path.exists(output_path) is False:
		os.mkdir(output_path)
	metrics_dir = os.path.join(output_path, 'metrics')
	if os.path.exists(metrics_dir) is False:
		os.mkdir(metrics_dir)

	# in_c = 3 for RGB images; in_c = 1 for gray images
	in_c = 1
	if in_c == 1:
		out_c = in_c
		mode = 'L'
		model_path = args.model_path_gray
	else:
		out_c = in_c
		mode = 'RGB'
		model_path = args.model_path_rgb

	with torch.no_grad():
		print('SSIM weight ----- ' + args.ssim_path[2])
		ssim_weight_str = args.ssim_path[2]
		model = load_model(model_path, in_c, out_c)
		# discover all pairs (IRx.png, VISx.png)
		all_files = os.listdir(test_path)
		indices = sorted({
			f.split('.')[0][2:]
			for f in all_files
			if (f.lower().startswith('ir') or f.lower().startswith('vis')) and (f.lower().endswith('.png') or f.lower().endswith('.jpg'))
		})
		rows = []
		for idx in indices:
			ir_name = f"IR{idx}.png" if f"IR{idx}.png" in all_files else f"IR{idx}.jpg"
			vi_name = f"VIS{idx}.png" if f"VIS{idx}.png" in all_files else f"VIS{idx}.jpg"
			ir_path = os.path.join(test_path, ir_name)
			vi_path = os.path.join(test_path, vi_name)
			if not (os.path.exists(ir_path) and os.path.exists(vi_path)):
				continue
			fused_path, m = run_demo(model, ir_path, vi_path, output_path, idx, fusion_type, network_type, strategy_type, ssim_weight_str, mode)
			row = {'index': idx, **m}
			rows.append(row)

		# save csv
		csv_path = os.path.join(metrics_dir, 'metrics.csv')
		headers = ['index'] + list(rows[0].keys())[1:] if rows else ['index']
		with open(csv_path, 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=headers)
			writer.writeheader()
			for r in rows:
				writer.writerow(r)

		# bar chart per metric (averages)
		if rows:
			avg = {}
			for k in rows[0].keys():
				if k == 'index':
					continue
				avg[k] = float(np.mean([r[k] for r in rows]))
			plt.figure(figsize=(8,4))
			plt.bar(list(avg.keys()), list(avg.values()))
			plt.xticks(rotation=30, ha='right')
			plt.tight_layout()
			plt.ylabel('value')
			plt.title('Average metrics')
			plt.savefig(os.path.join(metrics_dir, 'metrics_avg.png'))
			plt.close()
	print('Done......')

if __name__ == '__main__':
	main()
