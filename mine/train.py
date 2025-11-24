# Training Medical Image Fusion Network
import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from model import MedicalFusion_net
from args_fusion import args
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import loss


def main():
    """
    Main training function for medical image fusion
    Supports multiple modality pairs: PET-MRI, CT-MRI, SPECT-CT, etc.
    """
    # Specify modality pair
    modality1 = 'PET'  # Options: 'PET', 'CT', 'SPECT', 'MRI'
    modality2 = 'MRI'  # Options: 'MRI', 'CT', 'PET'

    # Load paired medical images
    modality1_path = os.path.join(args.dataset, modality1)
    modality2_path = os.path.join(args.dataset, modality2)

    # Get image lists with names
    imgs_mod1, names1 = utils.list_images(modality1_path)
    imgs_mod2, names2 = utils.list_images(modality2_path)

    # Verify pairing (names should match)
    assert len(imgs_mod1) == len(imgs_mod2), "Modality image counts don't match!"
    print(f'Found {len(imgs_mod1)} paired {modality1}-{modality2} images')

    # Limit training samples if needed
    train_num = min(len(imgs_mod1), args.train_num) if hasattr(args, 'train_num') else len(imgs_mod1)
    imgs_mod1 = imgs_mod1[:train_num]
    imgs_mod2 = imgs_mod2[:train_num]

    # Train with different configurations
    i = 2  # SSIM weight index
    train(i, imgs_mod1, imgs_mod2, modality1, modality2)


def train(i, imgs_mod1, imgs_mod2, modality1, modality2):
    """
    Training function for medical image fusion

    Args:
        i: SSIM weight index
        imgs_mod1: list of modality 1 image paths
        imgs_mod2: list of modality 2 image paths
        modality1: name of modality 1 (e.g., 'PET')
        modality2: name of modality 2 (e.g., 'MRI')
    """
    batch_size = args.batch_size

    # Initialize network
    in_c = 1  # Medical images are typically grayscale
    input_nc = in_c
    output_nc = in_c

    print(f'\n{"=" * 60}')
    print(f'Training Medical Image Fusion Network')
    print(f'Modality Pair: {modality1} + {modality2}')
    print(f'{"=" * 60}\n')

    model = MedicalFusion_net(input_nc, output_nc)

    # Resume from checkpoint if specified
    if args.resume is not None and os.path.exists(args.resume):
        print(f'Resuming from checkpoint: {args.resume}')
        model.load_state_dict(torch.load(args.resume))

    # Print model architecture
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nTotal parameters: {total_params:,} ({total_params * 4 / 1e6:.2f}MB)\n')

    # Optimizer with weight decay for regularization
    optimizer = Adam(model.parameters(), args.lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=5, verbose=True)

    # Loss functions
    mse_loss = torch.nn.MSELoss()
    ssim_loss = loss.StructuralSimilarityLoss()
    l1_loss = torch.nn.L1Loss()

    # Move to GPU
    if args.cuda:
        model.cuda()

    # Create save directories
    temp_path_model = os.path.join(args.save_model_dir,
                                   f'{modality1}_{modality2}_{args.ssim_path[i]}')
    os.makedirs(temp_path_model, exist_ok=True)

    temp_path_loss = os.path.join(args.save_loss_dir,
                                  f'{modality1}_{modality2}_{args.ssim_path[i]}')
    os.makedirs(temp_path_loss, exist_ok=True)

    # Training logs
    Loss_pixel = []
    Loss_ssim = []
    Loss_texture = []
    Loss_all = []

    print('Starting training...\n')
    tbar = trange(args.epochs)

    for e in tbar:
        print(f'\nEpoch {e + 1}/{args.epochs}')

        # Load paired dataset
        paired_paths, batches = utils.load_dataset_medical(
            imgs_mod1, imgs_mod2, batch_size, shuffle=True
        )

        model.train()
        epoch_loss = 0.0
        all_ssim_loss = 0.0
        all_pixel_loss = 0.0
        all_texture_loss = 0.0

        for batch in range(batches):
            # Get batch paths
            batch_pairs = paired_paths[batch * batch_size:(batch + 1) * batch_size]
            paths_mod1 = [p[0] for p in batch_pairs]
            paths_mod2 = [p[1] for p in batch_pairs]

            # Load images with modality-specific preprocessing
            img_mod1 = utils.get_train_images_medical(
                paths_mod1,
                height=args.HEIGHT,
                width=args.WIDTH,
                normalize_method='percentile',
                augment=True,
                modality=modality1
            )

            img_mod2 = utils.get_train_images_medical(
                paths_mod2,
                height=args.HEIGHT,
                width=args.WIDTH,
                normalize_method='percentile',
                augment=True,
                modality=modality2
            )

            # Move to GPU
            if args.cuda:
                img_mod1 = img_mod1.cuda()
                img_mod2 = img_mod2.cuda()

            img_mod1 = Variable(img_mod1, requires_grad=False)
            img_mod2 = Variable(img_mod2, requires_grad=False)

            # Forward pass
            optimizer.zero_grad()

            # Encode both modalities
            en1 = model.encoder(img_mod1)
            en2 = model.encoder(img_mod2)

            # Fusion
            f = model.fusion(en1, en2, strategy_type='attention')

            # Decode
            outputs = model.decoder(f)

            # Calculate losses
            ssim_loss_value = 0.0
            pixel_loss_value = 0.0
            texture_loss_value = 0.0

            for output in outputs:
                # Reconstruction loss with both modalities
                pixel_loss_1 = mse_loss(output, img_mod1)
                pixel_loss_2 = mse_loss(output, img_mod2)
                pixel_loss_temp = (pixel_loss_1 + pixel_loss_2) / 2

                # SSIM loss with both modalities
                ssim_loss_1 = 1 - ssim_loss(output, img_mod1, normalize=True)
                ssim_loss_2 = 1 - ssim_loss(output, img_mod2, normalize=True)
                ssim_loss_temp = (ssim_loss_1 + ssim_loss_2) / 2

                # Texture preservation loss (gradient-based)
                texture_loss_temp = texture_loss(output, img_mod1, img_mod2)

                ssim_loss_value += ssim_loss_temp
                pixel_loss_value += pixel_loss_temp
                texture_loss_value += texture_loss_temp

            # Average over outputs
            ssim_loss_value /= len(outputs)
            pixel_loss_value /= len(outputs)
            texture_loss_value /= len(outputs)

            # Total loss with medical-specific weighting
            total_loss = (pixel_loss_value +
                          args.ssim_weight[i] * ssim_loss_value +
                          0.5 * texture_loss_value)

            # Backward and optimize
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses
            all_pixel_loss += pixel_loss_value.item()
            all_ssim_loss += ssim_loss_value.item()
            all_texture_loss += texture_loss_value.item()
            epoch_loss += total_loss.item()

            # Log progress
            if (batch + 1) % args.log_interval == 0:
                avg_pixel = all_pixel_loss / args.log_interval
                avg_ssim = all_ssim_loss / args.log_interval
                avg_texture = all_texture_loss / args.log_interval
                avg_total = (avg_pixel + args.ssim_weight[i] * avg_ssim + 0.5 * avg_texture)

                mesg = (f"{time.ctime()}\tEpoch {e + 1}:\t[{batch + 1}/{batches}]\t"
                        f"pixel: {avg_pixel:.6f}\tssim: {avg_ssim:.6f}\t"
                        f"texture: {avg_texture:.6f}\ttotal: {avg_total:.6f}")
                tbar.set_description(mesg)

                Loss_pixel.append(avg_pixel)
                Loss_ssim.append(avg_ssim)
                Loss_texture.append(avg_texture)
                Loss_all.append(avg_total)

                all_pixel_loss = 0.0
                all_ssim_loss = 0.0
                all_texture_loss = 0.0

            # Save checkpoint periodically
            if (batch + 1) % (200 * args.log_interval) == 0:
                save_checkpoint(model, e, batch + 1, temp_path_model, temp_path_loss,
                                Loss_pixel, Loss_ssim, Loss_texture, Loss_all,
                                modality1, modality2, args.ssim_path[i])
                model.train()
                if args.cuda:
                    model.cuda()

        # Update learning rate based on epoch loss
        avg_epoch_loss = epoch_loss / batches
        scheduler.step(avg_epoch_loss)
        print(f'Epoch {e + 1} average loss: {avg_epoch_loss:.6f}')

    # Save final model and losses
    save_final_checkpoint(model, temp_path_model, temp_path_loss,
                          Loss_pixel, Loss_ssim, Loss_texture, Loss_all,
                          modality1, modality2, args.ssim_path[i], args.epochs)

    print(f'\n{"=" * 60}')
    print('Training completed successfully!')
    print(f'Model saved to: {temp_path_model}')
    print(f'{"=" * 60}\n')


def texture_loss(output, img1, img2):
    """
    Texture preservation loss using gradient information
    Important for medical images to preserve edges and details
    """

    # Compute gradients
    def gradient(img):
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)

        if img.is_cuda:
            kernel_x = kernel_x.cuda()
            kernel_y = kernel_y.cuda()

        grad_x = torch.nn.functional.conv2d(img, kernel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(img, kernel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    grad_output = gradient(output)
    grad_img1 = gradient(img1)
    grad_img2 = gradient(img2)

    # Fused image should preserve maximum gradient information
    grad_target = torch.max(grad_img1, grad_img2)
    loss = torch.nn.functional.l1_loss(grad_output, grad_target)

    return loss


def save_checkpoint(model, epoch, iteration, model_dir, loss_dir,
                    loss_pixel, loss_ssim, loss_texture, loss_all,
                    modality1, modality2, ssim_path):
    """Save model checkpoint and loss data"""
    model.eval()
    model.cpu()

    timestamp = str(time.ctime()).replace(' ', '_').replace(':', '_')

    # Save model
    model_filename = (f"Epoch_{epoch}_iters_{iteration}_{timestamp}_"
                      f"{modality1}_{modality2}_{ssim_path}.model")
    model_path = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_path)

    # Save losses
    def save_loss(loss_data, loss_name):
        loss_array = np.array(loss_data)
        loss_filename = (f"loss_{loss_name}_epoch_{epoch}_iters_{iteration}_"
                         f"{timestamp}_{modality1}_{modality2}_{ssim_path}.mat")
        loss_path = os.path.join(loss_dir, loss_filename)
        scio.savemat(loss_path, {f'loss_{loss_name}': loss_array})

    save_loss(loss_pixel, 'pixel')
    save_loss(loss_ssim, 'ssim')
    save_loss(loss_texture, 'texture')
    save_loss(loss_all, 'total')

    print(f'\nCheckpoint saved: {model_path}\n')


def save_final_checkpoint(model, model_dir, loss_dir,
                          loss_pixel, loss_ssim, loss_texture, loss_all,
                          modality1, modality2, ssim_path, epochs):
    """Save final model and losses"""
    model.eval()
    model.cpu()

    timestamp = str(time.ctime()).replace(' ', '_').replace(':', '_')

    # Save final model
    model_filename = (f"Final_epoch_{epochs}_{timestamp}_"
                      f"{modality1}_{modality2}_{ssim_path}.model")
    model_path = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_path)

    # Save final losses
    def save_loss(loss_data, loss_name):
        loss_array = np.array(loss_data)
        loss_filename = (f"Final_loss_{loss_name}_epoch_{epochs}_"
                         f"{timestamp}_{modality1}_{modality2}_{ssim_path}.mat")
        loss_path = os.path.join(loss_dir, loss_filename)
        scio.savemat(loss_path, {f'loss_{loss_name}': loss_array})

    save_loss(loss_pixel, 'pixel')
    save_loss(loss_ssim, 'ssim')
    save_loss(loss_texture, 'texture')
    save_loss(loss_all, 'total')

    print(f'\nFinal model saved: {model_path}')


if __name__ == "__main__":
    main()