# Test phase for Medical Image Fusion
import torch
from torch.autograd import Variable
from model import MedicalFusion_net
import utils
from args_fusion import args
import numpy as np
import time
import cv2
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def load_model(path, input_nc, output_nc):
    """Load trained medical fusion model"""
    model = MedicalFusion_net(input_nc, output_nc)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    model.load_state_dict(torch.load(path))

    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1e6  # 4 bytes per float32

    print(f'Model: {model._get_name()}')
    print(f'Parameters: {total_params:,} ({model_size_mb:.2f}MB)')

    model.eval()
    model.cuda()

    return model


def _generate_fusion_image(model, strategy_type, img1, img2):
    """
    Generate fused image from two modality inputs

    Args:
        model: trained fusion model
        strategy_type: fusion strategy ('attention', 'weighted', 'max', 'addition')
        img1: first modality image tensor
        img2: second modality image tensor

    Returns:
        fused image tensor
    """
    # Encode both modalities
    en1 = model.encoder(img1)
    en2 = model.encoder(img2)

    # Fusion with specified strategy
    f = model.fusion(en1, en2, strategy_type=strategy_type)

    # Decode to get fused image
    img_fusion = model.decoder(f)

    return img_fusion[0]


def run_demo(model, img1_path, img2_path, output_path_root, index,
             modality1, modality2, strategy_type, save_comparison=True):
    """
    Run fusion on a single image pair

    Args:
        model: trained model
        img1_path: path to modality 1 image
        img2_path: path to modality 2 image
        output_path_root: root directory for outputs
        index: image index/identifier
        modality1: name of modality 1 (e.g., 'PET')
        modality2: name of modality 2 (e.g., 'MRI')
        strategy_type: fusion strategy
        save_comparison: whether to save side-by-side comparison

    Returns:
        tuple of (output_path, metrics_dict)
    """
    # Load images with modality-specific preprocessing
    img1 = utils.get_test_images_medical(
        img1_path,
        height=None,
        width=None,
        normalize_method='percentile',
        modality=modality1
    )

    img2 = utils.get_test_images_medical(
        img2_path,
        height=None,
        width=None,
        normalize_method='percentile',
        modality=modality2
    )

    # Move to GPU
    if args.cuda:
        img1 = img1.cuda()
        img2 = img2.cuda()

    img1 = Variable(img1, requires_grad=False)
    img2 = Variable(img2, requires_grad=False)

    # Generate fusion
    with torch.no_grad():
        img_fusion = _generate_fusion_image(model, strategy_type, img1, img2)

    # Convert to numpy for saving
    if args.cuda:
        fused_np = img_fusion.cpu().clamp(0, 255).data[0].numpy()
        img1_np = img1.cpu().data[0].numpy()
        img2_np = img2.cpu().data[0].numpy()
    else:
        fused_np = img_fusion.clamp(0, 255).data[0].numpy()
        img1_np = img1.data[0].numpy()
        img2_np = img2.data[0].numpy()

    # Transpose to HxWxC format
    fused_np = fused_np.transpose(1, 2, 0)
    img1_np = img1_np.transpose(1, 2, 0)
    img2_np = img2_np.transpose(1, 2, 0)

    # Save fused image
    filename = f'fusion_{modality1}_{modality2}_{index}_{strategy_type}.png'
    output_path = os.path.join(output_path_root, filename)
    utils.save_medical_image(output_path, fused_np, format='png')

    # Save comparison if requested
    if save_comparison:
        utils.save_fusion_results(
            output_path_root,
            fused_np,
            img1_np,
            img2_np,
            f'{modality1}_{modality2}_{index}_{strategy_type}',
            save_comparison=True
        )

    # Calculate metrics
    metrics = utils.calculate_metrics(fused_np, img1_np, img2_np)

    print(f'Processed: {filename}')
    return output_path, metrics


def main():
    """
    Main testing function for medical image fusion
    """
    # Configuration
    modality1 = 'PET'  # Options: 'PET', 'CT', 'SPECT', 'MRI'
    modality2 = 'MRI'  # Options: 'MRI', 'CT', 'PET'

    test_path = os.path.join(args.test_path, f'{modality1}_{modality2}')
    network_type = 'medical_fusion'

    # Fusion strategies to test
    strategy_types = ['attention', 'weighted', 'max', 'addition']

    # Output directory
    output_path = os.path.join('./outputs', f'{modality1}_{modality2}')
    os.makedirs(output_path, exist_ok=True)

    metrics_dir = os.path.join(output_path, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    # Model configuration
    in_c = 1  # Medical images are grayscale
    out_c = 1

    print(f'\n{"=" * 70}')
    print(f'Medical Image Fusion Testing')
    print(f'Modality Pair: {modality1} + {modality2}')
    print(f'Test Path: {test_path}')
    print(f'{"=" * 70}\n')

    # Load model
    model_path = args.model_path_gray
    if not os.path.exists(model_path):
        print(f'Error: Model not found at {model_path}')
        return

    with torch.no_grad():
        model = load_model(model_path, in_c, out_c)

        # Discover all image pairs
        # Expected naming: <modality>_<id>.png (e.g., PET_001.png, MRI_001.png)
        all_files = os.listdir(test_path)

        # Extract unique IDs
        ids_mod1 = set()
        ids_mod2 = set()

        for f in all_files:
            if f.lower().startswith(modality1.lower()):
                img_id = f.split('.')[0].replace(modality1, '').replace(modality1.upper(), '').strip('_')
                ids_mod1.add(img_id)
            elif f.lower().startswith(modality2.lower()):
                img_id = f.split('.')[0].replace(modality2, '').replace(modality2.upper(), '').strip('_')
                ids_mod2.add(img_id)

        # Find common IDs
        common_ids = sorted(ids_mod1.intersection(ids_mod2))

        if not common_ids:
            print(f'No paired images found in {test_path}')
            print(f'Expected naming format: {modality1}_<id>.png and {modality2}_<id>.png')
            return

        print(f'Found {len(common_ids)} paired images\n')

        # Test each fusion strategy
        for strategy_type in strategy_types:
            print(f'\n{"=" * 70}')
            print(f'Testing with fusion strategy: {strategy_type.upper()}')
            print(f'{"=" * 70}\n')

            strategy_output_path = os.path.join(output_path, strategy_type)
            os.makedirs(strategy_output_path, exist_ok=True)

            rows = []

            # Process each image pair
            for img_id in tqdm(common_ids, desc=f'Processing ({strategy_type})'):
                # Find matching files
                mod1_files = [f for f in all_files if f.startswith(f'{modality1}_{img_id}.') or
                              f.startswith(f'{modality1}{img_id}.')]
                mod2_files = [f for f in all_files if f.startswith(f'{modality2}_{img_id}.') or
                              f.startswith(f'{modality2}{img_id}.')]

                if not mod1_files or not mod2_files:
                    continue

                img1_path = os.path.join(test_path, mod1_files[0])
                img2_path = os.path.join(test_path, mod2_files[0])

                try:
                    fused_path, metrics = run_demo(
                        model, img1_path, img2_path, strategy_output_path,
                        img_id, modality1, modality2, strategy_type,
                        save_comparison=True
                    )

                    row = {'id': img_id, 'strategy': strategy_type, **metrics}
                    rows.append(row)

                except Exception as e:
                    print(f'Error processing {img_id}: {str(e)}')
                    continue

            # Save metrics CSV for this strategy
            if rows:
                csv_path = os.path.join(metrics_dir, f'metrics_{strategy_type}.csv')
                headers = list(rows[0].keys())

                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)

                print(f'\nMetrics saved to: {csv_path}')

                # Print average metrics
                print(f'\nAverage Metrics ({strategy_type}):')
                print('-' * 50)
                for key in rows[0].keys():
                    if key not in ['id', 'strategy']:
                        avg_val = np.mean([r[key] for r in rows])
                        std_val = np.std([r[key] for r in rows])
                        print(f'{key:15s}: {avg_val:.4f} ± {std_val:.4f}')

        # Compare strategies
        print(f'\n{"=" * 70}')
        print('Comparing Fusion Strategies')
        print(f'{"=" * 70}\n')
        compare_strategies(metrics_dir, strategy_types, modality1, modality2)

    print(f'\n{"=" * 70}')
    print('Testing completed successfully!')
    print(f'Results saved to: {output_path}')
    print(f'{"=" * 70}\n')


def compare_strategies(metrics_dir, strategy_types, modality1, modality2):
    """
    Compare different fusion strategies

    Args:
        metrics_dir: directory containing metrics CSV files
        strategy_types: list of strategy names
        modality1: name of modality 1
        modality2: name of modality 2
    """
    all_metrics = {}

    # Load all strategy metrics
    for strategy in strategy_types:
        csv_path = os.path.join(metrics_dir, f'metrics_{strategy}.csv')
        if not os.path.exists(csv_path):
            continue

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if rows:
            all_metrics[strategy] = rows

    if not all_metrics:
        print('No metrics found for comparison')
        return

    # Calculate average metrics per strategy
    metric_names = [k for k in all_metrics[strategy_types[0]][0].keys()
                    if k not in ['id', 'strategy']]

    comparison_data = {metric: {} for metric in metric_names}

    for strategy, rows in all_metrics.items():
        for metric in metric_names:
            values = [float(r[metric]) for r in rows]
            comparison_data[metric][strategy] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metric_names):
        if idx >= len(axes):
            break

        strategies = list(comparison_data[metric].keys())
        means = [comparison_data[metric][s]['mean'] for s in strategies]
        stds = [comparison_data[metric][s]['std'] for s in strategies]

        axes[idx].bar(strategies, means, yerr=stds, capsize=5, alpha=0.7)
        axes[idx].set_title(f'{metric}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Value')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)

    # Remove extra subplots
    for idx in range(len(metric_names), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f'Fusion Strategy Comparison: {modality1} + {modality2}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    comparison_plot_path = os.path.join(metrics_dir, 'strategy_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Comparison plot saved to: {comparison_plot_path}')

    # Save comparison table
    table_path = os.path.join(metrics_dir, 'strategy_comparison.csv')
    with open(table_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Metric'] + [f'{s}_mean' for s in strategy_types] + \
                 [f'{s}_std' for s in strategy_types]
        writer.writerow(header)

        for metric in metric_names:
            row = [metric]
            for strategy in strategy_types:
                if strategy in comparison_data[metric]:
                    row.append(f"{comparison_data[metric][strategy]['mean']:.4f}")
            for strategy in strategy_types:
                if strategy in comparison_data[metric]:
                    row.append(f"{comparison_data[metric][strategy]['std']:.4f}")
            writer.writerow(row)

    print(f'Comparison table saved to: {table_path}')

    # Print summary
    print('\nBest Strategy per Metric:')
    print('-' * 50)
    for metric in metric_names:
        best_strategy = max(comparison_data[metric].items(),
                            key=lambda x: x[1]['mean'])
        print(f'{metric:15s}: {best_strategy[0]:12s} '
              f'({best_strategy[1]["mean"]:.4f})')


if __name__ == '__main__':
    main()