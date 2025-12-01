import json
import matplotlib.pyplot as plt
import numpy as np
import os


def parse_loss_data_from_file(file_path):
    """
    从JSON文件解析损失数据

    Args:
        file_path: JSON文件路径

    Returns:
        epochs: epoch列表
        loss_dict: 包含所有损失项的字典
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return [], {}
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} 不是有效的JSON格式")
        return [], {}

    epochs = []
    loss_dict = {
        "ssim_loss": [],
        "grad_loss": [],
        "contest_loss": [],
        "total_loss": []
    }

    train_data = data.get("train", [])

    # 数据是交替存储的：数值，字典，数值，字典...
    for i in range(0, len(train_data), 2):
        if i + 1 >= len(train_data):
            break

        # 第一个元素是总损失值
        total_loss = train_data[i]

        # 第二个元素是包含详细信息的字典
        detail_dict = train_data[i + 1]

        # 检查字典结构是否正确
        if isinstance(detail_dict, dict) and "epoch" in detail_dict and "losses" in detail_dict:
            epoch = detail_dict["epoch"]
            losses = detail_dict["losses"]

            if isinstance(losses, dict):
                epochs.append(epoch)

                # 提取所有损失项
                for loss_name in loss_dict.keys():
                    if loss_name in losses:
                        loss_dict[loss_name].append(losses[loss_name])
                    else:
                        # 如果某个损失项不存在，用NaN填充
                        loss_dict[loss_name].append(float('nan'))

    return epochs, loss_dict


def plot_all_losses_separately(file_path, save_dir=None):
    """
    从JSON文件绘制所有损失曲线，并分别保存为5个图像

    Args:
        file_path: JSON文件路径
        save_dir: 保存图片的目录（可选）
    """
    # 解析数据
    epochs, loss_dict = parse_loss_data_from_file(file_path)

    if not epochs:
        print("没有有效的数据可以绘制")
        return

    # 如果未指定保存目录，使用JSON文件所在目录
    if save_dir is None:
        save_dir = os.path.dirname(file_path)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 为每个损失项创建单独的图表
    for loss_name, loss_values in loss_dict.items():
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss_values, 'b-', linewidth=2, marker='o')
        plt.title(f'{loss_name.upper()} Loss Convergence')
        plt.xlabel('Epoch')
        plt.ylabel(f'{loss_name.upper()} Loss Value')
        plt.grid(True, alpha=0.3)

        # 设置y轴范围，排除NaN值
        valid_values = [v for v in loss_values if not np.isnan(v)]
        if valid_values:
            y_min = min(valid_values)
            y_max = max(valid_values)
            y_range = y_max - y_min
            plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # 保存图像
        save_path = os.path.join(save_dir, f'{loss_name}_loss.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{loss_name.upper()}损失曲线已保存到: {save_path}")

        plt.close()  # 关闭图表以释放内存

    # 打印损失下降百分比
    print("\n损失下降百分比:")
    for loss_name, loss_values in loss_dict.items():
        if len(loss_values) > 1 and not np.isnan(loss_values[0]) and not np.isnan(loss_values[-1]):
            percent_decrease = ((loss_values[0] - loss_values[-1]) / loss_values[0] * 100)
            print(f"{loss_name.upper()}损失: {percent_decrease:.2f}%")


def plot_all_losses_together(file_path, save_path=None):
    """
    从JSON文件绘制所有损失曲线在一个图表中

    Args:
        file_path: JSON文件路径
        save_path: 保存图片的路径（可选）
    """
    # 解析数据
    epochs, loss_dict = parse_loss_data_from_file(file_path)

    if not epochs:
        print("没有有效的数据可以绘制")
        return

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 为每个损失项绘制曲线
    colors = ['b', 'r', 'g', 'c', 'm']
    markers = ['o', 's', '^', 'D', 'v']

    for i, (loss_name, loss_values) in enumerate(loss_dict.items()):
        if any(not np.isnan(v) for v in loss_values):  # 确保有有效数据
            plt.plot(epochs, loss_values,
                     color=colors[i % len(colors)],
                     marker=markers[i % len(markers)],
                     label=loss_name.upper(),
                     linewidth=2)

    plt.title('All Losses Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 使用对数坐标（如果损失变化范围大）
    all_values = []
    for loss_values in loss_dict.values():
        all_values.extend([v for v in loss_values if not np.isnan(v)])

    if all_values and max(all_values) / min(all_values) > 100:
        plt.yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"所有损失曲线已保存到: {save_path}")

    plt.show()


# 使用示例
if __name__ == "__main__":
    # 替换为你的JSON文件路径
    json_file_path = "path/to/your/train_losses.json"  # 请修改为实际路径

    # 检查文件是否存在
    if os.path.exists(json_file_path):
        # 分别保存5个损失图像
        plot_all_losses_separately(json_file_path)

        # 可选：在一个图表中绘制所有损失曲线
        # plot_all_losses_together(json_file_path)
    else:
        print(f"文件不存在: {json_file_path}")
        print("请确保提供了正确的文件路径")