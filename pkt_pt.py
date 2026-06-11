import numpy as np
import matplotlib.pyplot as plt

# 根据原图弧度优化参数配置
config = {
    "YOLOv8-OBB": {
        "color": "#1f77b4",
        "peak": 0.82,  # 原图最终点对应8.2
        "start": 0.17,
        "sigmoid_range": (-4.2, 3.8),  # 调整S型相位
        "noise_amp": 0.008  # 减少噪声幅度
    },
    "Air YOLOv8-OBB(Ours)": {
        "color": "#ff7f0e",
        "peak": 0.85,  # 原图最终点对应8.5
        "start": 0.15,
        "sigmoid_range": (-3.8, 4.2),  # 更平缓的S型
        "noise_amp": 0.012  # 稍强噪声
    }
}

# 原图精确epoch节点
epochs = np.array([0, 20, 40, 60, 80, 100, 123, 140])


def create_authentic_curve(start, peak, x_range, noise_amp):
    """精确复现原图弧度的曲线生成函数"""
    x = np.linspace(x_range[0], x_range[1], len(epochs))
    # 改进的S型函数（增加曲率）
    sigmoid = 1 / (1 + np.exp(-x)) + 0.02 * x / (1 + abs(x))
    base_curve = start + (peak - start) * sigmoid

    # 原图特征噪声模式（中期波动较强）
    noise_pattern = np.array([0, 0.01, 0.03, -0.02, 0.01, -0.01, 0, 0])
    return np.clip(base_curve + noise_amp * noise_pattern, start - 0.02, peak + 0.01)


# 可视化增强设置
plt.figure(figsize=(10, 6))
ax = plt.gca()

# 绘制曲线
for model in config:
    params = config[model]
    curve = create_authentic_curve(params["start"], params["peak"],
                                   params["sigmoid_range"], params["noise_amp"])
    ax.plot(epochs, curve,
            color=params["color"],
            linewidth=2.8,  # 加粗线条
            marker='o',
            markersize=8,
            markeredgecolor='white',
            label=model)

# 精确坐标系统设置
ax.set_xticks(epochs)
ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
ax.set_ylim(0.15, 0.86)  # 扩展顶部空间
ax.set_xlim(-5, 145)

# 添加原图样式元素
ax.grid(True, linestyle=':', alpha=0.4)
ax.set_ylabel('mAP_0.5', fontsize=12, labelpad=12)
ax.set_xlabel('Training Epochs', fontsize=12, labelpad=12)
plt.title('Performance Comparison (OBB Detection)', fontsize=14, pad=18)

# 优化图例显示
legend = ax.legend(loc='lower right',
                   frameon=True,
                   framealpha=0.95,
                   edgecolor='#333333',
                   borderpad=1)
legend.get_frame().set_linewidth(1)

plt.tight_layout()
plt.show()