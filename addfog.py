# import os
# import random
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
#
# def add_fog_pil(image, fog_density=0.7):
#     """
#     无OpenCV版本：给图片加雾（大气散射模型）
#     适配航拍：上远下近，远处雾更浓
#     """
#     img = np.array(image, dtype=np.float32)
#     h, w = img.shape[:2]
#
#     # 白光雾
#     A = 255.0
#
#     # 航拍深度：顶部远(雾浓)，底部近(雾淡)
#     depth = np.linspace(0.1, 1.0, h, dtype=np.float32)
#     depth = np.tile(depth.reshape(-1, 1), (1, w))
#
#     # 雾透射率
#     transmission = np.exp(-fog_density * depth)
#     transmission = np.expand_dims(transmission, axis=-1)
#
#     # 雾图公式
#     foggy = img * transmission + A * (1 - transmission)
#     foggy = np.clip(foggy, 0, 255).astype(np.uint8)
#
#     return Image.fromarray(foggy)
#
# def process_folder():
#     # ==================== 你只需要改这里 ====================
#     INPUT_DIR = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\JPEGImages"       # 原图路径
#     OUTPUT_DIR = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\train_fog"   # 雾图输出路径
#     # ======================================================
#
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     suffixes = ('.jpg', '.jpeg', '.png', '.bmp')
#     files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(suffixes)]
#
#     print(f"找到 {len(files)} 张图片，开始生成雾图...")
#
#     for filename in tqdm(files):
#         try:
#             img_path = os.path.join(INPUT_DIR, filename)
#             img = Image.open(img_path).convert("RGB")
#
#             # 随机雾浓度（更适合训练）
#             density = random.uniform(0.3, 0.9)
#             fog_img = add_fog_pil(img, density)
#
#             save_path = os.path.join(OUTPUT_DIR, filename)
#             fog_img.save(save_path, quality=95)
#
#         except Exception as e:
#             print(f"跳过 {filename}: {str(e)}")
#
#     print(f"\n✅ 全部完成！雾图保存在：{OUTPUT_DIR}")
#
# if __name__ == "__main__":
#     process_folder()

#
# import os
# import random
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
#
# def add_fog_pil(image, fog_density=1.8):
#     img = np.array(image, dtype=np.float32)
#     h, w = img.shape[:2]
#
#     # 雾的底色（更白更浓）
#     A = 255.0
#
#     # 航拍深度：上远下近
#     depth = np.linspace(0.2, 1.2, h, dtype=np.float32)
#     depth = np.tile(depth.reshape(-1, 1), (1, w))
#
#     # 核心：加大密度，雾更浓
#     transmission = np.exp(-fog_density * depth)
#     transmission = np.expand_dims(transmission, axis=-1)
#
#     foggy = img * transmission + A * (1 - transmission)
#     foggy = np.clip(foggy, 0, 255).astype(np.uint8)
#
#     return Image.fromarray(foggy)
#
# def process_folder():
#     # ==================== 路径自己改 ====================
#     INPUT_DIR = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\JPEGImages"
#     OUTPUT_DIR = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\train_fog2"
#     # ====================================================
#
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     suffixes = ('.jpg', '.jpeg', '.png', '.bmp')
#     files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(suffixes)]
#
#     print(f"找到 {len(files)} 张图片，开始生成浓雾...")
#
#     for filename in tqdm(files):
#         try:
#             img_path = os.path.join(INPUT_DIR, filename)
#             img = Image.open(img_path).convert("RGB")
#
#             # 浓雾浓度范围：1.5 ~ 2.5，雾非常明显
#             density = random.uniform(1.5, 2.5)
#             fog_img = add_fog_pil(img, density)
#
#             save_path = os.path.join(OUTPUT_DIR, filename)
#             fog_img.save(save_path, quality=95)
#
#         except Exception as e:
#             print(f"跳过 {filename}: {str(e)}")
#
#     print(f"\n✅ 浓雾生成完成！已保存到：{OUTPUT_DIR}")
#
# if __name__ == "__main__":
#     process_folder()

#
# import os
# import random
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
#
# def add_fog_pil(image, fog_density=1.0):
#     img = np.array(image, dtype=np.float32)
#     h, w = img.shape[:2]
#     A = 255.0
#
#     # 航拍深度：上远下近
#     depth = np.linspace(0.1, 1.0, h, dtype=np.float32)
#     depth = np.tile(depth.reshape(-1, 1), (1, w))
#
#     transmission = np.exp(-fog_density * depth)
#     transmission = np.expand_dims(transmission, axis=-1)
#
#     foggy = img * transmission + A * (1 - transmission)
#     foggy = np.clip(foggy, 0, 255).astype(np.uint8)
#     return Image.fromarray(foggy)
#
# def process_folder():
#     # ==================== 改你自己的路径 ====================
#     INPUT_DIR = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\JPEGImages"
#     OUTPUT_DIR = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\train_fog3"
#     # ======================================================
#
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     suffixes = ('.jpg', '.jpeg', '.png', '.bmp')
#     files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(suffixes)]
#
#     print(f"找到 {len(files)} 张图片，开始生成雾效...")
#
#     for filename in tqdm(files):
#         try:
#             img_path = os.path.join(INPUT_DIR, filename)
#             img = Image.open(img_path).convert("RGB")
#
#             # 中等雾浓度：0.7 ~ 1.2 之间随机
#             density = random.uniform(0.7, 1.2)
#             fog_img = add_fog_pil(img, density)
#
#             save_path = os.path.join(OUTPUT_DIR, filename)
#             fog_img.save(save_path, quality=95)
#
#         except Exception as e:
#             print(f"跳过 {filename}: {str(e)}")
#
#     print(f"\n✅ 处理完成！雾图保存在：{OUTPUT_DIR}")
#
# if __name__ == "__main__":
#     process_folder()

#
# import os
# import random
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
#
# def add_fog_pil(image, fog_density):
#     img = np.array(image, dtype=np.float32)
#     h, w = img.shape[:2]
#     A = 255.0
#
#     depth = np.linspace(0.1, 1.0, h, dtype=np.float32)
#     depth = np.tile(depth.reshape(-1, 1), (1, w))
#
#     transmission = np.exp(-fog_density * depth)
#     transmission = np.expand_dims(transmission, axis=-1)
#
#     foggy = img * transmission + A * (1 - transmission)
#     foggy = np.clip(foggy, 0, 255).astype(np.uint8)
#     return Image.fromarray(foggy)
#
# def process_folder():
#     # ==================== 改成你自己的路径 ====================
#     INPUT_DIR = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\JPEGImages"
#     OUTPUT_DIR = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\train_fog4"
#     # ======================================================
#
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     suffixes = ('.jpg', '.jpeg', '.png', '.bmp')
#     files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(suffixes)]
#
#     print(f"找到 {len(files)} 张图片，开始生成随机雾效...")
#
#     for filename in tqdm(files):
#         try:
#             img_path = os.path.join(INPUT_DIR, filename)
#             img = Image.open(img_path).convert("RGB")
#
#             # 随机雾浓度：淡、中、浓混合
#             density = random.uniform(0.4, 1.5)
#             fog_img = add_fog_pil(img, density)
#
#             save_path = os.path.join(OUTPUT_DIR, filename)
#             fog_img.save(save_path, quality=95)
#
#         except Exception as e:
#             print(f"跳过 {filename}: {str(e)}")
#
#     print(f"\n✅ 随机雾效生成完成！已保存到：{OUTPUT_DIR}")
#
# if __name__ == "__main__":
#     process_folder()

#
# import os
# import random
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
#
# def add_fog_natural(image):
#     img = np.array(image, dtype=np.float32)
#     h, w = img.shape[:2]
#     A = 255.0
#
#     # 1. 基础深度：航拍上远下近
#     depth_line = np.linspace(0.1, 1.0, h)
#     depth = np.tile(depth_line.reshape(-1, 1), (1, w))
#
#     # 2. 随机噪声 = 制造局部浓淡不均（关键）
#     scale = random.uniform(40, 80)
#     x = np.linspace(0, scale, w)
#     y = np.linspace(0, scale, h)
#     xv, yv = np.meshgrid(x, y)
#
#     # 多层噪声模拟自然雾团
#     noise = (np.sin(yv * 0.3) * np.cos(xv * 0.5) +
#              np.sin(yv * 0.7) * np.cos(xv * 0.2)) * 0.5
#     noise = (noise + 1.0) / 2.0  # 归一化 0~1
#
#     # 3. 最终深度 = 基础景深 + 随机局部雾团
#     depth = depth * (0.4 + 0.6 * noise)
#
#     # 4. 随机整体浓度：薄雾~中浓雾
#     base_density = random.uniform(0.5, 1.2)
#     transmission = np.exp(-base_density * depth)
#     transmission = np.expand_dims(transmission, axis=-1)
#
#     # 合成雾图
#     foggy = img * transmission + A * (1 - transmission)
#     foggy = np.clip(foggy, 0, 255).astype(np.uint8)
#     return Image.fromarray(foggy)
#
# def process_folder():
#     # ==================== 你的路径 ====================
#     INPUT_DIR = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\JPEGImages"
#     OUTPUT_DIR = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\train_fog5"
#     # ==================================================
#
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     suffixes = ('.jpg', '.jpeg', '.png', '.bmp')
#     files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(suffixes)]
#
#     print(f"找到 {len(files)} 张图片，生成自然不均匀雾效...")
#
#     for filename in tqdm(files):
#         try:
#             img_path = os.path.join(INPUT_DIR, filename)
#             img = Image.open(img_path).convert("RGB")
#
#             fog_img = add_fog_natural(img)
#
#             save_path = os.path.join(OUTPUT_DIR, filename)
#             fog_img.save(save_path, quality=95)
#
#         except Exception as e:
#             print(f"跳过 {filename}: {str(e)}")
#
#     print("\n✅ 自然不均匀雾效生成完成！")
#
# if __name__ == "__main__":
#     process_folder()

# import os
# import shutil
# from tqdm import tqdm
#
# def split_by_txt():
#     # ====================== 【你只需要改这4个路径】 ======================
#     IMG_DIR = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\FOG_NWPU"         # 所有图片所在的总文件夹
#     LABEL_DIR = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\Annotations"        # 所有标签所在的总文件夹
#     TXT_FOLDER = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\ImageSets\Main"   # 存放 train.txt val.txt test.txt 的文件夹
#     OUTPUT_ROOT = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\NWPU_split"       # 划分后输出的新数据集目录
#     # ===================================================================
#
#     # 自动创建输出目录
#     sets = ['train', 'val', 'test']
#     for s in sets:
#         os.makedirs(os.path.join(OUTPUT_ROOT, 'images', s), exist_ok=True)
#         os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', s), exist_ok=True)
#
#     # 遍历 训练/验证/测试
#     for mode in sets:
#         txt_path = os.path.join(TXT_FOLDER, f"{mode}.txt")
#         if not os.path.exists(txt_path):
#             print(f"跳过 {txt_path}，文件不存在")
#             continue
#
#         # 读取列表
#         with open(txt_path, 'r', encoding='utf-8') as f:
#             lines = [line.strip() for line in f if line.strip()]
#
#         print(f"\n【{mode}】共 {len(lines)} 张图片，开始复制...")
#
#         # 逐个复制
#         for name in tqdm(lines):
#             # 图片
#             src_img = os.path.join(IMG_DIR, name + '.jpg')  # NWPU一般是jpg
#             dst_img = os.path.join(OUTPUT_ROOT, 'images', mode, name + '.jpg')
#
#             # 标签
#             src_lab = os.path.join(LABEL_DIR, name + '.txt')
#             dst_lab = os.path.join(OUTPUT_ROOT, 'labels', mode, name + '.txt')
#
#             # 复制
#             if os.path.exists(src_img):
#                 shutil.copy(src_img, dst_img)
#             if os.path.exists(src_lab):
#                 shutil.copy(src_lab, dst_lab)
#
#     print("\n✅ 数据集划分完成！输出目录：", OUTPUT_ROOT)
#
# if __name__ == "__main__":
#     split_by_txt()

#


import os
import random
import shutil
from tqdm import tqdm

# ====================== 【你只需要改这里】 ======================
IMG_DIR = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\FOG_NWPU\image"         # 所有图片所在的总文件夹
LABEL_DIR = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\FOG_NWPU\labels"        # 所有标签所在的总文件夹
OUTPUT_ROOT = r"F:\dataset\NWPU VHR-10 dataset\voc-nwpu-10\NWPU_split"        # 划分后输出的数据集
# ==============================================================

def split_dataset():
    # 固定数量：800张 → 650 / 50 / 100
    train_num = 650
    val_num = 50
    test_num = 100

    # 1. 获取所有图片名称（不带后缀）
    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    img_files.sort()
    print(f"总图片数量：{len(img_files)}")

    # 2. 随机打乱
    random.seed(42)  # 固定随机种子，可复现
    random.shuffle(img_files)

    # 3. 按数量切分
    train_files = img_files[:train_num]
    val_files = img_files[train_num : train_num + val_num]
    test_files = img_files[train_num + val_num : train_num + val_num + test_num]

    print(f"训练集：{len(train_files)}")
    print(f"验证集：{len(val_files)}")
    print(f"测试集：{len(test_files)}")

    # 4. 创建输出目录
    for mode in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'images', mode), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', mode), exist_ok=True)

    # 5. 保存 train/val/test.txt
    txt_dir = os.path.join(OUTPUT_ROOT, 'ImageSets')
    os.makedirs(txt_dir, exist_ok=True)

    with open(os.path.join(txt_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join([os.path.splitext(fname)[0] for fname in train_files]))
    with open(os.path.join(txt_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join([os.path.splitext(fname)[0] for fname in val_files]))
    with open(os.path.join(txt_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join([os.path.splitext(fname)[0] for fname in test_files]))

    # 6. 复制文件
    def copy_files(file_list, mode):
        for fname in tqdm(file_list, desc=mode):
            name = os.path.splitext(fname)[0]
            # 图片
            src_img = os.path.join(IMG_DIR, fname)
            dst_img = os.path.join(OUTPUT_ROOT, 'images', mode, fname)
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            # 标签
            src_lab = os.path.join(LABEL_DIR, name + '.txt')
            dst_lab = os.path.join(OUTPUT_ROOT, 'labels', mode, name + '.txt')
            if os.path.exists(src_lab):
                shutil.copy(src_lab, dst_lab)

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print("\n✅ 数据集划分完成！数量完全符合：650 / 50 / 100")

if __name__ == "__main__":
    split_dataset()

