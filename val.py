from ultralytics import YOLO

# 关键：添加 Windows 多进程保护的主程序入口
if __name__ == '__main__':
    # 优化1：删除重复的模型加载（原代码加载2次模型，第二次会覆盖第一次，保留自定义训练的best.pt即可）
    # 加载你训练好的自定义模型（路径已按你的代码保留，确保该路径下有best.pt文件）
    model = YOLO(r"F:\item\LPD-Net\runs\detect\train35\weights\best.pt")
    results = model.val()
    # 优化2：添加 workers=0（Windows 下强制单进程加载数据，彻底解决多进程启动冲突）
    # 若训练时已指定数据集（如train3的args中记录了data路径），可不用显式写data参数；若报错“找不到数据集”，需补充data='你的data.yaml路径'
    metrics = model.val(
        workers=0,  # 核心修复：Windows 多进程冲突的关键参数
        batch=4  # 可选：根据你的 RTX 2060 SUPER 显存调整（8G显存建议4-8，避免溢出）
    )

    # 优化3：添加打印逻辑，直观查看验证结果（原代码仅定义变量，不打印无法看到结果）
    print("=" * 50)
    print("验证结果汇总：")
    print(f"mAP50-95: {metrics.box.map:.4f}")  # 所有类别的平均mAP（50-95 IoU）
    print(f"mAP50:     {metrics.box.map50:.4f}")  # 所有类别的mAP（50 IoU）`
    print(f"mAP75:     {metrics.box.map75:.4f}")  # 所有类别的mAP（75 IoU）
    print(f"各类别mAP: {[round(m, 4) for m in metrics.box.maps]}")  # 每个类别的mAP50-95
    print("=" * 50)
