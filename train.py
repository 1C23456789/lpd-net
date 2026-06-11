
from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
import warnings
warnings.filterwarnings('ignore', message='torch.meshgrid: in an upcoming release')


if __name__=="__main__":

    # 使用自己的YOLOv8.yamy文件搭建模型并加载预训练权重训练模型
    model = YOLO(r"F:\item\LPD-Net\LPDNet.yaml")

    results = model.train(

        data=r"HazyDet.yaml",

        epochs=500,

        imgsz=640,

        batch=4,

        amp=True,

        # 雾天友好增强

        hsv_h=0.015,

        hsv_s=0.7,

        hsv_v=0.5,  # 更强的明度扰动

        degrees=10.0,

        translate=0.1,

        scale=0.5,

        shear=2.0,

        perspective=0.001,

        flipud=0.0,  # 关闭上下翻转

        fliplr=0.5,

        mosaic=1.0,

        mixup=0.1,

    )
