from paddleocr import PaddleOCR
import os
import time

# 获取当前脚本所在目录，并构建模型路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models", "official_models")

ocr = PaddleOCR(
    # PP-OCRv6 medium
    text_detection_model_name="PP-OCRv6_medium_det",
    text_detection_model_dir=os.path.join(MODELS_DIR, "PP-OCRv6_medium_det_onnx"),
    text_recognition_model_name="PP-OCRv6_medium_rec",
    text_recognition_model_dir=os.path.join(MODELS_DIR, "PP-OCRv6_medium_rec_onnx"),

    # 开启文本行方向分类
    use_textline_orientation=True,
    textline_orientation_model_dir=os.path.join(MODELS_DIR, "PP-LCNet_x1_0_textline_ori_onnx"),

    # 不需要的功能关闭
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,

    # 使用 ONNX Runtime
    engine="onnxruntime",
)

# region single image mode
image_path = "benchmark/a1.png"

# 识别一张
t0 = time.time()
result = ocr.predict(image_path)
t1 = time.time()
elapsed = t1 - t0

print(f"文件名: {image_path}")
print(f"采样时间: {elapsed:.4f}s")

for res in result:
    # 打印识别到的文本和坐标
    for text, box, score in zip(res["rec_texts"], res["rec_boxes"], res["rec_scores"]):
        print(f"  文本: {text}, 坐标: {box}, 置信度: {score:.4f}")

    # 或简写为
    # print(res["rec_texts"])   # 所有文本
    # print(res["rec_scores"])  # 所有分数

for res in result:
    # res.print()
    res.save_to_img(save_path="./output")
    # res.save_to_json(save_path="./output")

print()

# region batch image mode — 处理 benchmark 目录下全部图片
import glob

image_dir = "benchmark"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

image_paths = []
for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
    image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

print(f"共 {len(image_paths)} 张图片，开始循环识别...\n")

sample_times = []
for img_path in image_paths:
    # 识别一张
    t0 = time.time()
    result = ocr.predict(img_path)
    t1 = time.time()

    elapsed = t1 - t0
    sample_times.append(elapsed)

    filename = os.path.basename(img_path)
    print(f"文件名: {filename}")
    print(f"采样时间: {elapsed:.4f}s")

    # 输出该张图片的识别结果
    for res in result:
        for text, box, score in zip(res["rec_texts"], res["rec_boxes"], res["rec_scores"]):
            print(f"  文本: {text}, 坐标: {box}, 置信度: {score:.4f}")
        # res.print()
        res.save_to_img(save_path=output_dir)
        # res.save_to_json(save_path=output_dir)

    print()

avg_time = sum(sample_times) / len(sample_times) if sample_times else 0
print(f"总耗时: {sum(sample_times):.4f}s，平均一张图采样时间: {avg_time:.4f}s")
