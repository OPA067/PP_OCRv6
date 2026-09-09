<div align="center">

# 🔍 PP-OCRv6 ONNX + RKNN Deployment

**v1.0 — Stable Release** 🎉

A practical OCR inference and conversion toolkit built on [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) PP-OCRv6 models.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-v6-red)](https://github.com/PaddlePaddle/PaddleOCR)

</div>

---

## 📋 Overview

Two runtime modes are supported:

| Mode | Use Case | Engine |
|------|----------|--------|
| 🖥️ **ONNX** | PC-side quick preview & validation | `onnxruntime` |
| 🚀 **RKNN** | Rockchip edge deployment (NPU acceleration) | `rknn-toolkit2` |

**Pipeline:** Image → 📐 Text Detection (`det`) → 🔄 Orientation Classification (`cls`) → 📝 Text Recognition (`rec`) → ✅ Result

---

## 🖼️ Inference Demo

Below are sample outputs showing text detection (colored bounding boxes) and recognition results on diverse inputs:

#### 1. 💻 Code snippet

![b9 — low-quality OCR](output/b9_ocr_res_img.png)

> `sudo watch -n 0.5 '`
> `echo "====== NPU ======"`
> `cat /sys/kernel/debug/rknpu/load`
> `echo "====== NPU FREQ ======"`
> `cat /sys/class/devfreq/fdab0000.npu/cur_freq`

#### 2. 🌐 Mixed CJK + English

![a5 — code snippet OCR](output/a5_ocr_res_img.png)

> `benchmark - 基准常见释义`

#### 3. 🪪 ID card

![b8 — ID card OCR](output/b8_ocr_res_img.png)

> `中华人民共和国` · `居民身份证` · `签发机关 上海市公安局徐汇分局` · `有效期限 2005.10.08-2025.10.08`

#### 4. 📉 Low-quality / blurry

![a8 — mixed CJK+English OCR](output/a8_ocr_res_img.png)
> `现在你的代码状态`（upside-down input, auto-rotated）

---

## 📂 Project Structure

```
PP-OCRv6/
├── ocr_infer.py              # 🎯 Main inference entry: single-image + batch recognition with timing
├── onnx2rknn.py              # 🔄 ONNX → RKNN conversion script
├── requirements.txt          # 📦 Python dependencies
│
├── benchmark/                # 🧪 Test image set (20 images: a1-a10, b1-b10)
├── output/                   # 📸 Recognition result images (bounding boxes + recognized text)
│
└── models/
    ├── official_models/      # 🏛️ PaddlePaddle / ONNX official models
    │   ├── PP-OCRv6_medium_det/         # 📐 Text detection model
    │   ├── PP-OCRv6_medium_rec/         # 📝 Text recognition model
    │   └── PP-LCNet_x1_0_textline_ori/  # 🔄 Textline orientation classifier
    └── rknn/                 # ⚡ Converted .rknn models (det.rknn, cls.rknn, rec.rknn)
```

**🧩 Model roles:**

| Abbreviation | Full Name | Purpose |
|:---:|---------|---------|
| `det` | 📐 Text Detection | Locates text regions in the image and outputs bounding boxes |
| `cls` | 🔄 Textline Orientation Classification | Determines if a text line is rotated 180° and corrects it |
| `rec` | 📝 Text Recognition | Reads the text content within each detected region |

---

## ⚡ Quick Start

### 📦 Prerequisites

```bash
pip install paddleocr onnxruntime
```

### 🖼️ Single Image Inference

```bash
python ocr_infer.py
```

By default, this runs inference on `benchmark/a1.png` and outputs:
- 📌 Per-text bounding box coordinates, recognized text, and confidence score
- ⏱️ Total inference time for the image
- 💾 Result visualization saved to `output/a1_ocr_res_img.png`

### 📂 Batch Inference

The script automatically scans `benchmark/` for `*.png / *.jpg / *.jpeg / *.bmp` files and processes them sequentially:

```bash
python ocr_infer.py
```

**📄 Output format:**

```
文件名: a1.png
采样时间: 0.2341s
  文本: 你好, 坐标: [[...]], 置信度: 0.9876

共 20 张图片，开始循环识别...

文件名: a2.png
采样时间: 0.1987s
  文本: 测试文本, 坐标: [[...]], 置信度: 0.9532
  ...

总耗时: 4.3728s，平均一张图采样时间: 0.2186s
```

**🔑 Key behaviors:**
- ⚙️ Each image is processed independently via `ocr.predict()`
- 📊 Per-image timing, detected text, bounding boxes, and confidence scores are printed
- 📈 Final summary shows total elapsed time and average time per image
- 💾 Result images with overlaid bounding boxes are saved to `output/`

---

## 🔄 RKNN Conversion

Convert ONNX models to Rockchip RKNN format for NPU-accelerated edge inference.

**🏗️ Supported platforms:** `rk3562`, `rk3566`, `rk3568`, `rk3576`, `rk3588`

### 📦 Install RKNN Toolkit

```bash
pip install rknn-toolkit2   # 🖥️ PC-side (x86_64) compilation environment
```

### 🚀 Basic Usage

```bash
# 🔄 Convert all three models (default: RK3588, float precision)
python onnx2rknn.py

# 🎯 Convert specific models only
python onnx2rknn.py --model det rec

# 📊 INT8 quantization (requires calibration dataset)
python onnx2rknn.py --platform rk3588 --dtype i8 --dataset ./calib_data.txt

# 🔢 Float mode (default)
python onnx2rknn.py --platform rk3588 --dtype fp
```

### 📋 Script Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--platform` | `rk3588` | 🎯 Target Rockchip chip platform |
| `--dtype` | `fp` | 📊 Quantization type: `fp` (float), `i8` (INT8), `u8` (UINT8) |
| `--dataset` | `""` | 📂 Calibration dataset path (required for `i8`/`u8` quantization) |
| `--model` | `det cls rec` | 🧩 Which models to convert |
| `--output_dir` | `models/rknn` | 📁 Output directory for `.rknn` files |

### 📐 Dynamic Input Configuration

Different models require different input shape strategies:

```python
# det (detection): fully dynamic, common resolution tiers
[[1, 3, 320, 320]], [[1, 3, 640, 640]]

# rec (recognition): fixed height 48, variable width
[[1, 3, 48, 160]], [[1, 3, 48, 320]], [[1, 3, 48, 640]]

# cls (classification): fixed input
[[1, 3, 80, 160]]
```

### ⚙️ Operator Handling

The `exSoftmax13` operator in the recognition model is offloaded to **CPU** to avoid NPU precision issues:

```python
OP_TARGET = {
    "det": None,
    "cls": None,
    "rec": {"exSoftmax13": "cpu"},
}
```

---

## 📥 Model Download

Models are automatically downloaded on first run via `paddleocr` and cached to `models/official_models/`:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv6_medium_det",
    text_recognition_model_name="PP-OCRv6_medium_rec",
    use_textline_orientation=True,
)
```

| Model | Link |
|-------|------|
| PP-OCRv6_medium_det | 📐 [PaddleOCR Docs](https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html) |
| PP-OCRv6_medium_rec | 📝 [PaddleOCR Docs](https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html) |
| PP-LCNet_x1_0_textline_ori | 🔄 [PaddleOCR Docs](https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html) |

---

## 🛠️ Tech Stack

| Component | Role |
|-----------|------|
| **PaddleOCR v6** | 🔍 OCR pipeline (detection + orientation classification + recognition) |
| **ONNX Runtime** | 🖥️ CPU-side inference engine for PC preview |
| **RKNN Toolkit2** | ⚡ Rockchip NPU model compilation and runtime |
| **PP-OCRv6_medium** | ⚖️ Balanced accuracy/speed mid-size model from PaddlePaddle |

---

<div align="center">

## 📄 License

[Apache 2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)

</div>
