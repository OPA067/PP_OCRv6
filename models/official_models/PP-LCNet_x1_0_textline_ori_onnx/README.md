---
license: apache-2.0
library_name: PaddleOCR
language:
- en
- zh
pipeline_tag: image-to-text
tags:
- OCR
- PaddlePaddle
- PaddleOCR
- textline_orientation_classification
---

# PP-LCNet_x1_0_textline_ori

## Introduction

The text line orientation classification module primarily distinguishes the orientation of text lines and corrects them using post-processing. In processes such as document scanning and license/certificate photography, to capture clearer images, the capture device may be rotated, resulting in text lines in various orientations. Standard OCR pipelines cannot handle such data well. By utilizing image classification technology, the orientation of text lines can be predetermined and adjusted, thereby enhancing the accuracy of OCR processing. The key accuracy metrics are as follow:

<table>
<tr>
<th>Model</th>
<th>Recognition Avg Accuracy(%)</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_textline_ori</td>
<td>98.85</td>
<td>0.96</td>
<td>Text line classification model based on PP-LCNet_x0_25, with two classes: 0 degrees and 180 degrees</td>
</tr>
</table>

## Model Usage

### Install Dependencies

```shell
pip install -U paddleocr
pip install -U onnxruntime-gpu
```

### CLI Usage

```shell
paddleocr textline_orientation_classification -i ./demo.jpg --model_name PP-LCNet_x1_0_textline_ori --engine onnxruntime
```

### Python API Usage

```python
from paddleocr import TextLineOrientationClassification

model = TextLineOrientationClassification(
    model_name="PP-LCNet_x1_0_textline_ori",
    engine="onnxruntime",
)
output = model.predict("./demo.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_json(save_path="./output/res.json")
```