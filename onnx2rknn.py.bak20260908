import os
import sys
import argparse
from rknn.api import RKNN

# 模型默认路径
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "official_models")
MODELS = {
    "det": os.path.join(MODEL_DIR, "PP-OCRv6_medium_det_onnx", "inference.onnx"),
    "cls": os.path.join(MODEL_DIR, "PP-LCNet_x1_0_textline_ori_onnx", "inference.onnx"),
    "rec": os.path.join(MODEL_DIR, "PP-OCRv6_medium_rec_onnx", "inference.onnx"),
}

# 各模型动态输入配置（高度/宽度可按需调整）
DYNAMIC_INPUTS = {
    # det: 全动态，提供常见分辨率档位
    "det": [
        [[1, 3, 320, 320]],
        [[1, 3, 640, 640]],
    ],
    # cls: 固定输入，无需动态配置
    "cls": None,
    # rec: 高度固定48，宽度多档（与 export_rk.py 保持一致）
    "rec": [
        [[1, 3, 48, 160]],
        [[1, 3, 48, 320]],
        [[1, 3, 48, 640]],
    ],
}

# 需要放到 CPU 执行的算子（NPU 不支持或精度有问题时使用）
# exSoftmax13 仅存在于 rec 模型中
OP_TARGET = {
    "det": None,
    "cls": None,
    "rec": {"exSoftmax13": "cpu"},
}

# ONNX 模型中 batch 维度为动态时，需要在 load_onnx 时显式指定固定形状
LOAD_SHAPES = {
    "cls": {"inputs": ["x"], "input_size_list": [[1, 3, 80, 160]]},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="PP-OCRv6 ONNX to RKNN converter (det/cls/rec)"
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="rk3588",
        choices=["rk3562", "rk3566", "rk3568", "rk3588", "rk3576"],
        help="Target Rockchip platform (default: rk3588)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp",
        choices=["i8", "u8", "fp"],
        help="Quantization type: i8/u8 for INT8 quantization, fp for float (default: fp)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Path to calibration dataset file for quantization (required when dtype is i8/u8)",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["det", "cls", "rec"],
        choices=["det", "cls", "rec", "all"],
        help="Which models to convert (default: all three)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rknn"),
        help="Output directory for .rknn files",
    )
    return parser.parse_args()


def convert_model(model_name, onnx_path, platform, do_quant, dataset, output_dir):
    """转换单个模型为 RKNN 格式"""
    output_path = os.path.join(output_dir, f"{model_name}.rknn")

    print(f"\n{'='*50}")
    print(f"  Converting: {model_name}")
    print(f"  ONNX:  {onnx_path}")
    print(f"  Output: {output_path}")
    print(f"  Quant:  {do_quant}")
    print(f"{'='*50}")

    rknn = RKNN(verbose=False)

    # 1. 配置
    print(f"-- Configuring {model_name}...")
    config_kwargs = {
        "target_platform": platform,
    }
    if OP_TARGET.get(model_name):
        config_kwargs["op_target"] = OP_TARGET[model_name]
    if DYNAMIC_INPUTS[model_name] is not None:
        config_kwargs["dynamic_input"] = DYNAMIC_INPUTS[model_name]

    ret = rknn.config(**config_kwargs)
    if ret != 0:
        print(f"ERROR: Config {model_name} failed!")
        return False

    # 2. 加载 ONNX 模型
    print(f"-- Loading {model_name}...")
    load_kwargs = {"model": onnx_path}
    if model_name in LOAD_SHAPES:
        load_kwargs.update(LOAD_SHAPES[model_name])
    ret = rknn.load_onnx(**load_kwargs)
    if ret != 0:
        print(f"ERROR: Load {model_name} failed!")
        return False

    # 3. 构建（可选量化）
    print(f"-- Building {model_name}...")
    ret = rknn.build(do_quantization=do_quant, dataset=dataset)
    if ret != 0:
        print(f"ERROR: Build {model_name} failed!")
        return False

    # 4. 导出
    print(f"-- Exporting {model_name}...")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"ERROR: Export {model_name} failed!")
        return False

    rknn.release()
    print(f"-- {model_name} done -> {output_path}")
    return True


def main():
    args = parse_args()

    # 确定要转换的模型列表
    if "all" in args.model:
        model_list = ["det", "cls", "rec"]
    else:
        model_list = args.model

    # 量化相关校验
    do_quant = args.dtype in ("i8", "u8")
    if do_quant and not args.dataset:
        print("ERROR: --dataset is required when dtype is i8 or u8")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 逐个转换
    results = {}
    for name in model_list:
        onnx_path = MODELS[name]
        if not os.path.isfile(onnx_path):
            print(f"ERROR: ONNX model not found: {onnx_path}")
            results[name] = False
            continue
        results[name] = convert_model(
            name, onnx_path, args.platform, do_quant, args.dataset, args.output_dir
        )

    # 汇总
    print(f"\n{'='*50}")
    print("  Results:")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"    {name}: {status}")
    print(f"{'='*50}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
