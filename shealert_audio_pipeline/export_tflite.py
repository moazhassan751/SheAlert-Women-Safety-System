"""SheAlert TFLite Export & Latency Benchmarking (LM-5 On-Device Deployment).

Converts the trained Keras safety classifier head into optimized TensorFlow Lite
binaries (FP32 baseline and FP16 half-precision) and benchmarks CPU inference latency.
"""

import sys
import argparse
import time
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

import config


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained Keras safety head to TFLite.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=config.OUTPUT_DIR / "models" / "stage1_head_best.keras",
        help="Path to trained Keras model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR / "models",
        help="Output directory for .tflite models.",
    )
    parser.add_argument(
        "--num-benchmark-runs",
        type=int,
        default=100,
        help="Number of iterations for CPU latency benchmarking.",
    )
    return parser.parse_args()


def benchmark_tflite_model(tflite_bytes: bytes, input_shape: tuple, num_runs: int = 100) -> float:
    """Measures mean inference latency in milliseconds using TFLite Interpreter."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup runs
    for _ in range(10):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])

    # Timed runs
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model_path.exists():
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    print("=" * 65)
    print("SheAlert TFLite Export & Latency Benchmarking (LM-5)")
    print(f"Source Model: {args.model_path.name}")
    print("=" * 65)

    keras_model = tf.keras.models.load_model(str(args.model_path))
    input_shape = (1, keras_model.input_shape[-1])
    print(f"Model Input Shape: {input_shape} (Batch=1 for on-device inference)")

    # 1. Export FP32 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_fp32 = converter.convert()
    fp32_path = args.output_dir / "stage1_head_fp32.tflite"
    with open(fp32_path, "wb") as f:
        f.write(tflite_fp32)
    fp32_size_kb = len(tflite_fp32) / 1024.0
    print(f"Exported FP32 TFLite: {fp32_path.name} ({fp32_size_kb:.2f} KB)")

    # 2. Export FP16 Quantized TFLite
    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter_fp16.convert()
    fp16_path = args.output_dir / "stage1_head_fp16.tflite"
    with open(fp16_path, "wb") as f:
        f.write(tflite_fp16)
    fp16_size_kb = len(tflite_fp16) / 1024.0
    print(f"Exported FP16 TFLite: {fp16_path.name} ({fp16_size_kb:.2f} KB)")

    # 3. Benchmark Latencies
    print(f"\nBenchmarking inference latency on local CPU ({args.num_benchmark_runs} runs)...")
    lat_fp32 = benchmark_tflite_model(tflite_fp32, input_shape, args.num_benchmark_runs)
    lat_fp16 = benchmark_tflite_model(tflite_fp16, input_shape, args.num_benchmark_runs)

    print(f"  FP32 TFLite Latency : {lat_fp32:.3f} ms per prediction")
    print(f"  FP16 TFLite Latency : {lat_fp16:.3f} ms per prediction")

    benchmark_info = {
        "source_model": args.model_path.name,
        "input_feature_dim": keras_model.input_shape[-1],
        "fp32_model_size_kb": fp32_size_kb,
        "fp16_model_size_kb": fp16_size_kb,
        "fp32_latency_ms": lat_fp32,
        "fp16_latency_ms": lat_fp16,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    bench_path = args.output_dir / "tflite_benchmark.json"
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_info, f, indent=2)
    print(f"\nSaved benchmark metrics to {bench_path}")


if __name__ == "__main__":
    main()
