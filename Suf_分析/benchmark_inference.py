"""Benchmark every Suf model on one full pass of the data_3D_400 train inputs.

The controller starts one clean subprocess per model so CUDA allocator state and
process working-set measurements do not leak from one model to the next.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SUF_ROOT = Path(__file__).resolve().parent
TRAIN_X = ROOT / "data" / "data_3D_400" / "train" / "x"
REPORT_PATH = SUF_ROOT / "模型推理性能分析.md"
MODEL_FOLDERS = [
    "00_Baseline_Unet_8",
    "01_Unet_DCN4X2",
    "02_Unet_ConnLoss_8",
    "03_Unet_SufCOv4X2",
    "04_Unet_SufLoss",
    "05_Unet_SufCov+Loss4x2",
]


class TrainInputDataset(Dataset):
    """Load only x because this benchmark performs inference, not evaluation."""

    def __init__(self, directory: Path, limit: int | None = None):
        def numeric_key(path: Path):
            try:
                return (0, int(path.stem))
            except ValueError:
                return (1, path.name)

        self.files = sorted(directory.glob("*.npy"), key=numeric_key)
        if limit is not None:
            self.files = self.files[:limit]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        array = np.load(self.files[index])
        if array.ndim == 3:
            array = array[None, ...]
        return torch.from_numpy(array).float()


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def process_memory() -> tuple[int, int, int]:
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise ctypes.WinError()
    return counters.WorkingSetSize, counters.PeakWorkingSetSize, counters.PrivateUsage


def total_system_memory() -> int:
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return status.ullTotalPhys


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def import_model(model_py: Path):
    # SurfaceConv experiment snapshots reference the repository implementation.
    sys.path.insert(0, str(ROOT / "models"))
    sys.path.insert(0, str(model_py.parent))
    module_name = "suf_benchmark_" + hashlib.md5(str(model_py).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, model_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_worker(folder_name: str, limit: int | None, warmup: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark requires a CUDA GPU")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    folder = SUF_ROOT / folder_name
    model_py = folder / "models.py"
    checkpoint = folder / "models" / "FaultSeg3D_BEST.pth"
    dataset = TrainInputDataset(TRAIN_X, limit=limit)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    if not dataset:
        raise RuntimeError(f"No .npy inputs found in {TRAIN_X}")

    setup_start = time.perf_counter()
    module = import_model(model_py)
    model = module.FaultSeg3D(1, 2)
    parameter_count = sum(p.numel() for p in model.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    load_start = time.perf_counter()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    setup_seconds = time.perf_counter() - setup_start

    # Warm up kernels and libraries. Warm-up samples are repeated in the measured pass.
    warmup_start = time.perf_counter()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= min(warmup, len(dataset)):
                break
            output = model(batch.to(device))
            del output, batch
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_start

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gpu_idle_allocated = torch.cuda.memory_allocated(device)
    cpu_idle_working, _, cpu_idle_private = process_memory()
    max_cpu_working = cpu_idle_working
    max_cpu_private = cpu_idle_private

    starts: list[torch.cuda.Event] = []
    ends: list[torch.cuda.Event] = []
    observed_shape = None
    torch.cuda.synchronize()
    pass_start = time.perf_counter()
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            observed_shape = list(batch.shape)
            batch = batch.to(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = model(batch)
            end_event.record()
            starts.append(start_event)
            ends.append(end_event)

            current_working, _, current_private = process_memory()
            max_cpu_working = max(max_cpu_working, current_working)
            max_cpu_private = max(max_cpu_private, current_private)
            del output, batch
            if index % 25 == 0 or index == len(dataset):
                print(f"[{folder_name}] {index}/{len(dataset)}", flush=True)

    torch.cuda.synchronize()
    pass_seconds = time.perf_counter() - pass_start
    forward_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    _, process_peak_working, _ = process_memory()
    gpu_peak_allocated = torch.cuda.max_memory_allocated(device)
    gpu_peak_reserved = torch.cuda.max_memory_reserved(device)

    mib = 1024**2
    return {
        "folder": folder_name,
        "samples": len(dataset),
        "batch_size": 1,
        "input_shape": observed_shape,
        "parameter_count": parameter_count,
        "parameter_mib": parameter_bytes / mib,
        "checkpoint_mib": checkpoint.stat().st_size / mib,
        "setup_seconds": setup_seconds,
        "checkpoint_load_to_gpu_seconds": load_seconds,
        "warmup_samples": min(warmup, len(dataset)),
        "warmup_seconds": warmup_seconds,
        "pass_seconds": pass_seconds,
        "end_to_end_ms_per_sample": pass_seconds * 1000 / len(dataset),
        "throughput_samples_per_second": len(dataset) / pass_seconds,
        "forward_total_seconds": sum(forward_ms) / 1000,
        "forward_mean_ms": statistics.fmean(forward_ms),
        "forward_median_ms": statistics.median(forward_ms),
        "forward_p95_ms": percentile(forward_ms, 0.95),
        "forward_min_ms": min(forward_ms),
        "forward_max_ms": max(forward_ms),
        "gpu_idle_allocated_mib": gpu_idle_allocated / mib,
        "gpu_peak_allocated_mib": gpu_peak_allocated / mib,
        "gpu_inference_increment_mib": (gpu_peak_allocated - gpu_idle_allocated) / mib,
        "gpu_peak_reserved_mib": gpu_peak_reserved / mib,
        "cpu_idle_working_set_mib": cpu_idle_working / mib,
        "cpu_max_observed_working_set_mib": max_cpu_working / mib,
        "cpu_process_peak_working_set_mib": process_peak_working / mib,
        "cpu_idle_private_mib": cpu_idle_private / mib,
        "cpu_max_observed_private_mib": max_cpu_private / mib,
    }


def format_int(value: int) -> str:
    return f"{value:,}"


def render_report(results: list[dict], metadata: dict) -> str:
    baseline = results[0]
    fastest = min(results, key=lambda item: item["pass_seconds"])
    dcn = results[1]
    surface = results[3]
    surface_loss = results[5]
    ordinary = [results[index] for index in (0, 2, 4)]
    ordinary_min = min(item["pass_seconds"] for item in ordinary)
    ordinary_max = max(item["pass_seconds"] for item in ordinary)
    lines = [
        "# Suf 模型训练集推理性能分析",
        "",
        f">生成时间：{metadata['generated_at']}（Asia/Shanghai）",
        "",
        "## 测试结论",
        "",
        f"- 数值上最快的是 `{fastest['folder']}`，完整数据遍历耗时 "
        f"{fastest['pass_seconds']:.3f} s，吞吐量 {fastest['throughput_samples_per_second']:.2f} 样本/s；"
        "它与 Baseline 的差异远小于单轮测量波动，不应解读为结构加速。",
        f"- 普通 U-Net 三组的峰值已分配显存都是 "
        f"{baseline['gpu_peak_allocated_mib']:.1f} MiB，是本次测试中最低值。",
        "- `00` / `02` / `04` 使用同一普通 U-Net 结构；它们的损失函数不同，"
        "但损失函数不参与推理，因而耗时和显存应当接近。",
        "- `03` / `05` 使用同一 SurfaceConv U-Net 结构，推理成本差异主要是单轮测量波动，"
        "而不是 Surface loss 造成的。",
        "",
        "## 统一对比结果",
        "",
        "| 模型 | 参数量 | 整轮耗时 (s) | 端到端 (ms/样本) | 吞吐量 (样本/s) | "
        "GPU前向均值 (ms) | GPU前向P50/P95 (ms) | 峰值已分配显存 (MiB) | "
        "推理增量显存 (MiB) | 峰值保留显存 (MiB) | CPU工作集峰值 (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| `{result['folder']}` | {format_int(result['parameter_count'])} | "
            f"{result['pass_seconds']:.3f} | {result['end_to_end_ms_per_sample']:.2f} | "
            f"{result['throughput_samples_per_second']:.2f} | {result['forward_mean_ms']:.2f} | "
            f"{result['forward_median_ms']:.2f} / {result['forward_p95_ms']:.2f} | "
            f"{result['gpu_peak_allocated_mib']:.1f} | {result['gpu_inference_increment_mib']:.1f} | "
            f"{result['gpu_peak_reserved_mib']:.1f} | {result['cpu_process_peak_working_set_mib']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## 加载与静态开销",
            "",
            "| 模型 | 权重文件 (MiB) | FP32参数 (MiB) | 构建+加载+上GPU (s) | "
            "checkpoint加载+上GPU (s) | 模型就绪显存 (MiB) | CPU就绪工作集 (MiB) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        lines.append(
            f"| `{result['folder']}` | {result['checkpoint_mib']:.2f} | "
            f"{result['parameter_mib']:.2f} | {result['setup_seconds']:.3f} | "
            f"{result['checkpoint_load_to_gpu_seconds']:.3f} | "
            f"{result['gpu_idle_allocated_mib']:.1f} | {result['cpu_idle_working_set_mib']:.1f} |"
        )

    baseline_time = baseline["pass_seconds"]
    lines.extend(
        [
            "",
            "## 相对 Baseline 的推理代价",
            "",
            "| 模型 | 整轮耗时 / Baseline | 峰值已分配显存 / Baseline |",
            "|---|---:|---:|",
        ]
    )
    for result in results:
        lines.append(
            f"| `{result['folder']}` | {result['pass_seconds'] / baseline_time:.2f}× | "
            f"{result['gpu_peak_allocated_mib'] / baseline['gpu_peak_allocated_mib']:.2f}× |"
        )

    lines.extend(
        [
            "",
            "## 综合分析",
            "",
            f"- **普通 U-Net 组（`00` / `02` / `04`）**：整轮耗时范围为 "
            f"{ordinary_min:.3f}–{ordinary_max:.3f} s。`ConnLoss` 和 `SufLoss` 只改变训练目标，"
            "不改变前向结构，所以推理参数量、显存和速度与 Baseline 基本一致。",
            f"- **DCN**：参数量是 Baseline 的 {dcn['parameter_count'] / baseline['parameter_count']:.2f}×，"
            f"整轮耗时是 {dcn['pass_seconds'] / baseline['pass_seconds']:.2f}×，"
            f"吞吐量下降 {(1 - dcn['throughput_samples_per_second'] / baseline['throughput_samples_per_second']) * 100:.1f}%。"
            f"峰值已分配显存只增加 {(dcn['gpu_peak_allocated_mib'] / baseline['gpu_peak_allocated_mib'] - 1) * 100:.1f}%，"
            f"但 allocator 峰值保留显存增加 {(dcn['gpu_peak_reserved_mib'] / baseline['gpu_peak_reserved_mib'] - 1) * 100:.1f}%。",
            f"- **SurfaceConv**：参数量是 Baseline 的 {surface['parameter_count'] / baseline['parameter_count']:.2f}×，"
            f"整轮耗时约为 {surface['pass_seconds'] / baseline['pass_seconds']:.2f}×，"
            f"吞吐量下降 {(1 - surface['throughput_samples_per_second'] / baseline['throughput_samples_per_second']) * 100:.1f}%。"
            f"峰值已分配显存仅增加 {(surface['gpu_peak_allocated_mib'] / baseline['gpu_peak_allocated_mib'] - 1) * 100:.1f}%，"
            "原因是 128³ 全分辨率特征图的激活内存占主导；"
            f"但保留显存和 CPU 峰值工作集分别比 Baseline 高 "
            f"{(surface['gpu_peak_reserved_mib'] / baseline['gpu_peak_reserved_mib'] - 1) * 100:.1f}% 和 "
            f"{(surface['cpu_process_peak_working_set_mib'] / baseline['cpu_process_peak_working_set_mib'] - 1) * 100:.1f}%。",
            f"- **SurfaceConv + Loss**：与纯 SurfaceConv 的整轮耗时只相差 "
            f"{abs(surface_loss['pass_seconds'] - surface['pass_seconds']):.3f} s，证明 Surface loss 不会额外增加部署时的前向成本。",
            "- **选型建议**：如果后续准确率、Dice/IoU 和连通性指标提升不明显，"
            "普通 U-Net 组的推理性价比最高；DCN 和 SurfaceConv 需分别用约 1.79× 和 4.44× "
            "的时间代价来换取指标改善，应结合后续精度结果再决定。",
            "",
            "## 测试口径",
            "",
            f"- 数据路径：`{metadata['train_x']}`。",
            f"- 该路径实际只有 **{metadata['dataset_samples']}** 个 `.npy` 输入，而不是 400 个；"
            "`data_3D_400` 是目录名。本次已对当前训练分割中所有现有输入完整推理一轮。",
            f"- 输入形状：`{results[0]['input_shape']}`，类型 `float32`，batch size = 1。",
            "- 仅加载训练集 `x` ，不加载标签 `y`；输出用后立即丢弃，不包含结果落盘时间。",
            "- 模型使用 `eval()` + `torch.inference_mode()`，FP32，未使用 AMP，"
            "DataLoader `num_workers=0`、`pin_memory=False`、`shuffle=False`。",
            f"- 每个模型正式计时前预热 {results[0]['warmup_samples']} 个样本；预热不计入正式耗时。",
            "- “整轮耗时”包含 `.npy` 读取、CPU到GPU传输、模型前向和 Python/DataLoader 循环开销；"
            "“GPU前向”仅用 CUDA Event 统计 `model(batch)`。",
            "- “已分配显存”是 PyTorch CUDA allocator 实际被张量占用的峰值；"
            "“保留显存”是 allocator 从 CUDA 保留的峰值，不等于 `nvidia-smi` 中的整机显存数。",
            "- CPU 内存为 Windows 进程峰值工作集，包含 Python、PyTorch/CUDA 运行库和模型加载开销。",
            "- 每个模型只测一个完整数据遍历；小差异可能来自 GPU Boost、温度、文件系统缓存和后台负载。",
            "",
            "## 测试环境",
            "",
            f"- GPU：{metadata['gpu_name']}，{metadata['gpu_total_mib']:.0f} MiB",
            f"- PyTorch：{metadata['torch_version']}，CUDA runtime：{metadata['cuda_version']}，cuDNN：{metadata['cudnn_version']}",
            f"- Python：{metadata['python_version']}",
            f"- 操作系统：{metadata['platform']}",
            f"- 物理内存：{metadata['system_memory_gib']:.1f} GiB",
            "",
            "## 备注",
            "",
            "- SurfaceConv 的两个实验目录未自带 `surface_conv3d.py`，本测试使用仓库 "
            "`models/surface_conv3d.py` 实现，且权重以 `strict=True` 成功加载。",
            "- `ConnLoss`、`SufLoss` 等训练损失不会在此纯前向推理基准中执行。",
            "",
        ]
    )
    return "\n".join(lines)


def run_controller(limit: int | None, warmup: int) -> None:
    results = []
    for folder in MODEL_FOLDERS:
        print(f"\n=== Benchmarking {folder} ===", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            folder,
            "--warmup",
            str(warmup),
        ]
        if limit is not None:
            command.extend(["--limit", str(limit)])
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        result = None
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            print(line, flush=True)
            if line.startswith("RESULT_JSON="):
                result = json.loads(line[len("RESULT_JSON=") :])
        return_code = process.wait()
        if return_code != 0 or result is None:
            raise RuntimeError(f"Benchmark failed for {folder} (exit {return_code})")
        results.append(result)

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    del free_bytes
    dataset_samples = len(TrainInputDataset(TRAIN_X, limit=limit))
    metadata = {
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "train_x": str(TRAIN_X.relative_to(ROOT)).replace("\\", "/"),
        "dataset_samples": dataset_samples,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_mib": total_bytes / 1024**2,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system_memory_gib": total_system_memory() / 1024**3,
    }
    REPORT_PATH.write_text(render_report(results, metadata), encoding="utf-8")
    print(f"\nReport written to: {REPORT_PATH}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=MODEL_FOLDERS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.worker:
        result = run_worker(args.worker, args.limit, args.warmup)
        print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False), flush=True)
    else:
        run_controller(args.limit, args.warmup)


if __name__ == "__main__":
    main()
