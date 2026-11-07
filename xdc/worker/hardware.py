"""Cross-platform hardware detection and honest backend fallback."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import psutil
import torch

from ..protocol import Capabilities


@dataclass
class Detected:
    selected: str
    device_name: str
    backends: list[dict[str, Any]]
    gpu_memory_mb: int = 0
    notes: list[str] = field(default_factory=list)


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=timeout).decode("utf-8", "replace").strip()
    except Exception:
        return ""


def _cpu_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    if system == "Windows":
        return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def _backend_entries() -> tuple[list[dict[str, Any]], dict[str, torch.device]]:
    entries: list[dict[str, Any]] = []
    devices: dict[str, torch.device] = {}
    entries.append(
        {
            "kind": "cpu",
            "name": _cpu_name() or "CPU",
            "available": True,
            "memory_mb": int(psutil.virtual_memory().total // (1024 * 1024)),
            "reason": "",
        }
    )
    devices["cpu"] = torch.device("cpu")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            hip = bool(getattr(torch.version, "hip", None))
            kind = "rocm" if hip else "cuda"
            entries.append(
                {
                    "kind": kind,
                    "name": f"{props.name} (cuda:{i})",
                    "available": True,
                    "memory_mb": int(props.total_memory // (1024 * 1024)),
                    "reason": "",
                }
            )
            devices[f"{kind}:{i}"] = torch.device(f"cuda:{i}")
    else:
        reason = "CUDA runtime unavailable"
        entries.append({"kind": "cuda", "name": "NVIDIA CUDA", "available": False, "memory_mb": 0, "reason": reason})
        entries.append({"kind": "rocm", "name": "AMD ROCm", "available": False, "memory_mb": 0, "reason": "ROCm PyTorch build unavailable"})
    mps = getattr(torch.backends, "mps", None)
    mps_ok = bool(mps is not None and mps.is_available())
    entries.append(
        {
            "kind": "mps",
            "name": "Apple Metal (MPS)" if mps_ok else "Apple Metal (MPS)",
            "available": mps_ok,
            "memory_mb": int(psutil.virtual_memory().total // (1024 * 1024)) if mps_ok else 0,
            "reason": "" if mps_ok else "MPS unavailable or unsupported OS",
        }
    )
    if mps_ok:
        devices["mps"] = torch.device("mps")
    xpu = getattr(torch, "xpu", None)
    xpu_ok = False
    try:
        xpu_ok = bool(xpu is not None and xpu.is_available())
    except Exception:
        xpu_ok = False
    entries.append(
        {
            "kind": "xpu",
            "name": "Intel XPU",
            "available": xpu_ok,
            "memory_mb": 0,
            "reason": "" if xpu_ok else "Intel XPU PyTorch backend unavailable",
        }
    )
    if xpu_ok:
        devices["xpu"] = torch.device("xpu")
    # MLX is reported for transparency, but XDC's training/inference operators use
    # PyTorch.  Advertising an unused backend would violate the no-fake-support rule.
    mlx = False
    try:
        import importlib.util

        mlx = importlib.util.find_spec("mlx") is not None
    except Exception:
        pass
    entries.append(
        {
            "kind": "mlx",
            "name": "Apple MLX",
            "available": mlx,
            "memory_mb": 0,
            "reason": "detected; not used by XDC execution engine" if mlx else "not installed",
        }
    )
    entries.append(
        {
            "kind": "npu",
            "name": "NPU",
            "available": False,
            "memory_mb": 0,
            "reason": "no vendor-neutral NPU backend is exposed by PyTorch on this host",
        }
    )
    return entries, devices


def _pick(preferred: str, entries: list[dict[str, Any]], devices: dict[str, torch.device]) -> str:
    pref = (preferred or "auto").lower()
    if pref == "auto":
        if "cuda:0" in devices:
            return "cuda:0"
        if "rocm:0" in devices:
            return "rocm:0"
        if "mps" in devices:
            return "mps"
        if "xpu" in devices:
            return "xpu"
        return "cpu"
    if pref == "cuda":
        pref = "cuda:0"
    if pref in devices:
        return pref
    if pref.startswith("cuda") or pref.startswith("rocm") or pref in {"mps", "xpu"}:
        raise RuntimeError(f"requested backend {pref} is not available on this host")
    if pref == "cpu":
        return "cpu"
    raise ValueError(f"unknown backend {pref!r}")


def _benchmark(device: torch.device) -> tuple[float, str | None]:
    """Small matmul benchmark, used only for relative scheduling weights."""
    try:
        torch.manual_seed(0)
        a = torch.randn(512, 512, device=device)
        b = torch.randn(512, 512, device=device)
        for _ in range(2):
            a @ b
        _sync(device)
        start = time.perf_counter()
        n = 8
        for _ in range(n):
            a @ b
        _sync(device)
        elapsed = time.perf_counter() - start
        gflops = (2 * 512**3 * n) / max(elapsed, 1e-9) / 1e9
        return round(gflops, 2), None
    except Exception as exc:
        return 1.0, f"microbenchmark failed: {exc}"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "xpu":
        try:
            torch.xpu.synchronize()
        except Exception:
            pass


def detect(preferred: str = "auto", max_memory_mb: int = 0) -> tuple[Detected, Capabilities]:
    entries, devices = _backend_entries()
    selected = _pick(preferred, entries, devices)
    device = devices[selected]
    score, bench_note = _benchmark(device)
    notes = []
    if bench_note:
        notes.append(bench_note)
    if selected == "mps":
        notes.append("Apple unified memory; GPU shares system RAM")
        notes.append("auto policy: CPU below 50M parameters, MPS at or above (configurable threshold)")
    if selected == "cpu":
        notes.append("CPU fallback selected")
    gpu_mem = 0
    device_name = next((e["name"] for e in entries if e["kind"] == selected.split(":")[0]), selected)
    if selected.startswith("cuda") or selected.startswith("rocm") or selected == "xpu":
        try:
            idx = int(selected.split(":")[1]) if ":" in selected else 0
            if selected.startswith("cuda") or selected.startswith("rocm"):
                gpu_mem = int(torch.cuda.get_device_properties(idx).total_memory // (1024 * 1024))
            else:
                gpu_mem = int(torch.xpu.get_device_properties(idx).total_memory // (1024 * 1024))
        except Exception:
            pass
    total_mb = int(psutil.virtual_memory().total // (1024 * 1024))
    avail_mb = int(psutil.virtual_memory().available // (1024 * 1024))
    if max_memory_mb > 0:
        avail_mb = min(avail_mb, max_memory_mb)
    caps = Capabilities(
        platform=platform.platform(),
        machine=platform.machine(),
        hostname=socket.gethostname(),
        python=platform.python_version(),
        torch=torch.__version__,
        backend=selected,
        device_name=device_name,
        cpu_model=_cpu_name(),
        cpu_count=os.cpu_count() or 1,
        total_memory_mb=total_mb,
        available_memory_mb=avail_mb,
        gpu_memory_mb=gpu_mem,
        compute_score=max(score, 0.1),
        backends=entries,
        notes=notes,
    )
    return Detected(selected, device_name, entries, gpu_mem, notes), caps


def node_backend_summary(caps: Capabilities) -> str:
    available = [b["kind"] for b in caps.backends if b.get("available")]
    return f"{caps.device_name} [{caps.backend}] score={caps.compute_score:.1f}GF/s backends={','.join(available)}"
