"""
科里奥利项计算方式的复杂度基准 (P3.4) —— 诚实版。

对比三种算 Ḣ(q,q̇)·q̇ 的方法，都用同一个 H 网络，只改求导方式：
  A. explicit  : 显式构造 ∂H/∂q ∈ R^{n×n×n}（Christoffel 符号路线），再收缩
  B. reverse   : 对 (H·q̇) 的每个分量各做一次 reverse-mode，共 n 次
  C. jvp       : 前向模式一次方向导数（本文用法）

报告墙钟时间与峰值显存随 n 的变化。结论按实测写，不预设倍数。

    python paper/benchmark_coriolis.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper import core
from paper.core import save_json, provenance
from paper.models import EnhancedMLP

ROOT = core.ROOT
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HNet(torch.nn.Module):
    def __init__(self, n, hidden=(1024, 1024, 512, 256)):
        super().__init__()
        self.n = n
        self.net = EnhancedMLP(n, list(hidden), n * n, dropout_rate=0.0)

    def H(self, q):
        M = self.net(q).reshape(q.shape[0], self.n, self.n)
        return torch.bmm(M, M.transpose(1, 2)) + 1e-6 * torch.eye(
            self.n, device=q.device).expand(q.shape[0], self.n, self.n)


def method_jvp(model, q, dq):
    def f(qq):
        return torch.bmm(model.H(qq), dq.unsqueeze(2)).squeeze(2)
    _, out = torch.func.jvp(f, (q,), (dq,))
    return out


def method_reverse(model, q, dq):
    q = q.clone().requires_grad_(True)
    Hdq = torch.bmm(model.H(q), dq.unsqueeze(2)).squeeze(2)
    out = torch.zeros_like(Hdq)
    for j in range(model.n):
        g = torch.autograd.grad(Hdq[:, j].sum(), q, retain_graph=True, create_graph=True)[0]
        out[:, j] = (g * dq).sum(1)
    return out


def method_explicit(model, q, dq):
    """显式构造 ∂H/∂q（n³ 个元素）—— Christoffel 符号的常规做法。"""
    q = q.clone().requires_grad_(True)
    H = model.H(q)
    n = model.n
    dH = torch.zeros(q.shape[0], n, n, n, device=q.device)
    for j in range(n):
        for k in range(j, n):
            g = torch.autograd.grad(H[:, j, k].sum(), q, retain_graph=True, create_graph=True)[0]
            dH[:, j, k, :] = g
            if k != j:
                dH[:, k, j, :] = g
    return torch.einsum('bjki,bi,bk->bj', dH, dq, dq)


METHODS = {'explicit': method_explicit, 'reverse': method_reverse, 'jvp': method_jvp}


def bench(fn, model, q, dq, n_warm=3, n_rep=10):
    for _ in range(n_warm):
        fn(model, q, dq)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        out = fn(model, q, dq)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_rep
    mem = torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == 'cuda' else float('nan')
    return dt, mem, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dims', type=int, nargs='*', default=[7, 12, 20, 30, 50, 80])
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--out', default='experimental_results/v2_tables/coriolis_benchmark.json')
    a = ap.parse_args()

    rows = []
    for n in a.dims:
        torch.manual_seed(0)
        model = HNet(n).to(DEVICE)
        q = torch.randn(a.batch, n, device=DEVICE)
        dq = torch.randn(a.batch, n, device=DEVICE)
        ref = None
        entry = {'n': n, 'batch': a.batch,
                 'n_params': sum(p.numel() for p in model.parameters())}
        for name, fn in METHODS.items():
            try:
                dt, mem, out = bench(fn, model, q, dq, n_rep=a.reps)
                entry[f'{name}_ms'] = dt * 1e3
                entry[f'{name}_peak_MiB'] = mem
                if ref is None:
                    ref = out.detach()
                else:
                    entry[f'{name}_max_abs_diff_vs_explicit'] = float(
                        (out.detach() - ref).abs().max())
            except torch.OutOfMemoryError:
                entry[f'{name}_ms'] = float('nan')
                entry[f'{name}_peak_MiB'] = float('nan')
                entry[f'{name}_oom'] = True
            torch.cuda.empty_cache()
        for m in ['reverse', 'jvp']:
            if np.isfinite(entry.get(f'{m}_ms', float('nan'))):
                entry[f'speedup_explicit_over_{m}'] = entry['explicit_ms'] / entry[f'{m}_ms']
        rows.append(entry)
        print(f"n={n:3d}  explicit={entry['explicit_ms']:8.2f} ms  "
              f"reverse={entry['reverse_ms']:8.2f} ms  jvp={entry['jvp_ms']:8.2f} ms  "
              f"| jvp 相对 explicit 加速 {entry.get('speedup_explicit_over_jvp', float('nan')):.2f}x  "
              f"| 峰值显存 explicit={entry['explicit_peak_MiB']:.0f} MiB, "
              f"jvp={entry['jvp_peak_MiB']:.0f} MiB", flush=True)

    save_json(ROOT / a.out, {'rows': rows, 'device': str(DEVICE),
                             'note': '三种方法用同一个 H 网络，只改求导方式；'
                                     '数值一致性以 explicit 为基准报告最大绝对偏差',
                             'provenance': provenance()})
    print(f'\n结果写入 {a.out}')


if __name__ == '__main__':
    main()
