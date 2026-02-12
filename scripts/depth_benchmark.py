"""
Relative Depth Benchmarking Module

Evaluates depth estimation quality using scale- and shift-invariant metrics,
since monocular depth models often predict depth only up to an unknown 
scale and shift.

Both predicted and GT depths are normalised into a common [0, 1] range
before computing error metrics.  Two alignment modes are supported:
    - 'lsq'    : Least-squares affine fit  (pred → s·pred + t ≈ gt),
                  then both mapped to [0, 1].  Best when scales differ a lot.
    - 'minmax' : Independent min-max normalisation to [0, 1].

Rank-based metrics (Spearman, ordinal accuracy) are computed on the raw
values before normalisation since they only depend on ordering.

Metrics:
    - Scale-Invariant Log Error (SILog)
    - Relative Absolute Error (AbsRel)
    - Relative Squared Error (SqRel)
    - RMSE (linear and log)
    - Threshold Accuracy (δ < 1.25, 1.25², 1.25³)
    - Spearman Rank Correlation
    - Ordinal Ranking Accuracy (sampled pixel pairs)

Usage:
    bench = DepthBenchmark(min_depth=1e-3, max_depth=80.0, align='lsq')
    results = bench.evaluate(pred_depth, gt_depth)  
    bench.print_results(results)
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DepthMetrics:
    """Container for depth benchmark results."""
    abs_rel: float = 0.0
    sq_rel: float = 0.0
    rmse: float = 0.0
    rmse_log: float = 0.0
    silog: float = 0.0
    delta_1: float = 0.0       # δ < 1.25
    delta_2: float = 0.0       # δ < 1.25²
    delta_3: float = 0.0       # δ < 1.25³
    spearman: float = 0.0
    ordinal_acc: float = 0.0
    num_valid_pixels: int = 0

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def __repr__(self):
        lines = [
            "═" * 45,
            "  Relative Depth Benchmark Results",
            "═" * 45,
            f"  Valid pixels : {self.num_valid_pixels}",
            "─" * 45,
            f"  AbsRel       : {self.abs_rel:.4f}",
            f"  SqRel        : {self.sq_rel:.4f}",
            f"  RMSE         : {self.rmse:.4f}",
            f"  RMSE (log)   : {self.rmse_log:.4f}",
            f"  SILog        : {self.silog:.4f}",
            "─" * 45,
            f"  δ < 1.25     : {self.delta_1:.4f}",
            f"  δ < 1.25²    : {self.delta_2:.4f}",
            f"  δ < 1.25³    : {self.delta_3:.4f}",
            "─" * 45,
            f"  Spearman ρ   : {self.spearman:.4f}",
            f"  Ordinal Acc  : {self.ordinal_acc:.4f}",
            "═" * 45,
        ]
        return "\n".join(lines)


class DepthBenchmark:
    """
    Benchmark predicted depth against ground truth using relative depth metrics.

    Both predicted and GT depth are normalised into a common [0, 1] range
    before metric computation, so the results are scale- and shift-invariant.

    Two normalisation modes are available (controlled by `align`):
        'lsq'      – Least-squares affine alignment: find (s, t) that minimise
                      ||s * pred + t  -  gt||² over valid pixels, then normalise
                      both aligned-pred and gt to [0, 1].  (default)
        'minmax'   – Simply min-max normalise pred and gt independently to [0, 1].
        False/None – No alignment; only clamp to [min_depth, max_depth].

    Args:
        min_depth: Minimum valid depth value (masks out pixels below this).
        max_depth: Maximum valid depth value (masks out pixels above this).
        align: Normalisation mode – 'lsq', 'minmax', or False/None.
        num_ordinal_pairs: Number of random pixel pairs for ordinal accuracy.
    """

    def __init__(self, min_depth: float = 1e-3, max_depth: float = 80.0,
                 align: str = 'lsq', num_ordinal_pairs: int = 5000):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.align = align
        self.num_ordinal_pairs = num_ordinal_pairs

    # ── normalisation helpers ─────────────────────────────────────────

    @staticmethod
    def _minmax_normalise(t: torch.Tensor) -> torch.Tensor:
        """Normalise tensor to [0, 1] using its min and max."""
        t_min, t_max = t.min(), t.max()
        return (t - t_min) / (t_max - t_min + 1e-8)

    @staticmethod
    def _lsq_align(pred: torch.Tensor, gt: torch.Tensor):
        """
        Least-squares affine alignment: find s, t that minimise
        ||s * pred + t - gt||², then return both on a common [0, 1] scale.
        """
        # Solve  [pred, 1] @ [s, t]^T  =  gt   in least-squares sense
        ones = torch.ones_like(pred)
        A = torch.stack([pred, ones], dim=-1)          # (N, 2)
        b = gt                                          # (N,)
        # Normal equations: (A^T A) x = A^T b
        ATA = A.t() @ A                                # (2, 2)
        ATb = A.t() @ b                                # (2,)
        try:
            params = torch.linalg.solve(ATA, ATb)      # [s, t]
        except Exception:
            # Fallback to median scaling if solve fails
            s = torch.median(gt) / (torch.median(pred) + 1e-8)
            params = torch.tensor([s, 0.0], device=pred.device)

        pred_aligned = pred * params[0] + params[1]

        # Now normalise both to [0, 1] using the GT range
        combined = torch.cat([pred_aligned, gt])
        c_min, c_max = combined.min(), combined.max()
        scale = c_max - c_min + 1e-8

        return (pred_aligned - c_min) / scale, (gt - c_min) / scale

    # ── core entry point ──────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, pred: torch.Tensor, gt: torch.Tensor) -> DepthMetrics:
        """
        Evaluate a single prediction against ground truth.

        Args:
            pred: Predicted depth. Shape (H,W), (1,H,W), (B,H,W) or (B,1,H,W).
            gt:   Ground-truth depth. Same shape flexibility.

        Returns:
            DepthMetrics dataclass with all scores.
        """
        pred, gt = self._prepare(pred, gt)
        valid_mask = self._valid_mask(gt) & self._valid_mask(pred)

        if valid_mask.sum() < 10:
            return DepthMetrics()

        pred_v = pred[valid_mask]
        gt_v = gt[valid_mask]

        # Compute rank-based metrics BEFORE normalisation (order is preserved)
        spearman = self._spearman(pred_v, gt_v)
        ordinal_acc = self._ordinal_accuracy(pred_v, gt_v)

        # Normalise both into a common scale
        if self.align == 'lsq':
            pred_v, gt_v = self._lsq_align(pred_v, gt_v)
        elif self.align == 'minmax':
            pred_v = self._minmax_normalise(pred_v)
            gt_v = self._minmax_normalise(gt_v)

        # Small epsilon floor to avoid log(0) and division by zero
        eps = 1e-6
        pred_v = pred_v.clamp(min=eps)
        gt_v = gt_v.clamp(min=eps)

        metrics = DepthMetrics()
        metrics.num_valid_pixels = int(valid_mask.sum().item())

        # Standard metrics (on normalised depth)
        metrics.abs_rel = self._abs_rel(pred_v, gt_v)
        metrics.sq_rel = self._sq_rel(pred_v, gt_v)
        metrics.rmse = self._rmse(pred_v, gt_v)
        metrics.rmse_log = self._rmse_log(pred_v, gt_v)
        metrics.silog = self._silog(pred_v, gt_v)

        # Threshold accuracies
        metrics.delta_1, metrics.delta_2, metrics.delta_3 = self._deltas(pred_v, gt_v)

        # Rank-based (computed on raw values)
        metrics.spearman = spearman
        metrics.ordinal_acc = ordinal_acc

        return metrics

    @torch.no_grad()
    def evaluate_dataset(self, model, dataloader, device: torch.device,
                         depth_index: int = -1) -> DepthMetrics:
        """
        Evaluate model across an entire dataloader.

        Args:
            model: Model that returns (depth_list, segm_list).
            dataloader: DataLoader yielding (images, depth_gt, labels).
            device: Device to run inference on.
            depth_index: Which scale index to use from model output (default: -1, finest).

        Returns:
            Averaged DepthMetrics across all batches.
        """
        model.eval()
        all_metrics = []

        for images, depth_gt, _ in dataloader:
            images = images.to(device).float()
            depth_gt = depth_gt.to(device).float()

            depth_preds, _ = model(images)
            pred = depth_preds[depth_index]

            # Evaluate per sample in batch
            B = pred.shape[0]
            for b in range(B):
                m = self.evaluate(pred[b], depth_gt[b])
                if m.num_valid_pixels > 0:
                    all_metrics.append(m)

        return self._average_metrics(all_metrics)

    # ── individual metrics ────────────────────────────────────────────

    def _abs_rel(self, pred, gt):
        return (torch.abs(pred - gt) / gt).mean().item()

    def _sq_rel(self, pred, gt):
        return (((pred - gt) ** 2) / gt).mean().item()

    def _rmse(self, pred, gt):
        return torch.sqrt(((pred - gt) ** 2).mean()).item()

    def _rmse_log(self, pred, gt):
        log_diff = torch.log(pred) - torch.log(gt)
        return torch.sqrt((log_diff ** 2).mean()).item()

    def _silog(self, pred, gt):
        """Scale-Invariant Logarithmic Error (Eigen et al.)"""
        log_diff = torch.log(pred) - torch.log(gt)
        silog = torch.sqrt((log_diff ** 2).mean() - log_diff.mean() ** 2) * 100
        return silog.item()

    def _deltas(self, pred, gt):
        """Threshold accuracy: % of pixels where max(pred/gt, gt/pred) < thr."""
        ratio = torch.max(pred / gt, gt / pred)
        d1 = (ratio < 1.25).float().mean().item()
        d2 = (ratio < 1.25 ** 2).float().mean().item()
        d3 = (ratio < 1.25 ** 3).float().mean().item()
        return d1, d2, d3

    def _spearman(self, pred, gt):
        """Spearman rank correlation between flattened pred and gt."""
        pred_np = pred.cpu().numpy().ravel()
        gt_np = gt.cpu().numpy().ravel()

        # Subsample if too large for speed
        n = len(pred_np)
        if n > 50000:
            idx = np.random.choice(n, 50000, replace=False)
            pred_np = pred_np[idx]
            gt_np = gt_np[idx]

        pred_rank = _rankdata(pred_np)
        gt_rank = _rankdata(gt_np)

        d = pred_rank - gt_rank
        n = len(d)
        rho = 1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1) + 1e-8)
        return float(rho)

    def _ordinal_accuracy(self, pred, gt):
        """
        Sample random pixel pairs and check if predicted ordering matches GT.
        A pair is concordant if: (pred_i > pred_j) == (gt_i > gt_j).
        """
        n = pred.numel()
        num_pairs = min(self.num_ordinal_pairs, n * (n - 1) // 2)

        idx_a = torch.randint(0, n, (num_pairs,), device=pred.device)
        idx_b = torch.randint(0, n, (num_pairs,), device=pred.device)

        pred_flat = pred.reshape(-1)
        gt_flat = gt.reshape(-1)

        pred_order = pred_flat[idx_a] > pred_flat[idx_b]
        gt_order = gt_flat[idx_a] > gt_flat[idx_b]

        # Ignore ties in GT
        not_tied = gt_flat[idx_a] != gt_flat[idx_b]
        if not_tied.sum() == 0:
            return 1.0

        concordant = (pred_order == gt_order)[not_tied].float().mean()
        return concordant.item()

    # ── helpers ───────────────────────────────────────────────────────

    def _prepare(self, pred, gt):
        """Squeeze to (H, W) tensors on the same device."""
        pred = pred.detach().float().squeeze()
        gt = gt.detach().float().squeeze()

        # Handle batch dim: if (B,H,W) with B=1, squeeze it out
        if pred.dim() == 3 and pred.shape[0] == 1:
            pred = pred.squeeze(0)
        if gt.dim() == 3 and gt.shape[0] == 1:
            gt = gt.squeeze(0)

        # Resize pred to match gt if needed
        if pred.shape != gt.shape:
            pred = F.interpolate(
                pred.unsqueeze(0).unsqueeze(0),
                size=gt.shape[-2:],
                mode='bilinear',
                align_corners=True
            ).squeeze(0).squeeze(0)

        return pred.to(gt.device), gt

    def _valid_mask(self, t):
        """Mask out invalid depth pixels (zeros, negatives, non-finite)."""
        return (t > self.min_depth) & (t < self.max_depth) & torch.isfinite(t)

    def _average_metrics(self, metrics_list):
        """Average a list of DepthMetrics into a single result."""
        if not metrics_list:
            return DepthMetrics()

        avg = DepthMetrics()
        n = len(metrics_list)
        for m in metrics_list:
            avg.abs_rel += m.abs_rel
            avg.sq_rel += m.sq_rel
            avg.rmse += m.rmse
            avg.rmse_log += m.rmse_log
            avg.silog += m.silog
            avg.delta_1 += m.delta_1
            avg.delta_2 += m.delta_2
            avg.delta_3 += m.delta_3
            avg.spearman += m.spearman
            avg.ordinal_acc += m.ordinal_acc
            avg.num_valid_pixels += m.num_valid_pixels

        for attr in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'silog',
                      'delta_1', 'delta_2', 'delta_3', 'spearman', 'ordinal_acc']:
            setattr(avg, attr, getattr(avg, attr) / n)

        return avg

    @staticmethod
    def print_results(metrics: DepthMetrics):
        """Print the results in a formatted table."""
        print(metrics)


def _rankdata(arr):
    """Simple rank-data (average rank for ties)."""
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    return ranks
