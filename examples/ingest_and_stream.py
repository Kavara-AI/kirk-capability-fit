#!/usr/bin/env python3
"""Kirk API Ingestion & Stream — Developer Quickstart.

Reads a raw, local CSV or Parquet time-series file, formats it into Kirk's
expected N-channel × T joint covariance shape (General-Purpose Ingestion), 
and streams the matrices to the authenticated Kirk API endpoint to retrieve the
differential entropy time-series.

This is a general-purpose compiler: it consumes any multi-channel streaming
or historical dataset (sensors, network metrics, logs, or financial micro-data)
and translates it into Kirk's core geometric shape.

Prerequisites:
  pip install pandas numpy requests pyarrow
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from typing import Iterator, List, Tuple

# 1. Configuration — load from environment
KIRK_API_URL = os.environ.get("KIRK_API_URL", "https://kirk-mcp.kavara.ai/v1/infer")

# Support KIRK_API_TOKEN as the clean canonical variable name.
KIRK_API_TOKEN = os.environ.get("KIRK_API_TOKEN")

if not KIRK_API_TOKEN:
    print("error: KIRK_API_TOKEN environment variable is required.", file=sys.stderr)
    print("Please set it to your authorized Kavara API token.", file=sys.stderr)
    sys.exit(1)


def load_and_compile_tensor(
    file_path: Path, 
    N: int = 5, 
    complex_lag: bool = False
) -> tuple[list[list[float]], list[list[float]], list[int]]:
    """Read a general-purpose CSV or Parquet file and compile it into Kirk's expected N x T real/imaginary tensors."""
    if file_path.suffix in (".parquet", ".parq"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
    
    # Clean up and pivot to a wide time-by-variable matrix
    df_wide = df.pivot(index="timestamp", columns="channel_id", values="value")
    
    # Outage safeguard: if forward-filling is used, add tiny random jitter 
    # to prevent zero-change rows from appearing as "perfectly calm" anomalies
    is_na = df_wide.isna()
    df_wide = df_wide.ffill()
    if is_na.any().any():
        rng = np.random.default_rng(42)
        std_devs = df_wide.std()
        for col in df_wide.columns:
            col_std = std_devs[col] or 1.0
            noise = rng.normal(0, col_std * 1e-8, size=df_wide.shape[0])
            df_wide.loc[is_na[col], col] += noise[is_na[col]]
            
    # Apply per-channel transform strategy based on channel properties
    # Default to first_difference for general-purpose safety, pct_change for financial.
    transformed_cols = {}
    for col in df_wide.columns:
        series = df_wide[col]
        # Heuristically classify: if name looks financial, use percentage change or log-return;
        # if it's a general channel or contains zero/negative values, use first difference to prevent inf/NaN.
        is_financial = any(term in col.lower() for term in ("price", "asset", "ticker", "volume", "return"))
        has_zeros_or_negatives = (series <= 0.0).any()
        
        if is_financial and not has_zeros_or_negatives:
            # Financial percentage change
            transformed_cols[col] = series.pct_change()
        else:
            # General-purpose safe first difference (v_t - v_{t-1})
            transformed_cols[col] = series.diff()
            
    df_transformed = pd.DataFrame(transformed_cols, index=df_wide.index).dropna()
    
    # Select the top N channels (sorted alphabetically to ensure a stable, deterministic order across runs)
    active_channels = sorted(list(df_transformed.columns))[:N]
    matrix_data = df_transformed[active_channels].to_numpy(dtype=float).T  # shape (N, T)
    
    # Defensive programming: fail loudly before sending non-finite values (NaN, inf) downstream
    if not np.isfinite(matrix_data).all():
        raise ValueError("non-finite values (NaN or inf) remain in the compiled tensor after transformation")
        
    timestamps = [int(ts.timestamp() * 1e6) for ts in df_transformed.index]
    
    # Kirk expects paired real and imaginary channels (float64 lists)
    matrix_re = matrix_data.tolist()
    if complex_lag:
        # Complex-lagged path: set matrix_im[i][t] = matrix_re[i][t-1] for t >= 1
        matrix_im = []
        for row in matrix_re:
            imag_row = [0.0] + row[:-1]  # shift right by 1 (t-1 lag)
            matrix_im.append(imag_row)
    else:
        matrix_im = [[0.0] * len(matrix_re[0]) for _ in range(N)]  # zero imaginary channel
    
    return matrix_re, matrix_im, timestamps


def sliding_windows(
    matrix_re: List[List[float]],
    matrix_im: List[List[float]],
    timestamps: List[int],
    window_size: int = 10,
    warm_up: str = "expanding",   # "expanding" | "skip"
) -> Iterator[Tuple[List[List[float]], List[List[float]], int, int]]:
    """Yield successive windows over the compiled tensor.

    Parameters
    ----------
    matrix_re, matrix_im : N x T lists
    timestamps           : length-T list of microsecond timestamps
    window_size          : maximum window length (W)
    warm_up              : 
        "expanding" – first W-1 ticks use a growing window (1, 2, …, W-1)
        "skip"      – only emit full windows of size W (simplest)

    Yields
    ------
    (window_re, window_im, timestamp_us, effective_length)
    """
    N = len(matrix_re)
    T = len(matrix_re[0]) if N > 0 else 0

    if T == 0:
        return

    start_t = 0 if warm_up == "expanding" else window_size - 1

    for t in range(start_t, T):
        # Determine the left edge of the window
        left = max(0, t - window_size + 1)
        effective_len = t - left + 1

        window_re = [row[left : t + 1] for row in matrix_re]
        window_im = [row[left : t + 1] for row in matrix_im]

        yield window_re, window_im, timestamps[t], effective_len


class KirkRingBuffer:
    """Fixed-memory sliding window for Kirk inference.

    Memory is strictly O(N * W) regardless of stream length.
    Supports both real and imaginary channels.
    """

    def __init__(self, n_channels: int, window_size: int, complex_lag: bool = False):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")

        self.N = n_channels
        self.W = window_size
        self.complex_lag = complex_lag

        # Pre-allocate circular buffers (N x W)
        self._re = np.zeros((n_channels, window_size), dtype=np.float64)
        self._im = np.zeros((n_channels, window_size), dtype=np.float64)

        self._timestamps = np.zeros(window_size, dtype=np.int64)
        self._pos = 0          # next write index
        self._count = 0        # how many valid samples currently held
        self._last_re = np.zeros(n_channels, dtype=np.float64)  # pre-allocated for complex lag
        self._zero_im = np.zeros(n_channels, dtype=np.float64)  # pre-allocated zero-vector
        self._has_last_re = False

    def __len__(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count == self.W

    def push(
        self,
        sample_re: List[float] | np.ndarray,
        timestamp_us: int,
        sample_im: List[float] | np.ndarray | None = None,
    ) -> None:
        """Ingest one new timestep (length-N vector)."""
        sample_re = np.asarray(sample_re, dtype=np.float64)
        if sample_re.shape != (self.N,):
            raise ValueError(f"expected sample_re of shape ({self.N},), got {sample_re.shape}")

        # Determine imaginary part
        if sample_im is not None:
            sample_im = np.asarray(sample_im, dtype=np.float64)
            if sample_im.shape != (self.N,):
                raise ValueError(f"expected sample_im of shape ({self.N},), got {sample_im.shape}")
        elif self.complex_lag:
            if not self._has_last_re:
                sample_im = self._zero_im
            else:
                sample_im = self._last_re
        else:
            sample_im = self._zero_im

        # Write into ring
        self._re[:, self._pos] = sample_re
        self._im[:, self._pos] = sample_im
        self._timestamps[self._pos] = timestamp_us

        self._last_re[:] = sample_re  # in-place copy, zero allocation!
        self._has_last_re = True
        self._pos = (self._pos + 1) % self.W
        self._count = min(self._count + 1, self.W)

    def get_window(self) -> Tuple[List[List[float]], List[List[float]], int]:
        """Return the current window in chronological order.

        Returns
        -------
        matrix_re : list of lists, shape (N, effective_len)
        matrix_im : list of lists, shape (N, effective_len)
        timestamp_us : timestamp of the most recent sample
        """
        if self._count == 0:
            raise RuntimeError("buffer is empty")

        if self.is_full:
            # Oldest sample is at self._pos
            order = np.arange(self._pos, self._pos + self.W) % self.W
        else:
            # Not yet full — data lives in 0 .. _count-1
            order = np.arange(self._count)

        re_win = self._re[:, order]
        im_win = self._im[:, order]
        ts = int(self._timestamps[order[-1]])

        return re_win.tolist(), im_win.tolist(), ts

    def clear(self) -> None:
        """Reset the buffer to empty state."""
        self._re.fill(0.0)
        self._im.fill(0.0)
        self._timestamps.fill(0)
        self._pos = 0
        self._count = 0
        self._last_re.fill(0.0)
        self._zero_im.fill(0.0)
        self._has_last_re = False


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Kirk Ingest & Stream Quickstart")
    parser.add_argument("--complex-lag", action="store_true", help="Enable complex-lagged path (real=t, imag=t-1)")
    parser.add_argument("--stream", action="store_true", help="Run the real-time sequential streaming simulation (tick-by-tick)")
    parser.add_argument("--warm-up", choices=["expanding", "skip"], default="expanding", help="Warm-up policy: 'expanding' (growing 1..W window) or 'skip' (full W-size only)")
    parser.add_argument("--ring-buffer", action="store_true", help="Use a fixed-memory KirkRingBuffer instead of list-slicing for the stream")
    args = parser.parse_args()

    # Use general-purpose synthetic sample data
    base_dir = Path(__file__).resolve().parent.parent
    sample_csv = base_dir / "examples" / "sample_multichannel_data.csv"
    output_csv = base_dir / "examples" / "kirk_entropy_stream.csv"
    
    if not sample_csv.exists():
        print(f"error: sample data not found at {sample_csv}", file=sys.stderr)
        sys.exit(1)
        
    print(f"1. Reading raw multi-channel data from {sample_csv}...")
    matrix_re, matrix_im, timestamps = load_and_compile_tensor(sample_csv, N=5, complex_lag=args.complex_lag)
    
    if args.complex_lag:
        print("   [INFO] Complex-lagged path enabled: populating imaginary channel with 1-step lag.")

    headers = {
        "Authorization": f"Bearer {KIRK_API_TOKEN}",
        "Content-Type": "application/json"
    }

    if args.stream:
        entropy_stream = []
        out_timestamps = []
        
        if args.ring_buffer:
            print("2. [STREAMING] Initializing sequential fixed-memory KirkRingBuffer streaming...")
            buf = KirkRingBuffer(n_channels=len(matrix_re), window_size=10, complex_lag=args.complex_lag)
            T_total = len(matrix_re[0])
            
            for t in range(T_total):
                # Pull the real and imaginary column vectors at timestep t
                sample_re = [row[t] for row in matrix_re]
                sample_im = [row[t] for row in matrix_im] if not args.complex_lag else None
                
                buf.push(sample_re, timestamp_us=timestamps[t], sample_im=sample_im)
                
                # Retrieve the active sliding window from the ring buffer
                window_re, window_im, ts = buf.get_window()
                
                payload = {
                    "matrix_dim": len(window_re),
                    "matrix_re": window_re,
                    "matrix_im": window_im,
                    "timestamp_us": ts
                }
                try:
                    response = requests.post(KIRK_API_URL, json=payload, headers=headers, timeout=10)
                    response.raise_for_status()
                    result = response.json()
                    entropy_val = result.get("entropy", 0.0)
                    entropy_stream.append(entropy_val)
                    out_timestamps.append(ts)
                    print(f"   [RingBuffer Tick {t + 1}/{T_total}] Timestamp: {ts} | Buffer Size: {len(buf)}/10 | Entropy: {entropy_val:.4f}")
                    time.sleep(0.05)
                except Exception as e:
                    print(f"error: ring buffer stream failed at tick {t}: {e}", file=sys.stderr)
                    sys.exit(1)
        else:
            print(f"2. [STREAMING] Initializing sequential tick-by-tick streaming (warm-up policy: '{args.warm_up}')...")
            T_total = len(matrix_re[0])
            
            for window_re, window_im, ts, eff_len in sliding_windows(
                matrix_re, matrix_im, timestamps,
                window_size=10,
                warm_up=args.warm_up,
            ):
                payload = {
                    "matrix_dim": len(window_re),
                    "matrix_re": window_re,
                    "matrix_im": window_im,
                    "timestamp_us": ts
                }
                try:
                    response = requests.post(KIRK_API_URL, json=payload, headers=headers, timeout=10)
                    response.raise_for_status()
                    result = response.json()
                    # Record the latest single entropy output
                    entropy_val = result.get("entropy", 0.0)
                    entropy_stream.append(entropy_val)
                    out_timestamps.append(ts)
                    print(f"   [Tick {len(entropy_stream)}] Timestamp: {ts} | Ingested: {len(window_re)}x{eff_len} | Entropy: {entropy_val:.4f}")
                    time.sleep(0.05)  # pace the stream
                except Exception as e:
                    print(f"error: stream failed at timestamp {ts}: {e}", file=sys.stderr)
                    sys.exit(1)
    else:
        print(f"2. [BATCH] Streaming compiled {len(matrix_re)}x{len(matrix_re[0])} tensor in one single POST...")
        payload = {
            "matrix_dim": len(matrix_re),
            "matrix_re": matrix_re,
            "matrix_im": matrix_im,
            "timestamp_us": timestamps[-1]
        }
        try:
            response = requests.post(KIRK_API_URL, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            print(f"error: API call failed: {e}", file=sys.stderr)
            sys.exit(1)
            
        entropy_stream = result.get("entropy_stream", [])
        out_timestamps = result.get("timestamps_us", [])
        if not entropy_stream:
            entropy_stream = [result.get("entropy", 0.0)]
            out_timestamps = [timestamps[-1]]
            
        print(f"3. Received {len(entropy_stream)} entropy estimates from Kirk.")

    # Save the output locally for the HMM classifier
    df_out = pd.DataFrame({
        "timestamp_us": out_timestamps,
        "entropy": entropy_stream
    })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"✓ Success: wrote Kirk entropy stream to {output_csv}")


if __name__ == "__main__":
    main()
