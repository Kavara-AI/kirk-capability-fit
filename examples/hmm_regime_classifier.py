#!/usr/bin/env python3
"""Hidden Markov Model (HMM) Regime Classifier — Developer Quickstart.

Reads the Kirk differential entropy stream produced by `ingest_and_stream.py` and
applies a 3-State Hidden Markov Model (HMM) to classify the system into
Kavara's core structural regimes:
  - State 0: STABLE (low entropy, high correlation stability)
  - State 1: TRANSITION (rising entropy, correlation structures shifting)
  - State 2: VOLATILE (peak/unstable entropy, joint correlation breakdown)

Prerequisites:
  pip install pandas numpy hmmlearn matplotlib
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False


def run_hmm_classification(df: pd.DataFrame) -> np.ndarray:
    """Train a 3-State Gaussian HMM on the entropy time-series and return predicted states."""
    # Format the 1-D entropy column as a column vector (N_samples, 1) for hmmlearn
    X = df["entropy"].to_numpy().reshape(-1, 1)
    
    # Initialize a 3-State Gaussian HMM
    # covariance_type="spherical" is stable for 1-D features
    model = hmm.GaussianHMM(n_components=3, covariance_type="spherical", n_iter=100, random_state=42)
    
    print("Training 3-State Gaussian HMM on Kirk's entropy stream...")
    model.fit(X)
    
    # Predict the hidden state sequence (Viterbi path)
    states = model.predict(X)
    
    # Map hidden states deterministically to our logical regimes (Stable/Transition/Volatile)
    # by sorting the states based on their emission means (Stable = lowest mean, Volatile = highest)
    means = model.means_.flatten()
    sorted_state_indices = np.argsort(means)
    
    state_map = {sorted_state_indices[0]: 0,  # STABLE (lowest mean entropy)
                 sorted_state_indices[1]: 1,  # TRANSITION (medium mean entropy)
                 sorted_state_indices[2]: 2}  # VOLATILE (highest mean entropy)
    
    mapped_states = np.array([state_map[s] for s in states])
    return mapped_states


def run_threshold_fallback_classification(df: pd.DataFrame) -> np.ndarray:
    """Dependency-free threshold-based fallback if `hmmlearn` is not installed."""
    print("hmmlearn not found. Running dependency-free rolling z-score classifier...")
    entropy = df["entropy"].to_numpy()
    
    # Compute rolling z-score over a 20-period window
    window = min(20, len(entropy))
    r_mean = df["entropy"].rolling(window, min_periods=1).mean().to_numpy()
    r_std = df["entropy"].rolling(window, min_periods=1).std().to_numpy()
    r_std[r_std == 0.0] = 1.0  # prevent division by zero
    
    z_scores = (entropy - r_mean) / r_std
    
    # Map z-scores to logical states
    states = np.zeros(len(entropy), dtype=int)
    for i, z in enumerate(z_scores):
        if np.isnan(z) or z < 1.0:
            states[i] = 0  # STABLE
        elif 1.0 <= z < 2.0:
            states[i] = 1  # TRANSITION
        else:
            states[i] = 2  # VOLATILE (outlier)
            
    return states


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    input_csv = base_dir / "examples" / "kirk_entropy_stream.csv"
    
    if not input_csv.exists():
        print(f"error: Kirk entropy stream not found at {input_csv}", file=sys.stderr)
        print("Please run `python3 ingest_and_stream.py` first to generate it.", file=sys.stderr)
        sys.exit(1)
        
    df = pd.read_csv(input_csv)
    if len(df) < 5:
        print("error: entropy stream is too short to train an HMM.", file=sys.stderr)
        sys.exit(1)
        
    if HMMLEARN_AVAILABLE:
        states = run_hmm_classification(df)
    else:
        states = run_threshold_fallback_classification(df)
        
    df["regime_state"] = states
    state_labels = {0: "STABLE", 1: "TRANSITION", 2: "VOLATILE"}
    df["regime_label"] = df["regime_state"].map(state_labels)
    
    # Display the final classified timeline
    print("\n============================================================")
    print("KIRK HMM REGIME CLASSIFICATION TIMELINE")
    print("============================================================")
    for idx, row in df.tail(15).iterrows():
        print(f"Timestamp: {int(row['timestamp_us'])} | Entropy: {row['entropy']:.4f} | State: {row['regime_label']}")
    print("=====================================================")
    
    # Print summary counts
    counts = df["regime_label"].value_counts()
    print("\nRegime Distribution Summary:")
    for label, count in counts.items():
        print(f"  {label}: {count} timesteps ({count/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
