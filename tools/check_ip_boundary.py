#!/usr/bin/env python3
"""§8 IP-boundary check for the capability-fit artifact.

Enforces the mechanical half of §8. Run in CI so a one-person repo has a real gate
rather than a review requirement nobody can satisfy.

WHAT THIS CATCHES: disclosure that is pattern-detectable — build hashes, infrastructure
identifiers, engine artefact names, attestation measurements.

WHAT IT CANNOT CATCH: over-claim. EN-347 ("the human engineer configures nothing") was a
semantic overstatement, not a pattern, and no regex finds it. That still needs a human
reading the diff. This check narrows what a human must look for; it does not replace them.

NO CUSTOMER-NAME DENYLIST, deliberately: a list of customer names committed to a PUBLIC
repo would itself be the disclosure it is meant to prevent. Customer material is caught
by §8's convention (customers/<name>.md) and by review, not by this file.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CORE = ["SKILL.md"]          # README.md is a symlink to SKILL.md

# Hard fail — no legitimate use in a public capability document.
HARD = [
    (r"\b[0-9a-f]{16,}\b",              "build hash / digest (§8: no build hashes)"),
    (r"\bi-[0-9a-f]{8,}\b",             "EC2 instance id (§8: no infrastructure identifiers)"),
    (r"\bami-[0-9a-f]{6,}\b",           "AMI id (§8: no infrastructure identifiers)"),
    (r"\barn:aws:[a-z0-9\-]+:",         "AWS ARN (§8: no infrastructure identifiers)"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b",    "IPv4 address (§8: no host identifiers)"),
    (r"[a-z0-9\-]+\.amazonaws\.com",    "AWS hostname (§8: no host identifiers)"),
    (r"[a-z0-9\-]+\.ts\.net",           "tailnet hostname (§8: no host identifiers)"),
    (r"\bPCR[0-8]\b",                   "attestation measurement (§8: not a capability fact)"),
    (r"\bkirk_rs_edge\b|\.so\b",        "engine artefact / source layout (§8: no engine source layout)"),
]

# Report only — legitimate uses exist (§3.1 discusses per-channel normalization as a
# PROBLEM property; §8 names the boundary itself). A human decides.
SOFT = [
    (r"\bFrobenius\b|\bSTFT\b|\beigen\w*\b|\bHamiltonian\b",
     "internals vocabulary — is this describing HOW Kirk is built? (§8)"),
    (r"\bzgeev\b|\bzsyev\w*\b|\bLAPACK\b|\bAVX-?\d*\b",
     "kernel/library internals (§8)"),
    (r"normaliz(?:es|ation|ing)\b",
     "check this is a problem property (§3.1) or the boundary rule (§8), not construction"),
]


def scan(path: Path):
    hard, soft = [], []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        for pat, why in HARD:
            for m in re.finditer(pat, line, re.I):
                hard.append((n, m.group(0)[:40], why, stripped[:90]))
        for pat, why in SOFT:
            for m in re.finditer(pat, line, re.I):
                soft.append((n, m.group(0)[:40], why))
    return hard, soft


def main() -> int:
    total_hard = 0
    for name in CORE:
        p = Path(name)
        if not p.exists():
            print(f"::error::{name} missing"); return 2
        hard, soft = scan(p)
        print(f"— {name}: {len(hard)} hard, {len(soft)} advisory")
        for n, tok, why, ctx in hard:
            print(f"::error file={name},line={n}::{why} — found {tok!r}")
            print(f"    {ctx}")
        for n, tok, why in soft:
            print(f"::notice file={name},line={n}::advisory: {why} — {tok!r}")
        total_hard += len(hard)

    if total_hard:
        print(f"\nFAILED: {total_hard} §8 boundary violation(s).")
        print("§8: state the capability and stop. Mechanism belongs in the private record.")
        return 1
    print("\nPASS: no §8 hard violations. Advisories above are for the author, not blocking.")
    print("NOTE: this check finds DISCLOSURE. It cannot find OVER-CLAIM — read the diff.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
