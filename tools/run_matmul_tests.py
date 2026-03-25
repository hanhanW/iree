#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run matmul tests through the device-codegen-only pipeline.

For each test case:
1. Generate random input .npy files and compute expected output via NumPy
2. Compile func.func -> HSACO + metadata + workgroup_count.so
3. Launch kernel on GPU via run-hip-kernel --artifacts
4. Verify output against NumPy reference

Usage:
  python3 tools/run_matmul_tests.py \
      --test-dir artifacts/tests --build-dir build --target gfx1100

Compile-only (no GPU needed):
  python3 tools/run_matmul_tests.py \
      --test-dir artifacts/tests --build-dir build --target gfx1100 \
      --compile-only
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np


def run_cmd(cmd, **kwargs):
    """Run a command, return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, **kwargs
    )
    return result.returncode, result.stdout, result.stderr


def generate_inputs(test_meta, work_dir):
    """Generate random input .npy files and compute expected output.

    Returns dict mapping input names to .npy paths, plus expected_output path.
    """
    M = test_meta["M"]
    N = test_meta["N"]
    K = test_meta["K"]
    B = test_meta.get("batch")
    has_bias = test_meta.get("bias", False)

    rng = np.random.default_rng(seed=42)

    paths = {}

    if B is not None:
        lhs = rng.uniform(-0.5, 0.5, (B, M, K)).astype(np.float32)
        rhs = rng.uniform(-0.5, 0.5, (B, K, N)).astype(np.float32)
        output_buf = np.zeros((B, M, N), dtype=np.float32)
        # Batched matmul.
        expected = np.matmul(lhs, rhs)
        if has_bias:
            bias = rng.uniform(-0.5, 0.5, (B, M)).astype(np.float32)
            expected = expected + bias[:, :, np.newaxis]
    else:
        lhs = rng.uniform(-0.5, 0.5, (M, K)).astype(np.float32)
        rhs = rng.uniform(-0.5, 0.5, (K, N)).astype(np.float32)
        output_buf = np.zeros((M, N), dtype=np.float32)
        expected = lhs @ rhs
        if has_bias:
            bias = rng.uniform(-0.5, 0.5, (M,)).astype(np.float32)
            expected = expected + bias[:, np.newaxis]

    # Save inputs.
    lhs_path = os.path.join(work_dir, "lhs.npy")
    rhs_path = os.path.join(work_dir, "rhs.npy")
    np.save(lhs_path, lhs)
    np.save(rhs_path, rhs)
    paths["lhs"] = (lhs.shape, lhs_path)
    paths["rhs"] = (rhs.shape, rhs_path)

    if has_bias:
        bias_path = os.path.join(work_dir, "bias.npy")
        np.save(bias_path, bias)
        paths["bias"] = (bias.shape, bias_path)

    out_path = os.path.join(work_dir, "output.npy")
    np.save(out_path, output_buf)
    paths["output"] = (output_buf.shape, out_path)

    expected_path = os.path.join(work_dir, "expected.npy")
    np.save(expected_path, expected)
    paths["expected"] = expected_path

    return paths


def shape_to_input_str(shape, dtype="f32"):
    """Convert numpy shape to run-hip-kernel input descriptor."""
    return "x".join(str(d) for d in shape) + f"x{dtype}"


def compile_test(mlir_path, build_dir, target, work_dir):
    """Compile a test via iree-device-codegen --output-dir.

    Returns (success, error_msg).
    """
    device_codegen = os.path.join(build_dir, "tools", "iree-device-codegen")

    rc, out, err = run_cmd([
        device_codegen,
        f"--iree-gpu-test-target={target}",
        f"--output-dir={work_dir}",
        mlir_path,
    ])
    if rc != 0:
        return False, f"Codegen failed:\n{err[-500:]}"

    # Verify artifacts exist.
    for f in ["kernel.hsaco", "metadata.json", "workgroup_count.so"]:
        if not os.path.exists(os.path.join(work_dir, f)):
            return False, f"Missing artifact: {f}"

    return True, None


def run_test(test_meta, work_dir, runner_bin, input_paths):
    """Launch kernel on GPU and check correctness."""
    cmd = [
        runner_bin,
        f"--artifacts={work_dir}",
    ]

    # Input buffers in ABI order: lhs, rhs, [bias,] output.
    lhs_shape, lhs_path = input_paths["lhs"]
    rhs_shape, rhs_path = input_paths["rhs"]
    out_shape, out_path = input_paths["output"]

    cmd.append(f"--input={shape_to_input_str(lhs_shape)}=@{lhs_path}")
    cmd.append(f"--input={shape_to_input_str(rhs_shape)}=@{rhs_path}")
    if test_meta.get("bias", False):
        bias_shape, bias_path = input_paths["bias"]
        cmd.append(f"--input={shape_to_input_str(bias_shape)}=@{bias_path}")
    cmd.append(f"--input={shape_to_input_str(out_shape)}=@{out_path}")

    cmd.append(f"--expected_output=@{input_paths['expected']}")

    # Push constants for dynamic dims.
    # Each index argument becomes one i32 push constant in the kernel ABI
    # (loaded via hal.interface.constant.load and cast to index).
    pc_values = test_meta.get("push_constant_values", [])
    if pc_values:
        cmd.append(f"--push-constants={','.join(str(v) for v in pc_values)}")

    rc, out, err = run_cmd(cmd)
    output = out + err
    if "PASS" in output:
        m_err = re.search(r"Max relative error: ([0-9.e+-]+)", output)
        max_err = m_err.group(1) if m_err else "?"
        return 0, f"PASS (max rel err: {max_err})"
    elif rc != 0:
        return 1, f"FAIL: {output[-300:]}"
    else:
        return 1, f"FAIL: no PASS in output"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="artifacts/tests")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--target", default="gfx1100")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile, skip GPU execution",
    )
    parser.add_argument(
        "--filter", default=None, help="Only run tests matching this regex"
    )
    args = parser.parse_args()

    manifest_path = os.path.join(args.test_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(
            f"No manifest.json in {args.test_dir}. "
            f"Run generate_matmul_tests.py first."
        )
        sys.exit(1)

    manifest = json.load(open(manifest_path))

    # Check for run-hip-kernel binary.
    runner_bin = os.path.join(args.build_dir, "tools", "run-hip-kernel")
    has_runner = os.path.exists(runner_bin) and not args.compile_only

    if args.filter:
        pattern = re.compile(args.filter)
        manifest = [t for t in manifest if pattern.search(t["name"])]

    print(f"Running {len(manifest)} tests (target: {args.target})")
    if not has_runner:
        print("  (compile-only mode — no GPU execution)\n")
    else:
        print()

    passed = 0
    failed = 0
    results = []

    for test_meta in manifest:
        name = test_meta["name"]
        mlir_path = os.path.join(args.test_dir, f"{name}.mlir")

        if not os.path.exists(mlir_path):
            print(f"  SKIP  {name:45s} MLIR file not found")
            results.append({"name": name, "status": "SKIP"})
            continue

        with tempfile.TemporaryDirectory() as work_dir:
            # Compile.
            ok, err = compile_test(mlir_path, args.build_dir, args.target,
                                   work_dir)
            if not ok:
                print(f"  FAIL  {name:45s} {err[:80]}")
                failed += 1
                results.append({"name": name, "status": "COMPILE_FAIL",
                                "error": err})
                continue

            if not has_runner:
                print(f"  OK    {name:45s} compiled")
                passed += 1
                results.append({"name": name, "status": "COMPILE_OK"})
                continue

            # Generate inputs and expected output.
            input_paths = generate_inputs(test_meta, work_dir)

            # Run on GPU.
            rc, msg = run_test(test_meta, work_dir, runner_bin,
                               input_paths)
            if rc == 0:
                print(f"  PASS  {name:45s} {msg}")
                passed += 1
                results.append({"name": name, "status": "PASS", "msg": msg})
            else:
                print(f"  FAIL  {name:45s} {msg}")
                failed += 1
                results.append({"name": name, "status": "FAIL", "msg": msg})

    print(f"\n{'=' * 60}")
    print(
        f"Results: {passed} passed, {failed} failed "
        f"/ {len(manifest)} total"
    )
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
