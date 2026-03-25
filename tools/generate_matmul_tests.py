#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generate matmul test cases for the device-codegen-only POC.

Generates func.func MLIR inputs suitable for iree-device-codegen,
covering static and dynamic shapes with various sizes.

Usage:
  python3 tools/generate_matmul_tests.py --output-dir artifacts/tests
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass
class TestCase:
    name: str
    M: int
    N: int
    K: int
    batch: Optional[int] = None  # None = 2D matmul, int = batched
    bias: bool = False
    # Which dims are dynamic. Subset of {"B", "M", "N", "K"}.
    # "B" only valid when batch is set.
    dynamic_dims: Set[str] = field(default_factory=set)

    @property
    def is_batched(self):
        return self.batch is not None

    @property
    def dynamic_batch(self):
        return "B" in self.dynamic_dims

    @property
    def is_fully_dynamic(self):
        if self.is_batched:
            return self.dynamic_dims == {"B", "M", "N", "K"}
        return self.dynamic_dims == {"M", "N", "K"}


def _d(val, dynamic):
    """Return '?' if dynamic, else str(val)."""
    return "?" if dynamic else str(val)


def generate_matmul_mlir(tc: TestCase) -> str:
    """Generate a func.func for a matmul (optionally batched + bias)."""
    dyn = tc.dynamic_dims
    lines = []

    if tc.is_batched:
        B = _d(tc.batch, "B" in dyn)
        M = _d(tc.M, "M" in dyn)
        K = _d(tc.K, "K" in dyn)
        N = _d(tc.N, "N" in dyn)
        lhs_shape = f"{B}x{M}x{K}"
        rhs_shape = f"{B}x{K}x{N}"
        out_shape = f"{B}x{M}x{N}"
        bias_shape = f"{B}x{M}" if tc.bias else None
        matmul_maps = [
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>",
            "affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>",
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>",
        ]
        matmul_iters = '"parallel", "parallel", "parallel", "reduction"'
        bias_maps = [
            "affine_map<(d0, d1, d2) -> (d0, d1, d2)>",
            "affine_map<(d0, d1, d2) -> (d0, d1)>",
            "affine_map<(d0, d1, d2) -> (d0, d1, d2)>",
        ]
        bias_iters = '"parallel", "parallel", "parallel"'
    else:
        M = _d(tc.M, "M" in dyn)
        K = _d(tc.K, "K" in dyn)
        N = _d(tc.N, "N" in dyn)
        lhs_shape = f"{M}x{K}"
        rhs_shape = f"{K}x{N}"
        out_shape = f"{M}x{N}"
        bias_shape = f"{M}" if tc.bias else None
        matmul_maps = [
            "affine_map<(d0, d1, d2) -> (d0, d2)>",
            "affine_map<(d0, d1, d2) -> (d2, d1)>",
            "affine_map<(d0, d1, d2) -> (d0, d1)>",
        ]
        matmul_iters = '"parallel", "parallel", "reduction"'
        bias_maps = [
            "affine_map<(d0, d1) -> (d0, d1)>",
            "affine_map<(d0, d1) -> (d0)>",
            "affine_map<(d0, d1) -> (d0, d1)>",
        ]
        bias_iters = '"parallel", "parallel"'

    # Count dynamic dims per tensor to know how many index args to emit.
    def _count_dyn(shape_str):
        return shape_str.count("?")

    lhs_ndyn = _count_dyn(lhs_shape)
    rhs_ndyn = _count_dyn(rhs_shape)
    bias_ndyn = _count_dyn(bias_shape) if bias_shape else 0
    out_ndyn = _count_dyn(out_shape)

    # Build func signature: tensor, [dim indices...], tensor, [dim indices...], ...
    args = []
    dim_idx = 0

    args.append(f"%lhs : tensor<{lhs_shape}xf32>")
    for i in range(lhs_ndyn):
        args.append(f"%lhs_d{i} : index")
        dim_idx += 1

    args.append(f"%rhs : tensor<{rhs_shape}xf32>")
    for i in range(rhs_ndyn):
        args.append(f"%rhs_d{i} : index")
        dim_idx += 1

    if tc.bias:
        args.append(f"%bias : tensor<{bias_shape}xf32>")
        for i in range(bias_ndyn):
            args.append(f"%bias_d{i} : index")
            dim_idx += 1

    args.append(
        f'%output : tensor<{out_shape}xf32> {{iree.abi.output = 0 : index}}'
    )
    out_dim_names = []
    for i in range(out_ndyn):
        name = f"%out_d{i}"
        args.append(f"{name} : index")
        out_dim_names.append(name)
        dim_idx += 1

    sig = ",\n    ".join(args)
    lines.append(f"func.func @{tc.name}(")
    lines.append(f"    {sig}")
    lines.append(f") -> tensor<{out_shape}xf32> {{")
    lines.append(f"  %cst = arith.constant 0.000000e+00 : f32")

    # tensor.empty — pass dynamic dim SSA values.
    if out_ndyn > 0:
        dim_args = ", ".join(out_dim_names)
        lines.append(
            f"  %empty = tensor.empty({dim_args}) : tensor<{out_shape}xf32>"
        )
    else:
        lines.append(f"  %empty = tensor.empty() : tensor<{out_shape}xf32>")

    lines.append(
        f"  %fill = linalg.fill ins(%cst : f32)"
        f" outs(%empty : tensor<{out_shape}xf32>) -> tensor<{out_shape}xf32>"
    )

    # Matmul.
    maps_str = ", ".join(matmul_maps)
    lines.append(f"  %matmul = linalg.generic {{")
    lines.append(f"      indexing_maps = [{maps_str}],")
    lines.append(f"      iterator_types = [{matmul_iters}]}}")
    lines.append(
        f"      ins(%lhs, %rhs : tensor<{lhs_shape}xf32>, tensor<{rhs_shape}xf32>)"
    )
    lines.append(f"      outs(%fill : tensor<{out_shape}xf32>) {{")
    lines.append(f"    ^bb0(%a : f32, %b : f32, %c : f32):")
    lines.append(f"      %t0 = arith.mulf %a, %b : f32")
    lines.append(f"      %t1 = arith.addf %t0, %c : f32")
    lines.append(f"      linalg.yield %t1 : f32")
    lines.append(f"    }} -> tensor<{out_shape}xf32>")

    result = "%matmul"

    if tc.bias:
        bias_maps_str = ", ".join(bias_maps)
        lines.append(f"  %biased = linalg.generic {{")
        lines.append(f"      indexing_maps = [{bias_maps_str}],")
        lines.append(f"      iterator_types = [{bias_iters}]}}")
        lines.append(
            f"      ins(%matmul, %bias : tensor<{out_shape}xf32>, tensor<{bias_shape}xf32>)"
        )
        lines.append(f"      outs(%empty : tensor<{out_shape}xf32>) {{")
        lines.append(f"    ^bb0(%a : f32, %b : f32, %c : f32):")
        lines.append(f"      %t0 = arith.addf %a, %b : f32")
        lines.append(f"      linalg.yield %t0 : f32")
        lines.append(f"    }} -> tensor<{out_shape}xf32>")
        result = "%biased"

    lines.append(f"  return {result} : tensor<{out_shape}xf32>")
    lines.append(f"}}")
    return "\n".join(lines)


def generate_metadata(tc: TestCase) -> dict:
    """Generate metadata for the test runner."""
    meta = {"name": tc.name, "M": tc.M, "N": tc.N, "K": tc.K}
    if tc.is_batched:
        meta["batch"] = tc.batch
    meta["bias"] = tc.bias
    meta["dynamic_dims"] = sorted(tc.dynamic_dims)

    # Build the ordered list of push-constant values the runner must pass.
    # Convention: one index arg per dynamic dim per tensor, in argument order
    # (lhs dims, rhs dims, [bias dims,] output dims).
    # Each dim arg carries the concrete value of the corresponding "?" dim.
    pc_values = []
    dim_map = {"B": tc.batch or 0, "M": tc.M, "N": tc.N, "K": tc.K}

    if tc.is_batched:
        # lhs: BxMxK — dynamic dims in order
        for d in ["B", "M", "K"]:
            if d in tc.dynamic_dims:
                pc_values.append(dim_map[d])
        # rhs: BxKxN
        for d in ["B", "K", "N"]:
            if d in tc.dynamic_dims:
                pc_values.append(dim_map[d])
        # bias: BxM
        if tc.bias:
            for d in ["B", "M"]:
                if d in tc.dynamic_dims:
                    pc_values.append(dim_map[d])
        # output: BxMxN
        for d in ["B", "M", "N"]:
            if d in tc.dynamic_dims:
                pc_values.append(dim_map[d])
    else:
        # lhs: MxK
        for d in ["M", "K"]:
            if d in tc.dynamic_dims:
                pc_values.append(dim_map[d])
        # rhs: KxN
        for d in ["K", "N"]:
            if d in tc.dynamic_dims:
                pc_values.append(dim_map[d])
        # bias: M
        if tc.bias:
            for d in ["M"]:
                if d in tc.dynamic_dims:
                    pc_values.append(dim_map[d])
        # output: MxN
        for d in ["M", "N"]:
            if d in tc.dynamic_dims:
                pc_values.append(dim_map[d])

    meta["push_constant_values"] = pc_values
    meta["num_push_constants"] = len(pc_values)

    num_bindings = 3 if not tc.bias else 4
    meta["num_bindings"] = num_bindings
    return meta


# Batch sizes used to iterate dynamic-batch tests.
_POW2_BATCHES = [1, 2, 4, 8]
_NON_POW2_BATCHES = [3, 5, 7]


def _batched_variants(m, n, k, bias, static_batch=4):
    """Generate static + dynamic-batch variants for one (M,N,K,bias) shape.

    Produces:
      - One static test with batch=static_batch
      - One dynamic-batch test per batch in pow2 + non-pow2 lists
    """
    tag = "bias_" if bias else ""
    shape = f"{m}x{n}x{k}"
    cases = []

    # Static
    cases.append(TestCase(
        name=f"bmm_{tag}{static_batch}x{shape}",
        M=m, N=n, K=k, batch=static_batch, bias=bias,
    ))

    # Dynamic batch — power-of-two
    for b in _POW2_BATCHES:
        cases.append(TestCase(
            name=f"bmm_{tag}dyn_{b}x{shape}",
            M=m, N=n, K=k, batch=b, bias=bias, dynamic_dims={"B"},
        ))

    # Dynamic batch — non-power-of-two
    for b in _NON_POW2_BATCHES:
        cases.append(TestCase(
            name=f"bmm_{tag}dyn_{b}x{shape}",
            M=m, N=n, K=k, batch=b, bias=bias, dynamic_dims={"B"},
        ))

    return cases


def _fully_dynamic_variants(m, n, k, bias, batches):
    """Generate fully-dynamic tests (all dims are ?) for given batch sizes."""
    tag = "bias_" if bias else ""
    shape = f"{m}x{n}x{k}"
    cases = []
    for b in batches:
        cases.append(TestCase(
            name=f"bmm_{tag}alldyn_{b}x{shape}",
            M=m, N=n, K=k, batch=b, bias=bias,
            dynamic_dims={"B", "M", "N", "K"},
        ))
    return cases


# Test cases covering a range of scenarios.
#
# Organisation: 2D matmul first (no dynamic axis), then batched shapes
# where every (M,N,K,bias) combination gets static, dynamic-batch, and
# fully-dynamic variants so the shape coverage is identical.
TEST_CASES = [
    # =========================================================================
    # 2D matmul (unbatched — no batch axis to make dynamic)
    # =========================================================================
    # Small, power-of-two
    TestCase(name="matmul_8x8x8", M=8, N=8, K=8),
    TestCase(name="matmul_16x16x16", M=16, N=16, K=16),
    TestCase(name="matmul_32x32x32", M=32, N=32, K=32),
    # Rectangular
    TestCase(name="matmul_128x256x64", M=128, N=256, K=64),
    TestCase(name="matmul_256x512x128", M=256, N=512, K=128),
    # Large
    TestCase(name="matmul_512x512x512", M=512, N=512, K=512),
    # Bias
    TestCase(name="matmul_bias_128x256x64", M=128, N=256, K=64, bias=True),
    # Unaligned / non-power-of-two
    TestCase(name="matmul_33x65x17", M=33, N=65, K=17),
    TestCase(name="matmul_1x1000x1000", M=1, N=1000, K=1000),
    # =========================================================================
    # Batched 128x512x256 — no bias
    #   static B=4, dynamic-batch B=1..8, fully-dynamic B=4,7
    # =========================================================================
    *_batched_variants(m=128, n=512, k=256, bias=False),
    *_fully_dynamic_variants(m=128, n=512, k=256, bias=False, batches=[4, 7]),
    # =========================================================================
    # Batched 128x512x256 — with bias
    #   static B=4, dynamic-batch B=1..8, fully-dynamic B=4,7
    # =========================================================================
    *_batched_variants(m=128, n=512, k=256, bias=True),
    *_fully_dynamic_variants(m=128, n=512, k=256, bias=True, batches=[4, 7]),
    # =========================================================================
    # Batched 33x65x17 — unaligned inner dims, no bias
    #   static B=4, dynamic-batch B=1..8, fully-dynamic B=4,7
    # =========================================================================
    *_batched_variants(m=33, n=65, k=17, bias=False),
    *_fully_dynamic_variants(m=33, n=65, k=17, bias=False, batches=[4, 7]),
    # =========================================================================
    # Batched 33x65x17 — unaligned inner dims, with bias
    #   static B=4, dynamic-batch B=1..8, fully-dynamic B=4,7
    # =========================================================================
    *_batched_variants(m=33, n=65, k=17, bias=True),
    *_fully_dynamic_variants(m=33, n=65, k=17, bias=True, batches=[4, 7]),
]


def main():
    parser = argparse.ArgumentParser(description="Generate matmul test cases")
    parser.add_argument("--output-dir", default="artifacts/tests")
    parser.add_argument(
        "--list", action="store_true", help="List test names and exit"
    )
    args = parser.parse_args()

    if args.list:
        for tc in TEST_CASES:
            shape = f"{tc.M}x{tc.N}x{tc.K}"
            if tc.is_batched:
                shape = f"{tc.batch}x{shape}"
            flags = []
            if tc.bias:
                flags.append("bias")
            if tc.dynamic_dims:
                flags.append(f"dyn={','.join(sorted(tc.dynamic_dims))}")
            print(f"  {tc.name:45s} {shape:20s} {' '.join(flags)}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = []

    for tc in TEST_CASES:
        mlir = generate_matmul_mlir(tc)
        mlir_path = os.path.join(args.output_dir, f"{tc.name}.mlir")
        with open(mlir_path, "w") as f:
            f.write(mlir + "\n")

        meta = generate_metadata(tc)
        manifest.append(meta)
        print(f"  Generated: {tc.name}")

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerated {len(TEST_CASES)} tests in {args.output_dir}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
