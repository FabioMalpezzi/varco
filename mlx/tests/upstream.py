"""Run mlx's own python tests (mlx/python/tests) and print one line per
module plus the tests that fail or raise. Run it natively and in the
container and compare.

Usage: python upstream.py TESTS_DIR OUTPUT.json [module ...]

The result holds, by module, how many tests ran, the ones skipped and the
ones that failed or raised with the last line of what each said; and under
"info" what mx.metal.is_available() answers on this side. In a container
that is whether the Mac's server takes custom Metal kernels
(--custom-kernels), which changes what MLX's tests run and what is expected
of them: tests/compare.py reads it there.
"""

import json
import os
import sys
import unittest

MODULES = [
    "test_array", "test_ops", "test_reduce", "test_einsum", "test_blas",
    "test_random", "test_nn", "test_losses", "test_fast", "test_fast_sdpa",
    "test_autograd", "test_vmap", "test_eval", "test_compile", "test_conv",
    "test_conv_transpose", "test_upsample", "test_fft", "test_quantized",
    "test_bf16", "test_double", "test_linalg", "test_load", "test_init",
    "test_optimizers", "test_tree", "test_constants", "test_device",
    "test_memory", "test_graph", "test_threads",
]


def main():
    tests_dir, out = sys.argv[1], os.path.abspath(sys.argv[2])
    modules = sys.argv[3:] or MODULES
    os.chdir(tests_dir)
    sys.path.insert(0, tests_dir)
    import mlx.core as mx

    summary = {"info": {"metal": mx.metal.is_available(), "device": str(mx.default_device())}}
    for name in modules:
        try:
            suite = unittest.defaultTestLoader.loadTestsFromName(name)
        except Exception as e:  # noqa: BLE001
            summary[name] = {"ran": 0, "skipped": [], "bad": {"<import>": str(e)[:200]}}
            continue
        result = unittest.TestResult()
        suite.run(result)
        bad = {}
        for test, trace in result.failures + result.errors:
            last = [l for l in trace.strip().splitlines() if l.strip()][-1]
            bad[test.id().split(".", 1)[-1]] = last[:220]
        # The skipped ones by name: a test that one side skips was not checked there.
        skipped = [test.id().split(".", 1)[-1] for test, _ in result.skipped]
        summary[name] = {"ran": result.testsRun, "skipped": skipped, "bad": bad}
        print(f"{name}: ran {result.testsRun}, skipped {len(result.skipped)}, "
              f"failed or raised {len(bad)}", flush=True)
    # Written under a name of its own and renamed. tests/run-all.sh removes the result of the
    # run before on the Mac, and a container of Docker Desktop was twice seen to fail here, a
    # minute of tests later, opening that very name ("No such file or directory"): the name it
    # opens is now one that no earlier run can have left and nothing on the Mac removed.
    partial = f"{out}.{os.getpid()}.partial"
    with open(partial, "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(partial, out)


if __name__ == "__main__":
    main()
