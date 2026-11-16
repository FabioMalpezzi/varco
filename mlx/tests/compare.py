"""Compare what a stage of run-all.sh gave on the Mac and in the container.

    python compare.py [STAGE] NATIVE.json CONTAINER.json
    python compare.py --inventory STAGE NATIVE.json > expected/STAGE.txt

STAGE is coverage (the default), edge, gen, load, kernels, urgent or upstream.

The exit status is 0 only when both files hold exactly the tests of the
stage's inventory (expected/STAGE.txt: a test that is missing on either side,
or one that is not in the inventory, is a failure, and so is an empty file)
and every test ended as expected. As expected means the same on both sides,
in full: the type, the shape and a digest of all the bytes of every array,
and every value a program read, bit for bit. Nothing is compared with a
tolerance. The tests that are meant to end otherwise are in DECLARED below,
each with what it must do instead and why.

Bit for bit, of a value a program read, is its type and its bits (exact
below): 0.0 is not -0.0, 2 is not 2.0, true is not 1, a number is not the
text of it, and an infinity is the same as the same infinity only. A NaN is
never called the same, on both sides either: the files hold a NaN without
its bits (JSON writes every NaN alike, and so does float.hex), and what a
file does not hold cannot be compared. The arrays hold their NaNs in their
bytes, which the digests cover. A value that is not there (null) is not a
result, where a program cannot have read one.

Needs nothing but the standard library: no MLX and no GPU, so that
scripts/check-comparators.py can run it anywhere.
"""

import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# What is expected to end otherwise than with the same result on both sides.
#   ("raises", TYPE)             an error of that type on both sides
#   ("container raises", TEXT)   an error with that text in the container,
#                                whatever the Mac does
#   ("cpu of each side", WHICH)  the values of those places (counted from 0)
#                                are computed by the CPU of each side, or
#                                from what it computed: arrays of the same
#                                type and shape, whose bytes are not held
#                                to be the same. The other values are.
# A test that is not here must give a result on both sides, the same one. A
# test that is here and no longer does what is written fails too: the entry
# is then to be removed, not left behind.
DECLARED = {
    "coverage": {
        "cpu_linalg": ("cpu of each side", [2],
                       "mx.linalg.inv(a, stream=mx.cpu) runs on the CPU of each side, with the LAPACK each has "
                       "(Accelerate on the Mac, OpenBLAS in the container): the two inverses differ in 13 elements "
                       "of 25, by up to 384 units in the last place, and each is right to 1e-6. The product with "
                       "it that follows is the GPU's, and the same for the same inverse"),
    },
    "edge": {
        "float64_on_gpu": ("raises", "ValueError",
                           "MLX has no float64 on the GPU, on the Mac itself"),
        "huge_allocation": ("raises", "RuntimeError",
                            "more than Metal allocates at once: the case is that the error comes back"),
        "load_on_gpu_stream": ("raises", "RuntimeError",
                               "mlx reads a file on the cpu: a load on a gpu stream that the program names fails "
                               "when it is evaluated"),
        # docs/compatibility.md, "MLX bridge: what works".
        "segmented_mm": ("container raises", "primitive SegmentedMM is not supported by the remote gpu",
                         "mlx's import_function cannot rebuild the primitive on the Mac"),
        "fp8": ("container raises", "primitive ToFP8 is not supported by the remote gpu",
                "mlx's import_function cannot rebuild the primitive on the Mac"),
        "metal_kernel": ("container raises", "Custom Metal kernels are off",
                         "the server of this stage is started without --custom-kernels"),
    },
    "kernels": {
        "threadgroup_too_large": ("raises", "ValueError", "more threads in a group than Metal takes"),
        "syntax_error": ("raises", "RuntimeError", "a kernel that does not compile"),
        "bad_name": ("raises", "RuntimeError", "a kernel whose name is not an identifier"),
        "wrong_number_of_inputs": ("raises", "ValueError", "mlx checks it before the kernel runs"),
        "on_the_cpu": ("raises", "ValueError", "a kernel runs on the GPU only"),
        # What the container refuses on purpose and the Mac itself would run
        # (docs/security.md).
        "include_refused": ("container raises", "must not make the compiler read files",
                            "a kernel must not make the Mac's compiler read files"),
        "pragma_refused": ("container raises", "must not make the compiler read files",
                           "a kernel must not make the Mac's compiler read files"),
        "source_too_large": ("container raises", "the Mac takes at most",
                             "the server takes a source of 256 KB at most"),
        "imported_function": ("container raises", "only those that mx.fast.metal_kernel makes in this process",
                              "the kernel of a function imported from a file is not sent"),
    },
}

# MLX's own tests that fail or raise, by side (docs/compatibility.md, "MLX
# bridge: what works"). Any other failure, and any of these that passes, is
# a difference from what is declared.
UPSTREAM_FAILS = {
    "native": {
        ("test_nn", "TestBase.test_circular_leaks"),
    },
    # Whatever the Mac's server takes.
    "container": {
        ("test_array", "TestArray.test_buffer_protocol"),
        ("test_array", "TestArray.test_memoryless_copy"),
        ("test_blas", "TestBlas.test_gather_mm_sorted_vjp"),
        ("test_blas", "TestBlas.test_segmented_mm"),
        ("test_load", "TestLoad.test_load_f8_e4m3"),
        ("test_memory", "TestMemory.test_array_buffer_size"),
        ("test_nn", "TestBase.test_circular_leaks"),
        ("test_ops", "TestOps.test_to_from_fp8"),
        ("test_quantized", "TestQuantized.test_gather_qmm_grad"),
        ("test_quantized", "TestQuantized.test_gather_qqmm"),
        ("test_quantized", "TestQuantized.test_gather_qqmm_global_scale_matrix_paths"),
        ("test_quantized", "TestQuantized.test_qqmm"),
        ("test_quantized", "TestQuantized.test_qqmv"),
        ("test_threads", "TestThreads.test_threadlocal_stream"),
    },
}

# The container's besides, by whether the Mac's server takes custom Metal kernels: what
# mx.metal.is_available() answers in the container, which tests/upstream.py writes in its result
# ("info"). A server is started without --custom-kernels by default (docs/security.md), and so is
# the stage's;  SERVER_ARGS=--custom-kernels tests/run-all.sh upstream  runs it with them.
# Without them, MLX's five tests of custom kernels that are not marked to be skipped fail:
UPSTREAM_FAILS_WITHOUT_KERNELS = {
    ("test_fast", "TestFast.test_custom_kernel_args"),
    ("test_fast", "TestFast.test_custom_kernel_attributes"),
    ("test_fast", "TestFast.test_custom_kernel_basic"),
    ("test_fast", "TestFast.test_custom_kernel_helper"),
    ("test_fast", "TestFast.test_custom_kernel_strides"),
}
# With them those five pass, the tests that MLX marks to run only where mx.metal.is_available()
# run (UPSTREAM_SKIPS below), and nine fail, each for the reason given. None is a computation
# that comes out otherwise than on the Mac. Checked by bytes for the convolution with
# MLX_CONV_WINOGRAD=0 and the attention with MLX_SDPA_D512_MIN_KL=0: with the variable in the
# program's process alone the container has the bytes the Mac gives without it, and with the
# variable given to the server the bytes the Mac gives with it.
VARIABLE = ("the test chooses the Mac's kernel with {0} set in its own process, and MLX on the Mac reads "
            "the variable in the process that computes, which for a container is the server: the container "
            "is given what the Mac computes with the server's environment")
QQMM = ("mx.qqmm, which is the primitive QQMatmul: not supported by the remote gpu (mlx's import_function cannot "
        "rebuild it on the Mac), as in test_qqmm and test_qqmv, which fail for it whatever the server takes")
UPSTREAM_FAILS_WITH_KERNELS = {
    ("test_array", "TestInequality.test_dlx_device_type"):
        "the test holds that where mx.metal.is_available() an array is in Metal's memory (DLPack device 8). In a "
        "container the first says whether the server takes custom kernels, and an array's bytes are in the "
        "container's own memory (device 1, the CPU): both answers are right there",
    ("test_conv", "TestConv.test_conv2d_winograd_batch_tiling"):
        VARIABLE.format("MLX_CONV_WINOGRAD_WORKING_SET and MLX_CONV_WINOGRAD_TILE_BATCH"),
    ("test_conv", "TestConv.test_conv2d_winograd_disable"):
        VARIABLE.format("MLX_CONV_WINOGRAD=0"),
    ("test_fast_sdpa", "TestFastSDPA.test_sdpa_force_fused_metal"):
        VARIABLE.format("MLX_SDPA_D512_MIN_KL=0") + ". And a call with force_fused that no fused kernel takes is "
        "refused by the Mac, with MLX's own error, when the graph is evaluated: the test expects the error when "
        "the operation is made, without evaluating it",
    ("test_fast_sdpa", "TestFastSDPA.test_sdpa_vector_head_dim_512"):
        VARIABLE.format("MLX_SDPA_D512_MIN_KL=0") + ": started so, the server gives a container that passes this test",
    ("test_memory", "TestMemory.test_memory_info"):
        "where mx.metal.is_available() the test counts on Metal's cache of freed buffers. That cache is in the "
        "server on the Mac, shared by the containers; mx.get_cache_memory() in a container counts the "
        "container's own memory",
    ("test_memory", "TestMemory.test_wired_memory"):
        "mx.set_wired_limit() does nothing in a container and gives back 0: the limit of wired memory is the "
        "Mac's, and not a container's to set",
    ("test_quantized", "TestQuantized.test_fp_qmv_fast_unaligned_n_global_scale"): QQMM,
    ("test_quantized", "TestQuantized.test_qqmm_global_scale_matrix_kernels"): QQMM,
}

# MLX's own tests that may fail, on either side, and what their failure says: such a failure is
# reported and is not a difference; the same test failing with other words is, and so is every
# other failure.
UPSTREAM_MAY_FAIL = {
    ("test_ops", "TestOps.test_divmod"): (
        "Type <class 'numpy.float16'>",
        "the test draws its inputs from numpy's generator without a seed and holds mx.divmod to numpy's "
        "quotients: for some pairs of float16 the quotient rounds up to a whole number, which mx.divmod "
        "gives and numpy's floor is one below (81.1 by 9.016: 9 and 8), on the Mac itself. Measured, the "
        "test alone, repeated in one process: 32 failures in 1000 runs on the Mac and 23 in 1000 in a "
        "container, every one for float16. Of 1,000,000 seeded pairs 437 differ so, the same pairs with "
        "the same bytes on the Mac and in a container"),
}

# MLX's own tests that one side skips and the other runs, by the side that skips. A test that one
# side skips was not checked there: these are the ones known not to be, and why. Any other such
# test, and any of these that runs, is a difference from what is declared.
# The container's: each is marked by MLX to run only where mx.metal.is_available(), which in a
# container says whether the server takes custom Metal kernels, and the server of the stage is
# started without --custom-kernels, as a server is by default (docs/security.md). They test Metal
# kernels of MLX's own that the Mac runs for the container all the same; that they give the Mac's
# result through the bridge is not shown by this stage. Against a server started with
# --custom-kernels the container skips none of them (upstream_declared below).
UPSTREAM_SKIPS = {
    "native": set(),
    "container": {
        ("test_conv", "TestConv.test_conv2d_winograd_batch_tiling"),
        ("test_conv", "TestConv.test_conv2d_winograd_disable"),
        ("test_conv", "TestConv.test_conv_depthwise_c_multiple_of_8"),
        ("test_conv", "TestConv.test_conv_grouped_c_per_group_8"),
        ("test_conv_transpose", "TestConvTranspose.test_conv_transpose_unfold_tiling"),
        ("test_eval", "TestEval.test_eval_exception_after_cross_stream_wait"),
        ("test_eval", "TestEval.test_eval_exception_does_not_corrupt_state"),
        ("test_fast", "TestFast.test_custom_kernel_caching"),
        ("test_fast", "TestFast.test_custom_kernel_mixed_dtypes"),
        ("test_fast", "TestFast.test_custom_kernel_same_name_different_source_one_eval"),
        ("test_fast", "TestFast.test_custom_metal_kernel_math_mode"),
        ("test_fast", "TestFast.test_custom_metal_kernel_negative_template_int"),
        ("test_fast_sdpa", "TestFastSDPA.test_sdpa_force_fused_metal"),
        ("test_fast_sdpa", "TestFastSDPA.test_sdpa_full_d96_v64"),
        ("test_fast_sdpa", "TestFastSDPA.test_sdpa_pad_head_dim_opt_in"),
        ("test_fast_sdpa", "TestFastSDPA.test_sdpa_vector_2pass_threadgroup_limit"),
        ("test_fast_sdpa", "TestFastSDPA.test_sdpa_vector_d96_v64"),
        ("test_fast_sdpa", "TestFastSDPA.test_sdpa_vector_head_dim_512"),
        ("test_fft", "TestFFT.test_batched_bluestein_twiddle_table"),
        ("test_fft", "TestFFT.test_fft_too_large"),
        ("test_memory", "TestMemory.test_wired_memory"),
        ("test_ops", "TestOps.test_hadamard_m_only"),
        ("test_quantized", "TestQuantized.test_fp_qmv_fast_unaligned_n"),
        ("test_quantized", "TestQuantized.test_fp_qmv_fast_unaligned_n_global_scale"),
        ("test_quantized", "TestQuantized.test_gather_qmm_sorted_nax_large_m"),
        ("test_quantized", "TestQuantized.test_qmm_splitk_precision"),
        ("test_quantized", "TestQuantized.test_qmv_affine_bias_sum_precision"),
        ("test_quantized", "TestQuantized.test_qmv_fast_unaligned_n"),
        ("test_quantized", "TestQuantized.test_qqmm_global_scale_matrix_kernels"),
        ("test_reduce", "TestReduce.test_init_reduce_large_output"),
    },
}


def upstream_declared(side, kernels):
    """(the tests declared to fail, the tests declared to be skipped on that side only), on that side.

    `kernels`: what mx.metal.is_available() answers in the container, that is whether the Mac's
    server takes custom Metal kernels. The Mac's own side is the same either way."""
    if side == "native":
        return set(UPSTREAM_FAILS["native"]), set(UPSTREAM_SKIPS["native"])
    if kernels:
        return UPSTREAM_FAILS["container"] | set(UPSTREAM_FAILS_WITH_KERNELS), set()
    return UPSTREAM_FAILS["container"] | UPSTREAM_FAILS_WITHOUT_KERNELS, set(UPSTREAM_SKIPS["container"])


HEADER = """\
# The tests of the stage "{stage}" of mlx/tests/run-all.sh, one name per line.
# tests/compare.py fails when a result lacks one of them or holds another.
# After adding or removing a test, on a Mac, from mlx/:
#   tests/run-all.sh {stage}
#   python3 tests/compare.py --inventory {stage} results/{stage}-native.json > tests/expected/{stage}.txt
# and read the diff before committing it: a name that goes is a test that no
# longer runs. (The first of the two fails until the second is done.)
"""


def is_array(x):
    """The fingerprint of an array: its type, its shape and a digest of its bytes."""
    return (isinstance(x, dict) and isinstance(x.get("dtype"), str) and isinstance(x.get("shape"), list)
            and re.fullmatch(r"[0-9a-f]{64}", str(x.get("sha256"))) is not None)


# A float as float.hex writes it: all the bits of a number, an infinity, or "nan" for every NaN.
HEX = re.compile(r"-?(0x[0-9a-f]+(\.[0-9a-f]*)?p[+-][0-9]+|inf)|nan")


def exact(x):
    """A value of a result file as what is compared: with its type, and a float with all its bits.

    Python's own == is not that: 0.0 == -0.0, 2 == 2.0 and True == 1 all hold."""
    if isinstance(x, bool):
        return ("bool", x)
    if isinstance(x, int):
        return ("int", x)
    if isinstance(x, float):
        return ("float", x.hex())
    if isinstance(x, (list, tuple)):
        return [exact(v) for v in x]
    if isinstance(x, dict):
        return {key: exact(v) for key, v in x.items()}
    return x


def same(a, b):
    return exact(a) == exact(b)


def has(x, what):
    """Whether a value holds, at any depth, one that `what` is true of."""
    if isinstance(x, (list, tuple)):
        return any(has(v, what) for v in x)
    if isinstance(x, dict):
        return what(x) or any(has(v, what) for v in x.values())
    return what(x)


def is_nan(x):
    """A NaN as the files hold one: a number, or a float that coverage.py wrote with float.hex."""
    if isinstance(x, dict):
        x = x.get("value") if x.get("python") == "float" else None
        try:
            x = float.fromhex(x) if isinstance(x, str) else x
        except ValueError:
            return False
    return isinstance(x, float) and math.isnan(x)


def is_nothing(x):
    return x is None


def is_read(v):
    """A value a program read, as coverage.py keeps it: a float in hexadecimal, any other as it is."""
    if not (isinstance(v, dict) and "python" in v and "value" in v):
        return False
    # Text that only reads as a number ("1.5") is not what float.hex writes.
    return v["python"] != "float" or isinstance(v["value"], float) or (isinstance(v["value"], str) and HEX.fullmatch(v["value"]) is not None)


# --- what each stage's file holds: {name: ("ok", value) | ("raised", text) | ("unreadable", what)} ---


def read_coverage(doc):
    out = {}
    for name, r in doc["results"].items():
        if isinstance(r, dict) and isinstance(r.get("raised"), str):
            out[name] = ("raised", r["raised"])
        elif (isinstance(r, dict) and isinstance(r.get("values"), list) and r["values"]
              and all(is_array(v) or is_read(v) for v in r["values"])):
            out[name] = ("ok", r["values"])
        else:
            # A number alone, as coverage.py wrote before it kept fingerprints,
            # says too little: two different arrays can give the same one.
            out[name] = ("unreadable", r)
    return out


def read_edge(doc):
    out = {}
    for name, r in doc["results"].items():
        if isinstance(r, dict) and isinstance(r.get("raised"), str):
            out[name] = ("raised", r["raised"])
        elif isinstance(r, dict) and isinstance(r.get("ok"), str):
            out[name] = ("ok", r["ok"])
        else:
            out[name] = ("unreadable", r)
    return out


def read_gen(doc):
    out = {}
    for i, run in enumerate(doc["runs"]):
        tokens, logprobs = run.get("tokens"), run.get("logprobs")
        whole = isinstance(tokens, list) and tokens and all(type(t) is int for t in tokens)
        out[f"run {i + 1}, the tokens"] = ("ok", tokens) if whole else ("unreadable", tokens)
        out[f"run {i + 1}, the last logprobs"] = ("ok", logprobs) if is_array(logprobs) else ("unreadable", logprobs)
    return out


def read_load(doc):
    out = {}
    for name, r in doc["results"].items():
        for key, value in (r.items() if isinstance(r, dict) else [("", r)]):
            # load_files.py writes no null: one is a value that is not there.
            out[f"{name} {key}".strip()] = ("unreadable" if has(value, is_nothing) else "ok", value)
    return out


def read_kernels(doc):
    out = {}
    for name, r in doc["results"].items():
        if isinstance(r, str):
            out[name] = ("raised", r)
        elif isinstance(r, list) and r and all(is_array(v) or (isinstance(v, dict) and "text" in v) for v in r):
            out[name] = ("ok", r)
        else:
            out[name] = ("unreadable", r)
    return out


def read_urgent(doc):
    # A read gives a number, a truth or a list of them: never nothing.
    return {name: ("unreadable" if has(case["value"], is_nothing) else "ok", case["value"]) for name, case in doc["cases"].items()}


def upstream_modules(doc):
    """What a result of MLX's tests holds by module: all of it but what the side says of itself ("info")."""
    return {module: r for module, r in doc.items() if module != "info"}


def read_upstream(doc):
    """How many tests each module ran: what fails and what is skipped is looked at apart."""
    return {module: ("ok" if type(r["ran"]) is int else "unreadable", r["ran"]) for module, r in upstream_modules(doc).items()}


# (what reads the file, how the summary line goes on)
STAGES = {
    "coverage": (read_coverage, "tests give the same shapes, types and bytes"),
    "edge": (read_edge, "cases give the same result"),
    "gen": (read_gen, "results of the generation loop are the same"),
    "load": (read_load, "loaded arrays are the same"),
    "kernels": (read_kernels, "cases that run give the same bytes"),
    "urgent": (read_urgent, "reads give the same value"),
    "upstream": (read_upstream, "modules ran the same number of tests"),
}


def inventory(stage):
    path = os.path.join(HERE, "expected", stage + ".txt")
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip() and not line.startswith("#")]


def load(path, stage):
    """(the file as it is, what it holds by test name), or None after saying why not."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        return doc, STAGES[stage][0](doc)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"FAILED: {path} is not a result of the stage {stage}: {type(e).__name__}: {e}")
        return None


def short(outcome, limit=230):
    if outcome is None:
        return "nothing"
    kind, value = outcome
    if kind == "raised":
        return value[:limit]
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        return f"ok, {len(value)} values"
    text = value if isinstance(value, str) else json.dumps(value)
    return ("ok " if kind == "ok" else "not a result: ") + text[:limit]


def difference(a, b):
    """Where two results that should be the same part ways."""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{len(a)} values on the Mac, {len(b)} in the container"
        where = [i for i, (x, y) in enumerate(zip(a, b)) if not same(x, y)]
        i = where[0]
        if is_array(a[i]) and is_array(b[i]):
            parts = [f"{what} {a[i][key]} on the Mac and {b[i][key]} in the container"
                     for key, what in (("dtype", "type"), ("shape", "shape")) if a[i][key] != b[i][key]]
            if not parts:
                parts = [f"the bytes differ ({a[i]['dtype']} {a[i]['shape']})"]
            return f"{len(where)} of {len(a)} values differ; value {i}: " + ", ".join(parts)
        return f"{len(where)} of {len(a)} values differ; value {i}: {json.dumps(a[i])[:100]} on the Mac, {json.dumps(b[i])[:100]} in the container"
    if is_array(a) and is_array(b):
        return difference([a], [b]).partition("; value 0: ")[2]
    return ""


def judge(stage, native, other):
    """Print how the two sides compare, test by test of the inventory. True when all is as expected."""
    names = inventory(stage)
    declared = DECLARED.get(stage, {})
    alike, raise_both, refused, by_cpu, wrong = 0, [], [], [], []
    for name in names:
        a, b = native.get(name), other.get(name)
        kind, detail, _ = declared.get(name, ("same", None, None))
        if a is None or b is None:
            sides = [side for side, x in (("on the Mac", a), ("in the container", b)) if x is None]
            wrong.append((name, "MISSING " + " and ".join(sides), a, b))
        elif "unreadable" in (a[0], b[0]):
            wrong.append((name, "NOT A RESULT that can be compared in full", a, b))
        elif kind == "raises":
            if a[0] == b[0] == "raised" and a[1].partition(":")[0] == b[1].partition(":")[0] == detail:
                raise_both.append((name, a, b))
            else:
                wrong.append((name, f"NOT AS DECLARED, a {detail} on both sides", a, b))
        elif kind == "container raises":
            if b[0] == "raised" and detail in b[1]:
                refused.append((name, a, b))
            else:
                wrong.append((name, f"NOT AS DECLARED, an error in the container that says '{detail}'", a, b))
        elif kind == "cpu of each side":
            # The values of the places declared: arrays of one type and shape. Every other value: the same.
            listed = a[0] == b[0] == "ok" and isinstance(a[1], list) and isinstance(b[1], list) and len(a[1]) == len(b[1])
            if listed and all(is_array(x) and is_array(y) and (x["dtype"], x["shape"]) == (y["dtype"], y["shape"]) if i in detail
                              else same(x, y) for i, (x, y) in enumerate(zip(a[1], b[1]))) and all(i < len(a[1]) for i in detail):
                # (Declared so, it may still come out the same: two libraries can agree.)
                if all(same(a[1][i], b[1][i]) for i in detail):
                    alike += 1
                else:
                    by_cpu.append((name, a, b))
            else:
                wrong.append((name, ("NOT AS DECLARED, the same but for the values of each side's CPU. "
                                     + (difference(a[1], b[1]) if listed else "")).rstrip(". "), a, b))
        elif a[0] == b[0] == "raised":
            wrong.append((name, "RAISES on both sides, and is not declared to", a, b))
        elif a[0] == b[0] == "ok" and has(a[1], is_nan) and has(b[1], is_nan):
            wrong.append((name, "A NaN on both sides, which the files hold without its bits: it cannot be called the same", a, b))
        elif same(a, b):
            alike += 1
        else:
            wrong.append((name, ("DIFFERS: " + difference(a[1], b[1])).rstrip(": "), a, b))
    for name in sorted((set(native) | set(other)) - set(names)):
        wrong.append((name, "NOT IN THE INVENTORY (tests/expected/%s.txt)" % stage, native.get(name), other.get(name)))
    print(f"{alike} of {len(names) - len(raise_both) - len(refused) - len(by_cpu)} {STAGES[stage][1]}"
          + (f"; {len(raise_both)} raise on both sides" if raise_both else "")
          + (f"; {len(refused)} the container refuses on purpose" if refused else "")
          + (f"; {len(by_cpu)} the same but for what the CPU of each side computes" if by_cpu else "")
          + (f"; {len(wrong)} NOT AS EXPECTED" if wrong else ""))
    for title, cases in (("raise on both sides", raise_both), ("refused on purpose", refused),
                         ("the same but for the values of each side's CPU", by_cpu)):
        for name, a, b in cases:
            print(f"  {title}: {name}\n    native:    {short(a)}\n    container: {short(b)}")
    for name, why, a, b in wrong:
        print(f"  {why}: {name}\n    native:    {short(a)}\n    container: {short(b)}")
    if stage == "coverage" and any(why.startswith("DIFFERS") for _, why, _, _ in wrong):
        print("  (by how much: run coverage.py on both sides with DUMP=directory and give the two directories to scripts/tensor-diff.py)")
    return not wrong


def urgent_timings(a, b):
    """What the stage is about besides the values: which reads the container answers before the work is done."""
    ok = True
    for key in ("synchronize", "eval"):
        # As the reads above: by type and bits, and a NaN is the same as nothing.
        if not same(a[key]["value"], b[key]["value"]) or has([a[key]["value"], b[key]["value"]], lambda x: is_nan(x) or x is None):
            ok = False
            print(f"  DIFFERENT after {key}(): native {a[key]['value']!r}, container {b[key]['value']!r}")

    def early(c):
        return c["read_ms"] < 0.5 * c["work_ms"]

    cpu = [c for c in b["cases"].values() if c["cpu"]]
    gpu = [c for c in b["cases"].values() if not c["cpu"]]
    if not cpu or not gpu:
        print("  the container's file has no reads of both kinds")
        return False
    print(f"in the container: {sum(map(early, cpu))} of the {len(cpu)} reads that only select, move or compare "
          f"came before the work was half done (median {sorted(c['read_ms'] for c in cpu)[len(cpu) // 2]:.2f} ms "
          f"into {sorted(c['work_ms'] for c in cpu)[len(cpu) // 2]:.0f} ms of work), and "
          f"{sum(map(early, gpu))} of the {len(gpu)} that the GPU computes")
    print(f"natively: {sum(map(early, a['cases'].values()))} of {len(a['cases'])} came before "
          f"(median {sorted(c['read_ms'] for c in a['cases'].values())[len(a['cases']) // 2]:.0f} ms)")
    ok = ok and all(map(early, cpu)) and not any(map(early, gpu))
    for key in ("synchronize", "eval"):
        waited = b[key]["waited_ms"] > 0.8 * b[key]["work_ms"]
        ok = ok and waited
        print(f"in the container {key}() after such a read {'waits' if waited else 'DOES NOT WAIT'} for the work: "
              f"{b[key]['waited_ms']:.0f} of {b[key]['work_ms']:.0f} ms")
    print(f"the first read of the program in the container: after {b['first']['read_ms']:.0f} of "
          f"{b['first']['work_ms']:.0f} ms")
    print("urgent reads: all as expected" if ok else "urgent reads: NOT as expected")
    return ok


def methods(doc, key):
    """The test methods under that key, without what tells one of a method's sub-tests from another."""
    return {(module, re.sub(r" [\(\[].*", "", test, flags=re.S)) for module, r in upstream_modules(doc).items() for test in r[key]}


def upstream_said(doc):
    """What each test method that fails or raises said, sub-test by sub-test: {(module, method): [text]}."""
    said = {}
    for module, r in upstream_modules(doc).items():
        for test, text in r["bad"].items():
            said.setdefault((module, re.sub(r" [\(\[].*", "", test, flags=re.S)), []).append(str(text))
    return said


def upstream_outcomes(a, b):
    """The tests that fail or raise, and the tests one side skips and the other runs, against the declared ones."""
    ok = True
    # Whether the Mac's server takes custom Metal kernels, as the container found it: it decides what is declared.
    kernels = b["info"]["metal"]
    if type(kernels) is not bool:
        raise TypeError(f"the container's result does not say what mx.metal.is_available() answers there: {kernels!r}")
    print(f"the Mac's server takes custom Metal kernels (mx.metal.is_available() in the container): {'yes' if kernels else 'no'}")
    declared = {side: upstream_declared(side, kernels) for side in ("native", "container")}
    for side, doc in (("native", a), ("container", b)):
        said = upstream_said(doc)
        bad = set(said)
        # The ones that may fail: this time they did, with the words of the failure that is known.
        may = sorted(test for test in bad & set(UPSTREAM_MAY_FAIL) if all(UPSTREAM_MAY_FAIL[test][0] in text for text in said[test]))
        fails = declared[side][0]
        more, fewer = sorted(bad - fails - set(may)), sorted(fails - bad)
        modules = upstream_modules(doc).values()
        print(f"{side}: ran {sum(r['ran'] for r in modules)} tests, skipped {sum(len(r['skipped']) for r in modules)}, "
              f"{len(bad)} test methods fail or raise ({len(fails)} declared"
              + (f", and {len(may)} that may fail and did" if may else "") + ")")
        for module, method in may:
            print(f"  fails this time, as it may, {side}: {module} {method}: {UPSTREAM_MAY_FAIL[(module, method)][1]}")
        for module, method in more:
            ok = False
            print(f"  FAILS and is not declared to, {side}: {module} {method}"
                  + (": it may fail, but not with these words: " + said[(module, method)][0][:200] if (module, method) in UPSTREAM_MAY_FAIL else ""))
        for module, method in fewer:
            ok = False
            print(f"  PASSES and is declared to fail, {side}: {module} {method}")
    # A test that one side skips was not checked there, whatever the counts say: only the declared ones may be.
    skipped = {side: methods(doc, "skipped") for side, doc in (("native", a), ("container", b))}
    for side, other in (("container", "native"), ("native", "container")):
        only = skipped[side] - skipped[other]
        skips = declared[side][1]
        more, fewer = sorted(only - skips), sorted(skips - only)
        if only or skips:
            print(f"{side}: skips {len(only)} test methods that the {other} side runs ({len(skips)} declared)")
        for module, method in more:
            ok = False
            print(f"  SKIPPED on the {side} side only and not declared to be: {module} {method}")
        for module, method in fewer:
            ok = False
            print(f"  NOT SKIPPED on the {side} side only, and declared to be: {module} {method}")
    return ok


def main(argv):
    if argv[:1] == ["--inventory"]:
        if len(argv) != 3 or argv[1] not in STAGES:
            print(__doc__, file=sys.stderr)
            return 2
        read = load(argv[2], argv[1])
        if read is None:
            return 1
        sys.stdout.write(HEADER.format(stage=argv[1]))
        for name in read[1]:
            print(name)
        return 0
    stage = argv.pop(0) if argv and argv[0] in STAGES else "coverage"
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    native, other = load(argv[0], stage), load(argv[1], stage)
    if native is None or other is None:
        return 1
    for side, (doc, _) in (("native:   ", native), ("container:", other)):
        if isinstance(doc.get("info"), dict):
            print(side, doc["info"])
    try:
        ok = judge(stage, native[1], other[1])
    except OSError as e:
        print(f"FAILED: the stage {stage} has no inventory to compare with (tests/expected/{stage}.txt): {e}")
        return 1
    try:
        if stage == "urgent":
            ok = urgent_timings(native[0], other[0]) and ok
        elif stage == "upstream":
            ok = upstream_outcomes(native[0], other[0]) and ok
    except (KeyError, TypeError, IndexError) as e:
        print(f"FAILED: a result of the stage {stage} lacks what the stage checks: {type(e).__name__}: {e}")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
