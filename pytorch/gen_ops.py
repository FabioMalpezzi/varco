"""Lists, for this Mac's PyTorch, what the bridge needs to know of its operators. Two files in bridge/ops/:

    .venv-mac-<version>/bin/python gen_ops.py

- mps_ops-<version>.inc: the operators PyTorch has an MPS kernel for, as the include native.cpp
  is built with. The container's backend takes an operator whole only if a Mac would (native.cpp,
  take_operators).
- allowed-<version>.txt: the operators the Mac end runs for a container, by exact name and
  overload (bridge/policy.py reads it; the reasons are there). An operator is in it if a Mac
  can be asked for it on a GPU tensor (it has an MPS kernel, or is made of other operators on
  every backend, or runs on the CPU for a program that set PYTORCH_ENABLE_MPS_FALLBACK=1: those
  are marked `cpu`) and policy.listed lets it in: no family that works on anything but plain
  tensors, no operator that takes its arguments on trust, no argument that is not plain data,
  no string that is not one of a few named options.

Which operators there are, and which a Mac runs on its GPU, changes from one PyTorch to the
next: both lists are made per version, on the Mac, with that version, and kept in the repository.
"""

import collections
import os
import textwrap

import torch

from bridge import policy

version = torch.__version__.split("+")[0]
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bridge", "ops")
os.makedirs(folder, exist_ok=True)
has = torch._C._dispatch_has_kernel_for_dispatch_key
every = sorted(set(full for full in torch._C._dispatch_get_all_op_names() if full.startswith("aten::")))

names = sorted(set(full[6:] for full in every if has(full, "MPS")))
path = os.path.join(folder, f"mps_ops-{version}.inc")
with open(path, "w") as out:
    out.write(f"// Every operator PyTorch {version} for macOS has an MPS kernel for ({len(names)} of them).\n")
    out.write('// Made on a Mac with torch._C._dispatch_has_kernel_for_dispatch_key(name, "MPS"), by gen_ops.py; see native.cpp.\n')
    out.write("\n".join(textwrap.wrap(", ".join(f'"{n}"' for n in names) + ",", 118, break_long_words=False)) + "\n")
print(f"{path}: {len(names)} operators")

# The numbers policy.py takes for types are this PyTorch's.
for number, name in zip(policy.DTYPE_NUMBERS, ("uint8", "int8", "int16", "int32", "int64", "float16", "float32", "float64",
                                                "complex64", "complex128", "bool", "bfloat16")):
    made = torch.ops.aten.empty.memory_format([1], dtype=number).dtype
    if made != getattr(torch, name) or name not in policy.DTYPES:
        raise SystemExit(f"PyTorch {version} numbers its types otherwise ({number} is {made}, not {name}): see policy.py, DTYPE_NUMBERS")

allowed, left_out, trusting = [], collections.Counter(), []
for full in every:
    base, _, overload = full[6:].partition(".")
    if has(full, "MPS") or full[6:] in policy.WHOLE:
        how = ""
    elif has(full, "CompositeImplicitAutograd"):
        left_out["made of other operators before it reaches a backend: a container never sends it"] += 1
        continue
    elif has(full, "CompositeExplicitAutograd") or has(full, "CompositeExplicitAutogradNonFunctional"):
        how = ""
    elif has(full, "CPU"):
        how = " cpu"
    else:
        left_out["no kernel a Mac could run it with"] += 1
        continue
    try:
        schema = getattr(getattr(torch.ops.aten, base), overload or "default")._schema
    except (AttributeError, RuntimeError):
        left_out["not reachable from torch.ops.aten"] += 1
        continue
    why = policy.listed(full, schema)
    if why:
        if "on trust" in why or "does not check" in why:
            trusting.append(full[6:])
            why = "it takes its arguments on trust (policy.py: UNCHECKED, and the names that say unsafe)"
        elif "family" not in why:
            why = "an argument or result that is not plain data"
        left_out[why] += 1
        continue
    allowed.append(full[6:] + how)
path = os.path.join(folder, f"allowed-{version}.txt")
with open(path, "w") as out:
    out.write(f"# The operators the Mac end of the bridge runs for a container, PyTorch {version}: {len(allowed)} of {len(every)}.\n")
    out.write("# One per line, name.overload (no overload: the unnamed one). `cpu`: only for a program that set\n")
    out.write("# PYTORCH_ENABLE_MPS_FALLBACK=1, since a Mac has no GPU kernel for it. Made by gen_ops.py; read by bridge/policy.py.\n")
    out.write("\n".join(allowed) + "\n")
print(f"{path}: {len(allowed)} operators of {len(every)}")
for why, count in left_out.most_common():
    print(f"    {count:5d} left out: {why}")
print("    those that take their arguments on trust: " + " ".join(trusting))
