"""What the Mac end runs for a container, and on what: the one check both of its paths go through.

A container is not trusted. It names operators and gives them arguments; without a check the
Mac would run any of PyTorch's (`aten::from_file` reads a file of the Mac's and hands back its
bytes) on anything. So:

- **Operators.** Only those of a list, by exact name and overload: `ops/allowed-<version>.txt`,
  made on a Mac with that PyTorch by gen_ops.py from the rules below (`listed`), and kept in the
  repository so that what it admits can be read. An operator is looked up in PyTorch only after
  it was found in the list.
- **Arguments.** Each must be plain data of the kind the operator's schema takes at that place
  (`check`): a tensor of the connection by its number, a number, a boolean, a list of those, a
  type, a layout, a memory format, a generator by its number, the GPU or the CPU as a device,
  and a string only where the schema has one of a few named options, never a file's name.
- **Signatures.** The places a signature names in an operation's frame (which tensor, which
  real number) must be inside the frame it declares (`check`, with `counts`).
- **Tensors.** No tensor a container can name may reach outside its own memory (`usable`),
  whichever operator made it: PyTorch has operators that do not check (`_reshape_alias`), and
  one that fails half way leaves its argument with the new shape and the old memory
  (`resize_`). The C++ of the Mac end has the same check (executor.cpp, usable).
- **Kernels.** An operation runs with the GPU's kernel: one whose tensors are all on the Mac's
  CPU is refused (`placed`; executor.cpp has the same check), and so is an operator that takes
  its arguments on trust (`UNCHECKED`, and the names that say "unsafe" but for those in `READ`).

session.py calls `check` before it runs an operation asked for whole, and before it defines a
signature, in Python or in C++; executor.cpp is told the list and refuses what is not in it.

What this does not do: it does not make PyTorch's operators safe to give hostile arguments. The
list is made by rules, and the operators named above were read; the others were not read one by
one. An operator of the list that reads where its data tells it (an index beyond a tensor, on
the GPU) does what it does for a program on the Mac, and for a program that set
PYTORCH_ENABLE_MPS_FALLBACK=1 the operators marked `cpu` run the CPU's kernels on its data.
Each program has a process of its own for that reason.
"""

import os
import re

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops")

# The most tensors, results and real numbers an operation may have, and the longest list.
MOST = 4096
INT64 = 1 << 63

# The names of the types a tensor on the Mac's GPU, or lent from the CPU, can have. Not the
# quantized ones, which are not plain memory.
DTYPES = ("bool", "uint8", "int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64", "complex64",
          "complex128")
# The same as the numbers PyTorch's schemas carry (c10::ScalarType), for a type that comes as one.
DTYPE_NUMBERS = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 15)
MEMORY_FORMATS = ("contiguous_format", "preserve_format", "channels_last", "channels_last_3d")
DEVICES = ("mps", "mps:0", "cpu")

# Strings: only for these arguments, which name one of an operator's few ways of working
# (a rounding mode, a reduction) and are words...
WORDS = ("rounding_mode", "approximate", "reduce", "side", "UPLO", "driver", "mode")
WORD = re.compile(r"[A-Za-z0-9_]{1,64}\Z")
# ...or are text that goes into an error's message and nowhere else.
MESSAGES = ("api_name", "assert_msg")

# Operators that are never in the list, whatever their arguments: the families that work on
# what is not a plain tensor of the GPU or the CPU, and those that are not computation.
FAMILIES = re.compile(
    r"sparse|coalesce|to_dense|_csr|_csc|_bsr|_bsc|_coo|spdiags|sspaddmm|_spsolve|"  # sparse layouts
    r"quant|dequantize|int_repr|q_scale|q_zero_point|q_per_channel|int4pack|"  # quantized tensors
    r"mkldnn|cudnn|miopen|nnpack|xnnpack|vulkan|triton|cslt|"  # other backends
    r"nested|jagged|"  # nested tensors
    r"_test_|foobar|"  # PyTorch's own tests
    r"from_file|_print|record_stream|pin_memory"  # files, the Mac's output, streams, pinned memory
)
# The parts of a sparse tensor, by their whole names: `_indices` is one of these, and
# `max_pool2d_with_indices` and `tril_indices`, which a Mac runs on its GPU, are not.
SPARSE_PARTS = re.compile(r"(_nnz|_?indices|_?values|c?crow_indices|c?col_indices|row_indices)(_copy)?\Z")

# Operators that take on trust what they do not check. PyTorch's own callers give them arguments
# that are in order; a container gives them what it likes, and the memory they then read or write
# out of place is this process's own, not the GPU's. Each was read in PyTorch's source
# (aten/src/ATen/native, PyTorch 2.15.0), and none is in the list:
UNCHECKED = {
    # TensorAdvancedIndexing.cpp: "the indices are expected to be in bounds and is not checked", and the list
    # of indices is walked against the tensor's sizes without asking how many there are of each.
    "_unsafe_masked_index": "its indices are not checked against the tensor, nor counted against its dimensions",
    "_unsafe_masked_index_put_accumulate": "its indices are not checked against the tensor, nor counted against its dimensions",
    # SegmentReduce.cpp: offsets are never checked ("TODO: add checks when !unsafe"), lengths only on request,
    # and the kernels read the data where those say. The backward takes both as they come.
    "segment_reduce": "its offsets are not checked against the data",
    "_segment_reduce_backward": "its lengths and offsets are not checked against the data",
    # AdaptiveMaxPooling3d.cpp: `gradInput_p_d[maxp] += ...` for every index of a tensor of indices, on the
    # CPU, which is where a Mac runs it.
    "adaptive_max_pool3d_backward": "its indices are not checked against the gradient it writes",
}
# An operator whose name says "unsafe" is not in the list either, unless it is here: read, and found
# to do nothing but hand its arguments to an operator that checks them. (In PyTorch the word is
# about autograd, which is not told of the view, as often as it is about memory.)
READ = {
    "_unsafe_view": "view_impl, which is view (TensorShape.cpp)",
    "unsafe_split": "split, then a version counter of its own for each piece (TensorShape.cpp)",
    "unsafe_split_with_sizes": "split_with_sizes, then a version counter of its own for each piece (TensorShape.cpp)",
    "_unsafe_index": "the indices' types, then index (TensorAdvancedIndexing.cpp)",
    "_unsafe_index_put": "index_put (TensorAdvancedIndexing.cpp)",
}
# An argument by which an operator is told not to check: never passed on as set. `_index_put_impl_`
# has one, `unsafe`, which autograd sets for the gradient of an index (so programs do send it) and
# which PyTorch reads only on CUDA and its kin (TensorAdvancedIndexing.cpp, PyTorch 2.15.0): on
# a Mac it changes nothing, and here it is always given as False, whatever a container says.
NO_CHECK = ("unsafe",)


def trusting(base):
    """Why an operator of that name (without its overload) is one that takes its arguments on trust, or None."""
    if base in UNCHECKED:
        return f"its implementation takes its arguments on trust: {UNCHECKED[base]}"
    if "unsafe" in base and base not in READ:
        return "its name says it does not check its arguments, and it is not one of those read and found to (policy.py, READ)"
    return None


# Operators that PyTorch makes of other operators before they reach a backend, so that a
# container would never send them, but for these, which native.cpp sends whole: attention, when
# no gradient is wanted, because which of its two ways a Mac takes is the Mac's PyTorch's business
# (native.cpp, attention).
# And x.set_(tensor, offset, size, stride), which PyTorch makes of the operator that takes a storage:
# a storage does not travel, so the container sends this one in its place (client.py, _rebound_to_storage).
WHOLE = ("scaled_dot_product_attention", "set_.source_Tensor_storage_offset")


class Refused(ValueError):
    """What a container asked for is not something the Mac end does."""


def kind(type_):
    """A schema type as (how optional, list or not, what): what an argument of it may be."""
    optional = False
    if type_.kind() == "OptionalType":
        optional, type_ = True, type_.getElementType()
    many = False
    if type_.kind() == "ListType":
        many, type_ = True, type_.getElementType()
        if type_.kind() == "OptionalType":  # Tensor?[]: the indices of x[i, :, j]
            type_ = type_.getElementType()
            if type_.kind() != "TensorType":
                return optional, many, None
            return optional, many, "tensor or none"
    what = {"TensorType": "tensor", "IntType": "int", "FloatType": "float", "BoolType": "bool", "NumberType": "number",
            "StringType": "string", "DeviceObjType": "device", "GeneratorType": "generator"}.get(type_.kind())
    return optional, many, what


def listed(name, schema):
    """Whether an operator may be in the list at all, by its name and schema (gen_ops.py adds: and a Mac runs it).

    Returns None if it may, else why not.
    """
    base = (name[6:] if name.startswith("aten::") else name).partition(".")[0]
    if FAMILIES.search(name) or SPARSE_PARTS.match(base):
        return "its family is left out"
    why = trusting(base)
    if why:
        return why
    for argument in schema.arguments:
        _, _, what = kind(argument.type)
        if what is None:
            return f"argument {argument.name} is a {argument.type}"
        if what == "string" and argument.name not in WORDS and argument.name not in MESSAGES:
            return f"argument {argument.name} is a string"
    for result in schema.returns:
        _, _, what = kind(result.type)
        if what not in ("tensor", "int", "float", "bool", "number"):
            return f"a result is a {result.type}"
    return None


def read(version):
    """The list for a PyTorch version: {"name.overload": whether only the CPU runs it}."""
    path = os.path.join(FOLDER, f"allowed-{version}.txt")
    if not os.path.exists(path):
        raise RuntimeError(f"no list of operators for PyTorch {version} ({path}): run TORCH={version} ./build-mac.sh")
    allowed = {}
    with open(path) as lines:
        for line in lines:
            words = line.split()
            if words and not words[0].startswith("#"):
                allowed[words[0]] = len(words) > 1 and words[1] == "cpu"
    return allowed


class Policy:
    def __init__(self, version=None, cpu_fallback=None):
        import torch

        self.torch = torch
        self.allowed = read(version or torch.__version__.split("+")[0])
        # An operator a Mac has no GPU kernel for runs on its CPU only for a program that asked
        # (PYTORCH_ENABLE_MPS_FALLBACK=1, which this process was started with if the program had it).
        self.cpu_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1" if cpu_fallback is None else cpu_fallback
        self.operators = {}

    def names(self):
        """For executor.cpp: the names it may define, as "aten::name.overload"."""
        return {"aten::" + name for name, cpu in self.allowed.items() if self.cpu_fallback or not cpu}

    def operator(self, name, overload):
        """The operator of that name and overload, if it is in the list."""
        found = self.operators.get((name, overload))
        if found is not None:
            return found
        if not isinstance(name, str) or not isinstance(overload, str) or not name.startswith("aten::"):
            raise Refused(f"not an operator the Mac end runs: {name!r}")
        key = name[6:] if overload in ("", "default") else f"{name[6:]}.{overload}"
        # (The list is made with these rules; a list from before them, or one changed by hand, admits nothing they refuse.)
        if key not in self.allowed or trusting(name[6:]):
            raise Refused(f"{name} ({overload or 'default'}) is not in the list of operators the Mac end runs for a container")
        if self.allowed[key] and not self.cpu_fallback:
            raise Refused(f"{name} has no kernel for the Mac's GPU, and this program did not ask for the CPU "
                          "(PYTORCH_ENABLE_MPS_FALLBACK=1)")
        found = getattr(getattr(self.torch.ops.aten, name[6:]), overload or "default")
        self.operators[name, overload] = found
        return found

    def check(self, spec, counts=False):
        """An operation as a container sent it: the operator, if it and its arguments are what the Mac end takes.

        `spec`: "op", "ov", "a" (positional arguments) and "kw". With `counts` it is a signature:
        "nin", "nout", "nf" too, and its arguments name places in a frame.
        """
        if not isinstance(spec, dict):
            raise Refused("an operation is a dictionary")
        operator = self.operator(spec.get("op"), spec.get("ov"))
        schema = operator._schema
        positional, keywords = spec.get("a"), spec.get("kw")
        if not isinstance(positional, list) or not isinstance(keywords, dict):
            raise Refused(f"{schema.name}: arguments are a list and a dictionary")
        frame = None
        if counts:
            frame = {}
            for count in ("nin", "nout", "nf"):
                value = spec.get(count, 0)
                if type(value) is not int or not 0 <= value <= MOST:
                    raise Refused(f"{schema.name}: {count} is a number from 0 to {MOST}")
                frame[count] = value
            results = schema.returns
            whole = len(results) == 1 and kind(results[0].type)[1]  # one list of tensors, of any length
            if not whole and frame["nout"] != len(results):
                raise Refused(f"{schema.name} gives {len(results)} results, not {frame['nout']}")
        parameters = schema.arguments
        if len(positional) > len(parameters):
            raise Refused(f"{schema.name} takes {len(parameters)} arguments, not {len(positional)}")
        by_name = {p.name: i for i, p in enumerate(parameters)}
        given = set(range(len(positional)))
        for i, value in enumerate(positional):
            if parameters[i].kwarg_only:
                raise Refused(f"{schema.name}: {parameters[i].name} is given by name")
            self.argument(schema, parameters[i], value, frame)
        for name, value in keywords.items():
            i = by_name.get(name)
            if i is None:
                raise Refused(f"{schema.name} has no argument {name!r}")
            if i in given:
                raise Refused(f"{schema.name}: {name} is given twice")
            given.add(i)
            self.argument(schema, parameters[i], value, frame)
        for name in NO_CHECK:  # (after the checks above: what is there is a boolean of the operator's)
            i = by_name.get(name)
            if i is not None and i < len(positional):
                positional[i] = False
            elif i is not None and name in keywords:
                keywords[name] = False
        geometry(schema, parameters, positional, keywords, by_name)
        return operator

    def argument(self, schema, parameter, value, frame):
        optional, many, what = kind(parameter.type)

        def no(why):
            return Refused(f"{schema.name}: argument {parameter.name} {why}")

        if value is None:
            if optional or (what == "generator"):  # a generator is always optional in practice: none means the default one
                return
            raise no("is missing")
        if what is None:
            raise no(f"is a {parameter.type}, which a container cannot give")
        if many:
            if frame is not None and what in ("tensor", "tensor or none") and isinstance(value, dict) and set(value) == {"tl"}:
                span = value["tl"]
                if (not isinstance(span, list) or len(span) != 2 or any(type(v) is not int for v in span)
                        or not 0 <= span[0] <= frame["nin"] or not 0 <= span[1] <= frame["nin"] - span[0]):
                    raise no("is a list of tensors outside the frame")
                return
            if not isinstance(value, list) or len(value) > MOST:
                raise no("is a list")
            for element in value:
                if element is None and what == "tensor or none":
                    continue
                self.one(parameter, "tensor" if what == "tensor or none" else what, element, frame, no, in_list=True)
            return
        self.one(parameter, what, value, frame, no)

    def one(self, parameter, what, value, frame, no, in_list=False):
        if isinstance(value, dict):
            keys = set(value)
            real = frame is not None and not in_list and keys in ({"f"}, {"f", "dt"})
            if real and what in ("tensor", "number", "float"):  # a real number sent with each call
                if type(value["f"]) is not int or not 0 <= value["f"] < frame["nf"]:
                    raise no("is a real number outside the frame")
                if "dt" in value and value["dt"] not in (None, "") and (what != "tensor" or value["dt"] not in DTYPES):
                    raise no("is a real number of an unknown type")
                return
            if what == "tensor" and keys == {"t"}:
                number = value["t"]
                if type(number) is not int or not 0 <= number < (INT64 if frame is None else frame["nin"]):
                    raise no("is not a tensor's number" if frame is None else "is a tensor outside the frame")
                return
            if what == "tensor" and keys == {"s", "dt"}:  # a number the container had as a 0-dim tensor
                if type(value["s"]) not in (int, float, bool) or value["dt"] not in DTYPES:
                    raise no("is not a number of a known type")
                return
            if what == "int" and keys == {"dtype"} and parameter.name == "dtype" and value["dtype"] in DTYPES:
                return
            if what == "int" and keys == {"layout"} and parameter.name == "layout" and value["layout"] == "strided":
                return
            if what == "int" and keys == {"mf"} and parameter.name == "memory_format" and value["mf"] in MEMORY_FORMATS:
                return
            if what == "device" and keys == {"device"} and isinstance(value["device"], str) and value["device"] in DEVICES:
                return
            if what == "generator" and keys == {"g"} and type(value["g"]) is int and 0 <= value["g"] < INT64:
                return
            raise no(f"cannot be {value!r:.80}")
        if what == "tensor":  # a Python number where the operator takes a tensor: x + 1.5
            if type(value) in (int, float, bool) and not (type(value) is int and not -INT64 <= value < INT64):
                return
        elif what == "number":
            if type(value) in (int, float, bool) and not (type(value) is int and not -INT64 <= value < INT64):
                return
        elif what == "float":
            if type(value) is float or (type(value) is int and -INT64 <= value < INT64):
                return
        elif what == "bool":
            if type(value) is bool:
                return
        elif what == "int":
            if type(value) is int and -INT64 <= value < INT64:
                if parameter.name == "dtype" and value not in DTYPE_NUMBERS:
                    raise no("is not a type a tensor here can have")
                if parameter.name == "layout" and value != 0:
                    raise no("is not the plain layout")
                if parameter.name == "memory_format" and not 0 <= value < len(MEMORY_FORMATS):
                    raise no("is not a memory format")
                return
        elif what == "string":
            if isinstance(value, str) and (WORD.match(value) if parameter.name in WORDS else
                                           parameter.name in MESSAGES and len(value) <= 1024 and value.isprintable()):
                return
        raise no(f"cannot be {value!r:.80}")

    def placed(self, *values):
        """Refuses an operation whose tensors are all on the Mac's CPU.

        The Mac end computes on the GPU, with the GPU's kernels. A tensor on the CPU is here for one
        operation that takes it next to a tensor of the GPU's (a packed sequence's lengths), and
        PyTorch then runs the GPU's kernel. Given tensors of the CPU's alone it would run the CPU's
        kernel on a container's data, and some of those write where a tensor of indices says without
        looking (a pooling's gradient). An operation with no tensor at all makes one, and reads nothing.
        """
        tensors = gpu = False
        pending = list(values)
        while pending:
            value = pending.pop()
            if isinstance(value, self.torch.Tensor):
                tensors = True
                gpu = gpu or value.is_mps
            elif isinstance(value, (list, tuple)):
                pending.extend(value)
            elif isinstance(value, dict):
                pending.extend(value.values())
        if tensors and not gpu:
            raise Refused("an operation on tensors of the CPU's alone: the Mac end computes on the GPU, and takes a tensor "
                          "of the CPU's only next to one of the GPU's")

    def usable(self, tensor):
        """Whether a tensor is plain memory of the GPU's or the CPU's that it does not reach outside of."""
        torch = self.torch
        if tensor.layout != torch.strided or tensor.is_quantized or tensor.is_nested:
            return False
        device = tensor.device
        if device.type not in ("mps", "cpu") or device.index not in (None, 0):
            return False
        if tensor.numel() == 0:
            return True
        reach = tensor.storage_offset()
        if reach < 0:
            return False
        for size, stride in zip(tensor.shape, tensor.stride()):
            if stride < 0:
                return False
            reach += (size - 1) * stride
        return (reach + 1) * tensor.element_size() <= tensor.untyped_storage().nbytes()


def plain(value):
    """A size or a stride as a list of the numbers given outright, or None (a tensor, something the schema refuses)."""
    if isinstance(value, list) and all(type(v) is int for v in value):
        return value
    return None


def geometry(schema, parameters, positional, keywords, by_name):
    """Sizes, strides and offsets a container gives outright: of a sane sign, and small enough to count with.

    PyTorch checks them too. This is so that what reaches it cannot overflow its own
    arithmetic: the products are taken here, on numbers of any length.
    """

    def given(name):
        i = by_name.get(name)
        if i is None:
            return None
        if i < len(positional):
            return positional[i]
        return keywords.get(name)

    def no(why):
        return Refused(f"{schema.name}: {why}")

    elements = None
    for name in ("size", "shape"):
        sizes = plain(given(name))
        if sizes is None:
            continue
        elements = 1
        for size in sizes:
            if size < -1:  # -1: "work it out", for a view
                raise no(f"a size of {size}")
            elements *= max(size, 1)
        if elements * 16 >= INT64:  # 16: the largest element, a complex128
            raise no(f"a tensor of {elements} elements")
    offset = given("storage_offset")
    if type(offset) is int and offset < 0:
        raise no(f"an offset of {offset}")
    # Strides of memory, not of a convolution: an operator that takes them takes sizes too (as_strided,
    # empty_strided, set_, _reshape_alias).
    if "size" not in by_name or "stride" not in by_name:
        return
    strides, sizes = plain(given("stride")), plain(given("size"))
    if strides is None or sizes is None:
        return
    if len(strides) != len(sizes):
        raise no("as many strides as sizes")
    reach = offset if type(offset) is int else 0
    for size, stride in zip(sizes, strides):
        if stride < 0:
            raise no(f"a stride of {stride}")
        reach += max(size - 1, 0) * stride
    if (reach + 1) * 16 >= INT64:
        raise no(f"a tensor that reaches {reach} elements into its memory")
