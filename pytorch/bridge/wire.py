"""Framing shared by both ends. Every frame starts with a kind byte and a u32.

    J  u32 header length | header (JSON) | blobs      anything rare: data, questions, answers
    D  u32 signature     | u32 length | JSON          defines an operation's signature, once
    O  u32 signature     | u32 handles... | f64...    one operation: its tensors, then its results, then
                                                      the real numbers that change from call to call
    F  u32 count         | u32 handles...             tensors the container no longer needs

The J header's "b" lists its blobs' lengths. Operations are the hot path: a
signature holds everything that repeats (operator, constants, how many
tensors go in and come out), so one operation is a few integers. Real
numbers that change at every call (an optimizer's step size) are not in the
signature: they follow the integers.

JSON on purpose, not pickle: the Mac end must never run code the container chose.

The first frame on a connection is the container's J {"k": "hello"}: the
version of this protocol and its PyTorch version, which must both be the
Mac's (an operator's meaning, and the list of those a Mac runs whole, change
with the version), and the MPS settings of its environment, which the Mac's
process has to start with.

Every question carries a number, "i", and its answer carries it back: the
container checks that the answer it reads is the one to the question it asked.
"""

import json
import struct

# The version of what is written above. The image holds one end and the Mac's checkout the
# other: they refuse each other when they differ.
#   2: the questions' numbers, the first failure in the answers to copies asked for without
#      waiting, the version itself. (1 had no number: an end that sends none is taken for it.)
#   3: "get" with "storage" (all of the memory a tensor is in, for saving); an operation that is
#      waited for says how many numbers it set aside for its results ("room"), each tensor among
#      them comes back with its own ("n"), and a result can be a list of tensors ("l"); the call
#      "use_deterministic_algorithms".
PROTOCOL = 3

HEAD = struct.Struct("<cI")
U32 = struct.Struct("<I")

# The environment variables that change how MPS behaves: they are read when PyTorch loads, so
# the Mac's process for a program is started with the program's values of them.
ENVIRONMENT = ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO",
               "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL")


def send(w, header, blobs=()):
    """One frame, written in one piece: another thread's operation must not land inside it."""
    h = dict(header)
    if blobs:
        h["b"] = [len(b) for b in blobs]
    j = json.dumps(h, separators=(",", ":")).encode()
    w.write(HEAD.pack(b"J", len(j)) + j, *blobs)


def body(r, length):
    """The rest of a J frame, after its head."""
    h = json.loads(r.read(length))
    blobs = [r.read(k) for k in h.pop("b", [])]
    return h, blobs


def recv(r):
    head = r.read(5)
    if len(head) < 5:
        return None, None
    return body(r, HEAD.unpack(head)[1])


def define(w, signature, spec):
    j = json.dumps(spec, separators=(",", ":")).encode()
    w.write(HEAD.pack(b"D", signature) + U32.pack(len(j)) + j)


def raw(tensor):
    """A CPU tensor's values as bytes, in order, without a copy when it has them in order already.

    Through a view as bytes, not through numpy's own types: numpy has no bfloat16.
    """
    import torch

    tensor = tensor.detach().contiguous()
    if tensor.dim() == 0:
        tensor = tensor.reshape(1)
    if tensor.numel() == 0:
        return b""
    return memoryview(tensor.view(torch.uint8).numpy()).cast("B")
