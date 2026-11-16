"""`python -m bridge program.py`: runs a PyTorch program with "mps" on the Mac, and says what the bridge did.

The program needs no such start in the bridge's image, where every Python process gets the bridge
(auto.py): this is for the count of operations at the end, and for an image without the .pth file.
"""

import bridge_native  # before torch: it tells PyTorch there is an MPS device (native.cpp)  # isort: skip

import os
import runpy
import sys

from .auto import quiet

quiet()

import torch  # noqa: E402, F401

from .install import install  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m bridge PROGRAM [ARGS...]")
    bridge = install()
    sys.argv = sys.argv[1:]
    sys.path[0] = os.path.dirname(os.path.abspath(sys.argv[0]))  # as `python PROGRAM` has it
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        try:
            bridge.report()
        finally:
            bridge.close()


main()
