"""Every Python process in the container gets the bridge, with no change to the program.

The image has a .pth file that imports this module when Python starts (Dockerfile). It does
nothing until the program imports torch: then native.cpp's backend is loaded first, as it must be
(it answers PyTorch's first question, whether there is an MPS device), and install.py runs as
soon as PyTorch is ready. So `python train.py` is all it takes, and the processes a program
starts (data-loader workers started afresh, subprocess.run([sys.executable, ...])) get it too.

BRIDGE=0 in the environment turns it off: PyTorch is then the plain CPU build of the image.
"""

import importlib.abc
import importlib.util
import os
import sys


def quiet():
    """PyTorch has a few MPS kernels written in Python, registered while it is imported. They replace
    native.cpp's kernel for those operators, which is right (they run here, and what they are made
    of goes to the Mac, as on a Mac), and PyTorch warns of the replacement: not something a program
    did, or can act on."""
    import warnings

    warnings.filterwarnings("ignore", message="Warning only once for all operators")


class Loader(importlib.abc.Loader):
    """PyTorch's own loader, and install() once it has run."""

    def __init__(self, loader):
        self.loader = loader

    def __getattr__(self, name):  # whatever else is asked of a loader (resources, source) is PyTorch's
        return getattr(self.loader, name)

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        from .install import install

        install()


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "torch":
            return None
        sys.meta_path.remove(self)
        try:
            import bridge_native  # noqa: F401 - before torch, see native.cpp
        except ImportError:  # not the bridge's image: PyTorch as it is
            return None
        quiet()
        spec = importlib.util.find_spec("torch")
        if spec is not None and spec.loader is not None:
            spec.loader = Loader(spec.loader)
        return spec


if os.environ.get("BRIDGE", "1") != "0" and "torch" not in sys.modules:
    sys.meta_path.insert(0, Finder())
