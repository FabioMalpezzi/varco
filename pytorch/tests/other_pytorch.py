"""A container whose PyTorch is not the Mac's is refused, and told both versions:   python other_pytorch.py   (in the container)

The two ends of the bridge must run the same PyTorch: which operators there are, what they take
and what a Mac runs on its GPU change from one version to the next. The container end says its
version in its greeting, and the Mac end refuses another (session.py). This checks, on the real
bridge and against the real server, what the program is left with:

- its first operation on the GPU fails, with an error that names the container's PyTorch and the
  Mac's, both, and says that the image and the Mac end have to be built for the same version,
  2.15.0 being the one this repository is built and tested for;
- nothing was computed for it, and the connection is not kept;
- with its own version back, the same process connects and computes.

The other version is made here, by giving PyTorch another name for the time of the greeting:
the image is not changed for it. Not a program of the batteries: on a Mac there is one PyTorch.
"""

import sys
from unittest.mock import patch

import torch

from bridge import install

device = torch.device("mps")
bridge = install.bridge
failures = []
OURS = torch.__version__.split("+")[0]
OTHER = "2.14.1"  # any version but the Mac's


def check(what, ok, detail=""):
    print(("ok      " if ok else "WRONG   ") + what + (f": {detail}" if detail and not ok else ""), flush=True)
    if not ok:
        failures.append(what)


check("the bridge has not spoken to the Mac yet", bridge.sock is None)
said = None
with patch.object(torch, "__version__", OTHER + "+cpu"):
    try:
        value = float(torch.arange(6.0, device=device).sum())
    except RuntimeError as e:
        said = str(e)
    else:
        check("a container with another PyTorch is refused", False, f"no error, and {value!r}")
if said is not None:
    check("a container with another PyTorch is refused", "must be the same" in said, said[:300])
    check(f"the error names the container's PyTorch ({OTHER})", f"the container has PyTorch {OTHER}" in said, said[:300])
    check(f"and the Mac's ({OURS})", f"the Mac's bridge server {OURS}" in said, said[:400])
    check("it says the image and the Mac end are built for one version", "have to be built for the same version" in said, said[:400])
    check("and which one this repository is built and tested for", "PyTorch 2.15.0 is the one this repository is built and tested for" in said, said[:400])
    check("it does not send the user to a server of the container's version", "server.sh" not in said and "TORCH=" not in said, said[:400])
    print("        " + said[:400], flush=True)
check("the refused connection is not kept", bridge.sock is None)

total = float(torch.arange(6.0, device=device).sum())
check("with its own PyTorch the same process connects", bridge.sock is not None)
check("and computes", total == 15.0, total)
print("all as it should be" if not failures else f"{len(failures)} WRONG")
sys.exit(1 if failures else 0)
