"""A bridge from PyTorch in a Linux container to the GPU of the Mac it runs on.

Container end: native.cpp (the MPS backend), client.py, install.py, auto.py.
Mac end: server.py (the door), session.py with executor.cpp (one process per program).
Both: wire.py.
"""
