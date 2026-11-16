"""Round-trip time of the connection to the Mac server, without mlx: sends
HELLO frames and waits for each reply. Usage: python rtt.py [host:port]"""
import socket, struct, sys, time
addr = sys.argv[1] if len(sys.argv) > 1 else "host.docker.internal:47800"
host, port = addr.rsplit(":", 1)
s = socket.create_connection((host, int(port)))
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
frame = bytes([1]) + struct.pack("<Q", 4) + struct.pack("<I", 1)
def recv(n):
    b = b""
    while len(b) < n:
        b += s.recv(n - len(b))
    return b
ts = []
for _ in range(2000):
    t = time.perf_counter()
    s.sendall(frame)
    head = recv(9)
    recv(struct.unpack("<Q", head[1:])[0])
    ts.append(time.perf_counter() - t)
ts.sort()
print(f"round trip to {addr}: median {1e6*ts[len(ts)//2]:.0f} us, best {1e6*ts[0]:.0f} us, p95 {1e6*ts[int(len(ts)*.95)]:.0f} us")
