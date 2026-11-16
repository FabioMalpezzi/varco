"""Run a command that loads a large model, and stop it at once if the Mac
runs short of memory: free memory under a floor, or swap in use growing.
    python3 mem_guard.py [--floor-gb 8] [--swap-mb 256] -- COMMAND...
When it stops the command it also stops the Mac server (server/ctl.sh stop)
and removes the container of tests/lm.sh, which hold the model.
Prints the least free memory it saw.
"""
import os
import re
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sysctl(name):
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def free_gb(total):
    # The kernel's own figure, the one memory_pressure prints.
    return total * int(sysctl("kern.memorystatus_level")) / 100


def swap_mb():
    m = re.search(r"used = ([\d.,]+)M", sysctl("vm.swapusage"))
    return float(m.group(1).replace(",", ".")) if m else 0.0


def main():
    args = sys.argv[1:]
    floor, swap = 8.0, 256.0
    while args and args[0] != "--":
        if args[0] == "--floor-gb":
            floor = float(args[1])
        elif args[0] == "--swap-mb":
            swap = float(args[1])
        args = args[2:]
    command = args[1:]
    total = int(sysctl("hw.memsize")) / 2**30
    swap_start = swap_mb()
    least = free_gb(total)
    print(f"[guard] {least:.0f} GB free of {total:.0f}, swap in use {swap_start:.0f} MB; "
          f"stops under {floor:.0f} GB free or {swap:.0f} MB more swap", flush=True)
    if least < floor + 4:
        print("[guard] not enough free memory to start", flush=True)
        return 3
    proc = subprocess.Popen(command, start_new_session=True)
    why = None
    while proc.poll() is None:
        time.sleep(0.5)
        free = free_gb(total)
        least = min(least, free)
        if free < floor:
            why = f"free memory fell to {free:.1f} GB"
        elif swap_mb() - swap_start > swap:
            why = f"swap in use grew to {swap_mb():.0f} MB"
        if why:
            print(f"[guard] STOPPING: {why}", flush=True)
            os.killpg(proc.pid, signal.SIGKILL)
            subprocess.run([os.path.join(ROOT, "server/ctl.sh"), "stop"])
            subprocess.run(["docker", "rm", "-f", "varco-mlx-lm-run"], capture_output=True,
                           env={"DOCKER_HOST": f"unix://{os.path.expanduser('~')}/.docker/run/docker.sock", **os.environ})
            break
    proc.wait()
    print(f"[guard] least free memory seen: {least:.0f} GB; swap in use now {swap_mb():.0f} MB", flush=True)
    return 4 if why else proc.returncode


if __name__ == "__main__":
    sys.exit(main())
