"""
Real-time progress monitor for run_opendss_comp_gpu.py
Usage: python scripts/check_progress.py
"""

import re
import time
import os
from collections import defaultdict

LOG_FILE = "logs/gpu_run.log"
N_STEPS  = 240_000
INTERVAL = 5   # refresh every 5 seconds


def parse_log(path):
    agents      = defaultdict(lambda: {"steps": 0, "reward": 0.0, "mean100": 0.0})
    agent_order = []
    last_agent  = None

    ep_pat     = re.compile(r'ep\s+\d+\s+\|\s+steps\s+(\d+)\s+\|\s+reward\s+([-\d.]+)\s+\|\s+vviol\s+\d+\s+\|\s+mean100\s+([-\d.]+)')
    header_pat = re.compile(r'\[GPU\d\]\s+([\w\-]+)\s+seed=(\d+)')

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        m = header_pat.search(line)
        if m:
            last_agent = f"{m.group(1)} seed{m.group(2)}"
            if last_agent not in agent_order:
                agent_order.append(last_agent)

        m2 = ep_pat.search(line)
        if m2 and last_agent:
            steps   = int(m2.group(1))
            reward  = float(m2.group(2))
            mean100 = float(m2.group(3))
            if steps > agents[last_agent]["steps"]:
                agents[last_agent]["steps"]   = steps
                agents[last_agent]["reward"]  = reward
                agents[last_agent]["mean100"] = mean100

    return agents, agent_order


def render(agents, agent_order):
    os.system("clear")
    print("=" * 68)
    print(f"  QE-SAC GPU Training Monitor        target: {N_STEPS:,} steps/agent")
    print("=" * 68)
    print(f"  {'Agent':<24} {'Steps':>8}  {'%':>6}  {'Reward':>8}  {'Mean100':>8}")
    print("-" * 68)

    total_pct = 0.0
    count     = 0

    for name in agent_order:
        info  = agents[name]
        steps = info["steps"]
        pct   = steps / N_STEPS * 100
        filled = int(pct / 5)
        bar   = "#" * filled + "-" * (20 - filled)
        print(f"  {name:<24} {steps:>8,}  {pct:>5.1f}%  {info['reward']:>8.3f}  {info['mean100']:>8.3f}")
        print(f"  {'':24} [{bar}]")
        total_pct += pct
        count     += 1

    if count:
        avg = total_pct / count
        print("-" * 68)
        print(f"  {'OVERALL AVERAGE':<24} {'':>8}  {avg:>5.1f}%")

    print("=" * 68)
    print("  Paper targets:  QE-SAC -5.39  |  Classical-SAC -5.41  |  QC-SAC -5.91")
    print(f"  Auto-refresh every {INTERVAL}s  |  Ctrl+C to exit")
    print("=" * 68)


def main():
    print(f"Watching {LOG_FILE} ...")
    while True:
        if not os.path.exists(LOG_FILE):
            print(f"Waiting for {LOG_FILE} to appear ...")
            time.sleep(INTERVAL)
            continue
        try:
            agents, order = parse_log(LOG_FILE)
            render(agents, order)
        except Exception as e:
            print(f"Parse error: {e}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
