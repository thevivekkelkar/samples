# =============================================================
# client.py  —  SmartCity Traffic Control System
# =============================================================
# HTTP client for connecting to the running FastAPI server.
#
# Used by:
#   - inference.py (server mode)
#   - Anyone who wants to connect to your HuggingFace Space
#
# Note: train.py uses the environment DIRECTLY (no HTTP)
#       because that is faster for training.
#       This client is for testing, demos, and inference.
#
# HOW TO USE:
#   Terminal 1: cd server && python app.py
#   Terminal 2: python client.py
# =============================================================

import requests
import json
import random
from typing import List, Optional


class SmartCityClient:
    """
    Client that talks to the SmartCity Traffic API server.

    Example:
        client = SmartCityClient("http://localhost:8000")
        client.health()
        obs = client.reset(task="medium")
        while True:
            obs = client.step(agent_id=0, phase=2)
            if obs.get("done"):
                break
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()
        print(f"SmartCityClient → {self.base_url}")

    # ── Health check ──────────────────────────────────────────
    def health(self) -> dict:
        """Check if server is running."""
        r = self.session.get(f"{self.base_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    # ── Reset ─────────────────────────────────────────────────
    def reset(self, task: str = "easy") -> dict:
        """
        Reset the environment. Returns initial observation (agent 0).

        Args:
            task: "easy", "medium", "hard", or "expert"

        Returns:
            observation dict for agent 0
        """
        r = self.session.post(
            f"{self.base_url}/reset",
            json={"task": task}
        )
        r.raise_for_status()
        return r.json()

    # ── Step ──────────────────────────────────────────────────
    def step(self, agent_id: int, phase: int) -> dict:
        """
        Send one agent's action to the environment.

        Args:
            agent_id: which intersection (0-3)
            phase:    which lane gets green (0=N, 1=S, 2=E, 3=W)

        Returns:
            observation dict for this agent
        """
        r = self.session.post(
            f"{self.base_url}/step",
            json={"agent_id": agent_id, "phase": phase}
        )
        r.raise_for_status()
        return r.json()

    # ── Get current state ─────────────────────────────────────
    def get_state(self) -> dict:
        """Get full city state (all 4 intersections)."""
        r = self.session.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    # ── Metadata ──────────────────────────────────────────────
    def get_metadata(self) -> dict:
        """Get environment metadata."""
        r = self.session.get(f"{self.base_url}/metadata")
        r.raise_for_status()
        return r.json()

    # ── Run a full random episode ─────────────────────────────
    def run_random_episode(
        self,
        task:    str  = "easy",
        verbose: bool = False,
    ) -> float:
        """
        Run one complete episode with random agent actions.
        Useful for testing and baseline measurement.

        Returns:
            total_reward for the episode
        """
        self.reset(task=task)
        total   = 0.0
        step    = 0
        done    = False

        print(f"\nRunning random episode (task={task})...")

        while not done:
            total_step_reward = 0.0

            for agent_id in range(4):
                phase  = random.randint(0, 3)
                obs    = self.step(agent_id=agent_id, phase=phase)
                reward = obs.get("reward") or 0.0
                done   = obs.get("done", False)
                total_step_reward += reward

            total += total_step_reward
            step  += 1

            if verbose and step % 50 == 0:
                state = self.get_state()
                print(f"  Step {step:3d} | step_reward={total_step_reward:8.1f} | "
                      f"total={total:10.1f} | time={state.get('time_slot')}")

        print(f"  Done. Steps={step}  Total reward={total:.1f}")
        return total


# =============================================================
# QUICK TEST
# Run WHILE server is running in another terminal:
#   Terminal 1: cd server && python app.py
#   Terminal 2: python client.py
# =============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Testing SmartCityClient")
    print("=" * 50)

    client = SmartCityClient("http://localhost:8000")

    # Test 1: Health check
    print("\n1. Health check...")
    try:
        h = client.health()
        print(f"   Status: {h.get('status', '?')}")
    except Exception as e:
        print(f"   ERROR: {e}")
        print("   Is the server running? cd server && python app.py")
        exit(1)

    # Test 2: Reset
    print("\n2. Reset to easy...")
    obs = client.reset(task="easy")
    print(f"   Got observation keys: {list(obs.keys())}")

    # Test 3: Manual steps
    print("\n3. Taking 4 manual steps (one per agent)...")
    for agent_id in range(4):
        obs = client.step(agent_id=agent_id, phase=random.randint(0, 3))
        print(f"   Agent {agent_id}: reward={obs.get('reward'):.1f}  "
              f"done={obs.get('done')}")

    # Test 4: Full state
    print("\n4. Getting full city state...")
    state = client.get_state()
    print(f"   Step:       {state.get('step')}")
    print(f"   Time slot:  {state.get('time_slot')}")
    print(f"   Lane counts: {state.get('all_lane_counts')}")

    # Test 5: Full random episode
    print("\n5. Full random episode...")
    total = client.run_random_episode(task="easy", verbose=True)
    print(f"\n   Episode total: {total:.1f}")

    print("\nclient.py working correctly! ✓")
