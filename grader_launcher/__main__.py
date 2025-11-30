# grader_launcher/__main__.py

import os
import sys
import signal
import subprocess
from pathlib import Path
from typing import List


def run_command(cmd: List[str], cwd: Path) -> subprocess.Popen:
    """
    Start a subprocess and return the Popen object.
    """
    return subprocess.Popen(cmd, cwd=str(cwd))


def ensure_frontend_dependencies(frontend_dir: Path) -> None:
    """
    Run `npm install` in the frontend directory if node_modules is missing.
    """
    node_modules = frontend_dir / "node_modules"
    if node_modules.exists():
        return

    print("[grader-agent] node_modules not found; running `npm install`...")
    result = subprocess.run(
        ["npm", "install"],
        cwd=str(frontend_dir),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"`npm install` failed with exit code {result.returncode} in {frontend_dir}"
        )
    print("[grader-agent] npm install complete.")


def main() -> None:
    """
    Start both backend (FastAPI) and frontend (Next.js dev) together.

    - Backend: uvicorn grader_backend.main:app
    - Frontend: npm run dev (inside grader_frontend)

    Env vars you can override:
      GRADER_BACKEND_HOST (default: 0.0.0.0)
      GRADER_BACKEND_PORT (default: 8000)
      GRADER_FRONTEND_PORT (default: 3000)
      GRADER_BACKEND_RELOAD (default: "true")
    """
    # Repo root: .../Grader_AI_Agent
    root = Path(__file__).resolve().parents[1]
    backend_dir = root  # backend module is importable by package
    frontend_dir = root / "grader_frontend"

    # --- Ensure frontend deps ---
    ensure_frontend_dependencies(frontend_dir)

    # --- Backend command ---
    host = os.getenv("GRADER_BACKEND_HOST", "0.0.0.0")
    port = os.getenv("GRADER_BACKEND_PORT", "8000")
    reload_flag = os.getenv("GRADER_BACKEND_RELOAD", "true").lower() == "true"

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "grader_backend.main:app",
        "--host",
        host,
        "--port",
        port,
    ]
    if reload_flag:
        backend_cmd.append("--reload")

    # --- Frontend command ---
    frontend_port = os.getenv("GRADER_FRONTEND_PORT", "3000")
    # NEXT_PUBLIC_API_BASE should point to backend
    os.environ.setdefault("NEXT_PUBLIC_API_BASE", f"http://localhost:{port}")

    frontend_cmd = [
        "npm",
        "run",
        "dev",
        "--",
        "--port",
        frontend_port,
    ]

    print(f"[grader-agent] Starting backend on {host}:{port}...")
    backend_proc = run_command(backend_cmd, cwd=backend_dir)

    print(f"[grader-agent] Starting frontend on http://localhost:{frontend_port} ...")
    frontend_proc = run_command(frontend_cmd, cwd=frontend_dir)

    # Graceful shutdown
    try:
        # Wait for either process to exit, or Ctrl+C
        while True:
            backend_ret = backend_proc.poll()
            frontend_ret = frontend_proc.poll()
            if backend_ret is not None:
                print(f"[grader-agent] Backend exited with code {backend_ret}")
                break
            if frontend_ret is not None:
                print(f"[grader-agent] Frontend exited with code {frontend_ret}")
                break
            # Sleep briefly
            try:
                # simple, portable sleep
                import time
                time.sleep(1.0)
            except KeyboardInterrupt:
                raise
    except KeyboardInterrupt:
        print("\n[grader-agent] Caught Ctrl+C, shutting down...")

    # Terminate children
    for proc, name in [(backend_proc, "backend"), (frontend_proc, "frontend")]:
        if proc.poll() is None:
            print(f"[grader-agent] Terminating {name}...")
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                proc.terminate()

    # Final wait
    try:
        backend_proc.wait(timeout=5)
    except Exception:
        backend_proc.kill()
    try:
        frontend_proc.wait(timeout=5)
    except Exception:
        frontend_proc.kill()

    print("[grader-agent] All processes stopped.")


if __name__ == "__main__":
    main()
