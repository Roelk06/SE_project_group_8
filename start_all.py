import argparse
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"


def wait_for_orchestrator(url, seconds=30):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/health", timeout=1).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.3)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ui", action="store_true", help="do not start the Streamlit interface")
    parser.add_argument("--port", type=int, default=8000, help="orchestrator port (default 8000)")
    args = parser.parse_args()

    url = f"http://{HOST}:{args.port}"
    python = sys.executable
    processes = []

    def start(label, command):
        print(f"starting {label} ...")
        processes.append((label, subprocess.Popen(command, cwd=ROOT)))

    try:
        start("orchestrator", [python, "-m", "uvicorn", "services.orchestrator:app",
                               "--host", HOST, "--port", str(args.port), "--log-level", "warning"])
        if not wait_for_orchestrator(url):
            print("The orchestrator did not start. Is port", args.port, "already in use?")
            return 1

        orchestrator_args = ["--orchestrator", url]
        start("AI server (text, urls)", [python, "-m", "services.ai_server", "--name", "ai-text",
                                         "--types", "sms", "email", "url"] + orchestrator_args)
        start("AI server (images)", [python, "-m", "services.ai_server", "--name", "ai-image",
                                     "--types", "image"] + orchestrator_args)

        if not args.no_ui:
            env_note = "" if args.port == 8000 else f"  (set ORCHESTRATOR_URL={url} first)"
            start("interface" + env_note, [python, "-m", "streamlit", "run", str(ROOT / "services" / "app.py"),
                                           "--server.address", HOST,
                                           "--browser.gatherUsageStats", "false"])

        print("\nAll started. The AI servers need a few seconds to load their models.")
        print("Press Ctrl+C to stop everything.\n")
        while True:
            time.sleep(1)
            for label, process in processes:
                if process.poll() is not None:
                    print(f"\n'{label}' stopped (exit code {process.returncode}). Stopping everything.")
                    return 1
    except KeyboardInterrupt:
        print("\nStopping...")
        return 0
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    sys.exit(main())
