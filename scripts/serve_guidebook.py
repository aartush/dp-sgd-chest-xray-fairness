"""Serve the guidebook on a local URL.

Binds to localhost only — this never exposes the page to the network. Opening
the HTML file directly also works; serving it is for when you want a real URL in
the address bar (slides, screen-sharing) or want to print to PDF as a website.

Usage:  python scripts/serve_guidebook.py [--port 8000] [--open]
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDEBOOK = ROOT / "docs" / "guidebook"


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter than the default
        print("  " + (fmt % args))


def serve(port: int, open_browser: bool) -> None:
    if not (GUIDEBOOK / "index.html").exists():
        raise SystemExit(
            "guidebook not found. Build it first:\n"
            "  .venv/Scripts/python.exe -m src.make_guidebook"
        )
    handler = functools.partial(Handler, directory=str(GUIDEBOOK))
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"Guidebook served at {url}")
        print("Press Ctrl+C to stop.")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="open a browser automatically")
    a = ap.parse_args()
    serve(a.port, a.open)
