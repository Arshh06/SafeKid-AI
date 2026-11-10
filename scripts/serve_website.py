#!/usr/bin/env python3
"""SafeKid AI — Operational Security Monitoring & Control Room Server

Serves the SafeKid AI operator interface and research dashboard backed by the full
computer-vision pipeline, background job runner, REST API, ROI configuration engine,
community notification gateway, and ESP32 hardware device adapter.

Usage:
    python scripts/serve_website.py [--port 8080] [--bind 127.0.0.1]
"""

import argparse
from functools import partial
import http.server
import os
from pathlib import Path
import sys

# Ensure src is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
WEBSITE_DIR = REPO_ROOT / "website"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class SafeKidHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler with proper MIME types, CORS support, and QA compatibility."""

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.html': 'text/html; charset=utf-8',
        '.css': 'text/css; charset=utf-8',
        '.js': 'application/javascript; charset=utf-8',
        '.json': 'application/json; charset=utf-8',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.svg': 'image/svg+xml',
        '.mp4': 'video/mp4',
    }

    def end_headers(self):
        """Add security and caching headers."""
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, format, *args):
        """Custom structured server log formatting."""
        sys.stderr.write(f"[SafeKid WebServer] {self.log_date_time_string()} - {format % args}\n")


def run_server(port: int = 8080, host: str = "127.0.0.1"):
    """Starts the SafeKid AI operational control room server."""
    if not WEBSITE_DIR.exists():
        raise FileNotFoundError(f"Website root directory not found: {WEBSITE_DIR}")

    from safekid_ai.server.app import create_app
    from werkzeug.serving import run_simple

    app = create_app(base_dir=REPO_ROOT, website_dir=WEBSITE_DIR)

    print("=" * 72)
    print("SafeKid AI — Child Safety Monitoring Control Room & Operational Server")
    print("=" * 72)
    print(f"  Repo Root:      {REPO_ROOT}")
    print(f"  Website Dir:    {WEBSITE_DIR}")
    print(f"  Local Address:  http://{host}:{port}/")
    print(f"  REST API Base:  http://{host}:{port}/api/")
    print(f"  Status:         OPERATIONAL PIPELINE & HARDWARE ADAPTERS READY")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 72)

    run_simple(
        hostname=host,
        port=port,
        application=app,
        use_reloader=False,
        use_debugger=False,
        threaded=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the SafeKid AI operational application.")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--bind", type=str, default="127.0.0.1", help="Address to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    try:
        run_server(port=args.port, host=args.bind)
    except KeyboardInterrupt:
        print("\n[SafeKid WebServer] Server stopped by user.")
