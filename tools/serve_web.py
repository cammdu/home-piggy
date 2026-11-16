"""Serve web/ locally the way the Artifact host does: index.html is wrapped in a standard document skeleton.

Usage: python tools/serve_web.py [port]   -> http://localhost:8765/
"""
from __future__ import annotations

import http.server
import sys
from functools import partial
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
SKELETON = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);'
            'padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,sans-serif}'
            'img{max-width:100%}[hidden]{display:none!important}</style></head><body>{body}</body></html>')


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] in ("/", "/index.html"):
            body = SKELETON.replace("{body}", (WEB / "index.html").read_text(encoding="utf-8")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(WEB))).serve_forever()
