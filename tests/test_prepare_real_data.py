import io
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import prepare_real_data as prd

BODY = bytes(range(256)) * 4  # 1024 bytes standing in for the remote file


class _Handler(BaseHTTPRequestHandler):
    honor_range = True

    def do_GET(self):
        rng = self.headers.get("Range")
        if self.honor_range and rng:
            start, end = (int(v) for v in rng.removeprefix("bytes=").split("-"))
            part = BODY[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(BODY)}")
        else:  # a server or proxy that ignores Range: the whole file with 200
            part = BODY
            self.send_response(200)
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        self.wfile.write(part)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    def start(honor_range):
        handler = type("H", (_Handler,), {"honor_range": honor_range})
        httpd = HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_port}/base.fbin"
    servers = []
    yield start
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


def test_ranged_download_writes_only_the_prefix(server, tmp_path):
    src = dict(url=server(honor_range=True), size=100)
    dest = tmp_path / "x.fbin"
    prd.download("deep1m", src, dest)
    assert dest.read_bytes() == BODY[:100]


def test_ranged_download_refuses_server_that_ignores_range(server, tmp_path):
    src = dict(url=server(honor_range=False), size=100)
    dest = tmp_path / "x.fbin"
    with pytest.raises(RuntimeError, match="refusing"):
        prd.download("deep1m", src, dest)
    assert not dest.exists() and not dest.with_suffix(".fbin.part").exists()


def test_check_range_response():
    prd.check_range_response(206, "bytes 0-99/1000", 100)
    for status, cr in [(200, None), (206, "bytes 0-49/1000"), (206, None)]:
        with pytest.raises(RuntimeError):
            prd.check_range_response(status, cr, 100)


def test_copy_bounded_stops_at_size():
    out = io.BytesIO()
    prd.copy_bounded(io.BytesIO(BODY[:10]), out, 10, chunk=3)
    assert out.getvalue() == BODY[:10]
    with pytest.raises(RuntimeError, match="longer"):
        prd.copy_bounded(io.BytesIO(BODY), io.BytesIO(), 10, chunk=3)
    with pytest.raises(RuntimeError, match="early"):
        prd.copy_bounded(io.BytesIO(BODY[:5]), io.BytesIO(), 10, chunk=3)
