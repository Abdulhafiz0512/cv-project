"""Fetch the organizers' sample videos from Google Drive.

The raw camera files are 4K H.264 4:2:2 10-bit at ~147 Mbit/s (2-6 GB each).
They are streamed straight from Drive into ffmpeg (no full local copy) and
written as 1080p H.264 with the original frame timing, which is what the dev
loop, EDA and website use. `--native-clip N` instead keeps the first N seconds
untouched (stream copy) to benchmark decoding of the real test-set format.

Drive answers a plain or open-ended `Range: bytes=0-` download of these files
with a "Quota exceeded" page, but serves bounded ranges. A tiny local proxy
therefore turns ffmpeg's reads into bounded, prefetched chunk requests.

    python scripts/fetch_samples.py --out data/samples
    python scripts/fetch_samples.py --out data/samples_4k --native-clip 12 --only C3905
"""
from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import imageio_ffmpeg

# file name on the camera -> Google Drive file id (links from Videos.pdf)
SAMPLES = {  # shortest first, so work can start on it early
    "C3905": "1aJ-QsAZVYJtLKHiRvKKeBq1D3GWNobRd",
    "C3896": "1kR9jODA2Wotw4gwkvpRKdqFADNJNc1nS",
    "C3897": "1hp8DYeqtYHSwfM6qAo9FPSRHlpMFrIN_",
    "C3902": "10cHEReCWzO3u-Vk1CnNgHAx6egGy5MwJ",
}
DRIVE_URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
CHUNK = 4 << 20      # bytes per bounded range request (see --chunk-kb)
PREFETCH = 2         # concurrent chunk requests per stream (see --prefetch)
PROGRESS: dict[str, int] = {}  # drive id -> bytes served to ffmpeg


def _get(file_id: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(DRIVE_URL.format(file_id), headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        if resp.status != 206 or len(data) != end - start + 1:
            raise OSError(f"unexpected reply {resp.status}, {len(data)} bytes")
        return data


def fetch_range(file_id: str, start: int, end: int, cancel: threading.Event) -> bytes:
    """Bounded range read. Drive's per-file download quota answers a random share
    of requests with a "Quota exceeded" page; retrying soon after usually works."""
    while not cancel.is_set():
        try:
            return _get(file_id, start, end)
        except OSError:
            cancel.wait(random.uniform(3.0, 8.0))
    raise OSError("cancelled")


def remote_size(file_id: str) -> int:
    while True:
        try:
            req = urllib.request.Request(DRIVE_URL.format(file_id), headers={"Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return int(resp.headers["Content-Range"].rsplit("/", 1)[1])
        except (OSError, KeyError, ValueError):
            time.sleep(random.uniform(2.0, 6.0))


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address) -> None:  # dropped connections are expected
        pass


class RangeProxy(BaseHTTPRequestHandler):
    """GET /<drive_id> with an optional Range header, served from bounded chunks."""
    sizes: dict[str, int] = {}
    lock = threading.Lock()

    def log_message(self, *args) -> None:  # keep ffmpeg's console clean
        pass

    def _size(self, file_id: str) -> int:
        with self.lock:
            if file_id not in self.sizes:
                self.sizes[file_id] = remote_size(file_id)
            return self.sizes[file_id]

    def do_GET(self) -> None:
        file_id = self.path.strip("/")
        total = self._size(file_id)
        start, end = 0, total - 1
        m = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        if m:
            start = int(m.group(1))
            end = min(end, int(m.group(2))) if m.group(2) else end
        self.send_response(206 if m else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if m:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.end_headers()
        bounds = [(a, min(a + CHUNK - 1, end)) for a in range(start, end + 1, CHUNK)]
        cancel = threading.Event()
        pool = ThreadPoolExecutor(PREFETCH)
        pending = [pool.submit(fetch_range, file_id, a, b, cancel) for a, b in bounds[:PREFETCH]]
        nxt = PREFETCH
        try:
            while pending:
                data = pending.pop(0).result()
                if nxt < len(bounds):
                    pending.append(pool.submit(fetch_range, file_id, *bounds[nxt], cancel))
                    nxt += 1
                self.wfile.write(data)
                PROGRESS[file_id] = PROGRESS.get(file_id, 0) + len(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # ffmpeg seeked elsewhere and dropped this connection
        finally:
            cancel.set()  # stop retry loops of prefetches nobody will read
            pool.shutdown(wait=False, cancel_futures=True)


def _seconds(stamp: str) -> float:
    h, m, sec = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def transcode(name: str, url: str, out_dir: Path, height: int, crf: int, native_clip: float,
              attempts: int = 3) -> str:
    """Transcode one remote file; retried until the output covers the whole source."""
    dst = out_dir / f"{name}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        return f"{name}: exists, skipped"
    tmp = dst.with_suffix(".part.mp4")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-stats_period", "30", "-y", "-i", url]
    if native_clip > 0:
        cmd += ["-t", str(native_clip), "-map", "0:v:0", "-c", "copy"]
    else:
        cmd += ["-map", "0:v:0", "-vf", f"scale=-2:{height}:flags=area", "-fps_mode", "passthrough",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                # fragmented output stays readable while the (slow) download is still running
                "-movflags", "+frag_keyframe+empty_moov+default_base_moof"]
    for attempt in range(1, attempts + 1):
        t0 = time.time()
        proc = subprocess.run([*cmd, str(tmp)], capture_output=True, text=True)
        src = re.search(r"Duration: (\d+:\d+:[\d.]+)", proc.stderr)
        out = re.findall(r"time=(\d+:\d+:[\d.]+)", proc.stderr)
        want = native_clip if native_clip > 0 else (_seconds(src.group(1)) if src else 0.0)
        got = _seconds(out[-1]) if out else 0.0
        if proc.returncode == 0 and want > 0 and got >= want - 0.5:
            tmp.replace(dst)
            return f"{name}: ok ({dst.stat().st_size / 1e6:.0f} MB, {got:.1f}s of video, {time.time() - t0:.0f}s)"
        print(f"{name}: attempt {attempt} incomplete ({got:.1f}/{want:.1f}s)\n{proc.stderr[-800:]}", flush=True)
        tmp.unlink(missing_ok=True)
    return f"{name}: FAILED after {attempts} attempts"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/samples")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--crf", type=int, default=25)
    ap.add_argument("--native-clip", type=float, default=0.0, help="seconds of untouched 4K to keep")
    ap.add_argument("--only", nargs="*", help="subset of sample names")
    ap.add_argument("--parallel", type=int, default=4, help="files transcoded at once")
    ap.add_argument("--chunk-kb", type=int, default=4096, help="bytes per range request (quota-limited Drive "
                    "files succeed more often with small chunks)")
    ap.add_argument("--prefetch", type=int, default=2, help="concurrent range requests per stream")
    args = ap.parse_args()

    global CHUNK, PREFETCH
    CHUNK, PREFETCH = args.chunk_kb << 10, args.prefetch
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    server = QuietServer(("127.0.0.1", 0), RangeProxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def report() -> None:
        while True:
            time.sleep(60)
            done = {k: f"{PROGRESS.get(v, 0) / 1e9:.2f} GB" for k, v in SAMPLES.items() if v in PROGRESS}
            print(time.strftime("%H:%M"), done, flush=True)
    threading.Thread(target=report, daemon=True).start()

    todo = {k: v for k, v in SAMPLES.items() if not args.only or k in args.only}
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        jobs = [pool.submit(transcode, k, f"{base}/{v}", out_dir, args.height, args.crf, args.native_clip)
                for k, v in todo.items()]
        for job in jobs:
            print(job.result(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
