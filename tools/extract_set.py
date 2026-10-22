#!/usr/bin/env python3
"""
Extract a MammosighTR set directly from its raw split-zip parts.

The `.zip.001..N` files are raw byte splits of one big zip (7z `Type = Split`),
so they are NOT readable by 7zz as a zip. This script exposes them as a single
seekable file (MultiPart) and lets Python's zipfile read/extract them:
  - no `cat` step, no extra ~160 GB copy
  - per-file CRC checked automatically by zipfile
  - resumable: already-extracted files with the right size are skipped
  - optional --delete-parts only after a fully successful extraction

Examples:
  tools/extract_set.py EGITIM1 --list                 # inspect only (no writes)
  tools/extract_set.py EGITIM1                        # extract to extracted/
  tools/extract_set.py EGITIM1 --delete-parts         # extract then remove parts
"""
import argparse
import bisect
import io
import os
import re
import shutil
import sys
import time
import zipfile

SETS = {
    "EGITIM1": {"stem": "MG_EGITIM_1.zip.", "root": "MG_EGITIM_1", "entries": 19897},
    "EGITIM2": {"stem": "MG_EGITIM_2.zip.", "root": "MG_EGITIM_2", "entries": 33806},
    "YARISMA": {"stem": "MG_YARISMA.zip.", "root": "MG_YARISMA", "entries": 10001},
}


class MultiPart(io.RawIOBase):
    def __init__(self, paths):
        self.paths = paths
        self.sizes = [os.path.getsize(p) for p in paths]
        self.starts = []
        t = 0
        for s in self.sizes:
            self.starts.append(t)
            t += s
        self.total = t
        self.fps = [open(p, "rb") for p in paths]
        self.pos = 0

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.total + off
        self.pos = max(0, min(self.pos, self.total))
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.total - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.total:
            i = bisect.bisect_right(self.starts, self.pos) - 1
            off = self.pos - self.starts[i]
            self.fps[i].seek(off)
            c = self.fps[i].read(min(n, self.sizes[i] - off))
            if not c:
                break
            out += c
            self.pos += len(c)
            n -= len(c)
        return bytes(out)

    def close(self):
        for f in self.fps:
            f.close()
        super().close()


def find_parts(parts_dir, stem):
    pat = re.compile(re.escape(stem) + r"\d{3}$")
    files = [f for f in os.listdir(parts_dir) if pat.fullmatch(f)]
    return [os.path.join(parts_dir, f) for f in sorted(files)]


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def fmt_eta(sec):
    if sec < 0 or sec > 86400 * 7:
        return "--"
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", choices=sorted(SETS))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="extracted")
    ap.add_argument("--list", action="store_true", help="only list archive inventory, extract nothing")
    ap.add_argument("--delete-parts", action="store_true", help="delete .zip.00N parts after full success")
    ap.add_argument("--jobs", type=int, default=0, help="reserved; extraction is write-bound, keep 0")
    args = ap.parse_args()

    spec = SETS[args.set]
    parts_dir = os.path.join(args.data_dir, args.set)
    check_dir = args.data_dir if os.path.isdir(os.path.join(parts_dir, "")) or os.path.isdir(parts_dir) else args.data_dir
    parts = find_parts(parts_dir, spec["stem"])
    if not parts:
        parts = find_parts(args.data_dir, spec["stem"])
    if not parts:
        sys.exit(f"no parts matching {spec['stem']}NNN found under {parts_dir} or {args.data_dir}")

    comp = sum(os.path.getsize(p) for p in parts)
    print(f"set={args.set} parts={len(parts)} compressed={human(comp)}", flush=True)

    mp = MultiPart(parts)
    try:
        zf = zipfile.ZipFile(mp)
    except zipfile.BadZipFile as e:
        sys.exit(f"ERROR: bad zip: {e}")

    infos = zf.infolist()
    files = [i for i in infos if not i.is_dir()]
    dcm = [i for i in files if i.filename.lower().endswith(".dcm")]
    total = sum(i.file_size for i in files)
    print(
        f"entries={len(infos)} files={len(files)} dcm={len(dcm)} uncompressed={human(total)} "
        f"expected_entries={spec['entries']}",
        flush=True,
    )

    if args.list:
        mp.close()
        return

    out_root = os.path.join(args.out, spec["root"])
    os.makedirs(out_root, exist_ok=True)
    t0 = time.time()
    last = 0.0
    done_bytes = 0
    done_files = 0
    skipped = 0
    for info in infos:
        target = os.path.join(args.out, info.filename)
        if info.is_dir():
            os.makedirs(target, exist_ok=True)
            continue
        if os.path.exists(target) and os.path.getsize(target) == info.file_size:
            done_bytes += info.file_size
            done_files += 1
            skipped += 1
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".part"
        try:
            with zf.open(info) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        done_bytes += info.file_size
        done_files += 1
        now = time.time()
        if now - last >= 2.0:
            last = now
            el = now - t0
            rate = done_bytes / el if el > 0 else 0
            eta = (total - done_bytes) / rate if rate > 0 else -1
            pct = 100.0 * done_bytes / total if total else 100.0
            print(
                f"  [{pct:5.1f}%] {done_files}/{len(files)} files  {human(done_bytes)}/{human(total)}  "
                f"{human(rate)}/s  ETA {fmt_eta(eta)}",
                flush=True,
            )

    mp.close()
    el = time.time() - t0
    rate = done_bytes / el if el > 0 else 0
    status = "OK" if done_files == len(files) else "INCOMPLETE"
    print(
        f"DONE {args.set}: files={done_files}/{len(files)} skipped={skipped} bytes={human(done_bytes)} "
        f"in {fmt_eta(el)} ({human(rate)}/s) -> {out_root} [{status}]",
        flush=True,
    )
    if done_files != len(files):
        sys.exit(1)

    if args.delete_parts:
        freed = 0
        for p in parts:
            freed += os.path.getsize(p)
            os.remove(p)
        print(f"deleted {len(parts)} parts, freed {human(freed)}", flush=True)


if __name__ == "__main__":
    main()
