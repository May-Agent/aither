import fcntl
import json
import os
import re
import select
import shutil
import struct
import sys
import tempfile
import termios
import time

MAGIC = bytes.fromhex(sys.argv[1])
MUX_FD = int(sys.argv[2])
READY_FD = int(sys.argv[3])
PIPELINES = json.loads(sys.argv[4])

MAX_FRAME_DATA = 500
BACKPRESSURE_HIGH = 65536
SGR_COMPLETE_RE = re.compile(rb"\x1b\[[0-9;:]*m")
SGR_TAIL_RE = re.compile(rb"\x1b(?:\[[0-9;:]*)?\Z")
TIMES_PART_RE = re.compile(r"(\d+)m([0-9.]+)s")


class SgrStripper:
    def __init__(self):
        self.tail = b""

    def feed(self, chunk):
        if not chunk:
            return b""
        data = self.tail + chunk
        self.tail = b""
        m = SGR_TAIL_RE.search(data)
        if m and (len(data) - m.start()) <= 32:
            self.tail = data[m.start() :]
            data = data[: m.start()]
        return SGR_COMPLETE_RE.sub(b"", data)

    def flush(self):
        if not self.tail:
            return b""
        rem = SGR_COMPLETE_RE.sub(b"", self.tail)
        self.tail = b""
        return rem


def write_all(fd, data):
    view = memoryview(data)
    off = 0
    total = len(data)
    while off < total:
        n = os.write(fd, view[off:])
        off += n


def emit_frame(kind, step, payload=b""):
    if not payload:
        hdr = MAGIC + struct.pack(">BBH", kind, step, 0)
        write_all(MUX_FD, hdr)
        return
    view = memoryview(payload)
    off = 0
    total = len(payload)
    while off < total:
        part = bytes(view[off : off + MAX_FRAME_DATA])
        hdr = MAGIC + struct.pack(">BBH", kind, step, len(part))
        write_all(MUX_FD, hdr + part)
        off += len(part)


def parse_times_ns(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fp:
            lines = fp.read().splitlines()
        if len(lines) < 2:
            return 0
        total_sec = 0.0
        for m in TIMES_PART_RE.finditer(lines[1]):
            total_sec += int(m.group(1)) * 60.0 + float(m.group(2))
        return int(total_sec * 1_000_000_000)
    except OSError:
        return 0


def pipe_unread_bytes(fd):
    if fd < 0:
        return 0
    try:
        buf = fcntl.ioctl(fd, termios.FIONREAD, b"\x00\x00\x00\x00")
        return struct.unpack("i", buf)[0]
    except OSError:
        return 0


try:
    tap_dir = tempfile.mkdtemp(prefix=".aither-tap-")
except OSError:
    tap_dir = tempfile.mkdtemp(prefix=".aither-tap-", dir=".")

rset = set()
wset = set()

quit_path = os.path.join(tap_dir, "quit.done")
os.mkfifo(quit_path, 0o600)
quit_r_hold = os.open(quit_path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
quit_w_hold = os.open(quit_path, os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)

ctl_path = os.path.join(tap_dir, "ctl")
os.mkfifo(ctl_path, 0o600)
ctl_r = os.open(ctl_path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
ctl_w_hold = os.open(ctl_path, os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
rset.add(ctl_r)

outer_path = os.path.join(tap_dir, "outer.out")
os.mkfifo(outer_path, 0o600)
outer_r = os.open(outer_path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
outer_w_hold = os.open(outer_path, os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
rset.add(outer_r)


class StageState:
    def __init__(self, p_idx, k_idx, global_step, cfg, prev_eff_injected):
        self.p_idx = p_idx
        self.k_idx = k_idx
        self.step = global_step
        self.own_injected = bool(cfg["injected"])
        self.line_slicer = bool(cfg["line_slicer"])
        self.eff_injected = self.own_injected or (prev_eff_injected and self.line_slicer)
        self.strip_to_next = False
        self.out_stripper = SgrStripper()
        self.err_stripper = SgrStripper()

        prefix = os.path.join(tap_dir, f"p{p_idx}_s{k_idx}")
        self.times_path = f"{prefix}.times"

        os.mkfifo(f"{prefix}.out", 0o600)
        self.out_r = os.open(f"{prefix}.out", os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        self.out_w_hold = os.open(f"{prefix}.out", os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        self.out_polling = True
        rset.add(self.out_r)

        os.mkfifo(f"{prefix}.err", 0o600)
        self.err_r = os.open(f"{prefix}.err", os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        self.err_w_hold = os.open(f"{prefix}.err", os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        rset.add(self.err_r)

        os.mkfifo(f"{prefix}.ack", 0o600)
        self.ack_r_hold = os.open(f"{prefix}.ack", os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        self.ack_w = os.open(f"{prefix}.ack", os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)

        os.mkfifo(f"{prefix}.done", 0o600)
        self.done_r_hold = os.open(f"{prefix}.done", os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        self.done_w = os.open(f"{prefix}.done", os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)

        if k_idx > 0:
            os.mkfifo(f"{prefix}.in", 0o600)
            self.in_r_hold = os.open(f"{prefix}.in", os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            self.in_w = os.open(f"{prefix}.in", os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        else:
            self.in_r_hold = -1
            self.in_w = -1

        self.pending_in = bytearray()
        self.in_polling_out = False
        self.downstream_closed = False
        self.upstream_eof = k_idx == 0

        self.started_ns = None
        self.active_since_ns = None
        self.active_accum_ns = 0
        self.exit_code = None
        self.out_eof = False
        self.err_eof = False
        self.settled = False


pipelines = []
fd_to_out = {}
fd_to_err = {}
fd_to_in = {}
global_step = 0

for p_idx, p_spec in enumerate(PIPELINES):
    p_stages = []
    prev_eff = False
    for k_idx, s_cfg in enumerate(p_spec["stages"]):
        st = StageState(p_idx, k_idx, global_step, s_cfg, prev_eff)
        global_step += 1
        prev_eff = st.eff_injected
        p_stages.append(st)
        fd_to_out[st.out_r] = st
        fd_to_err[st.err_r] = st
        if st.in_w >= 0:
            fd_to_in[st.in_w] = st
    for k_idx in range(len(p_stages) - 1):
        p_stages[k_idx].strip_to_next = (
            p_stages[k_idx].eff_injected and not p_stages[k_idx + 1].line_slicer
        )
    pipelines.append(p_stages)


def close_fd(fd):
    if fd >= 0:
        rset.discard(fd)
        wset.discard(fd)
        try:
            os.close(fd)
        except OSError:
            pass
    return -1


def update_activity(st, now):
    if st.k_idx == 0 or st.started_ns is None or st.settled:
        return
    busy = (
        bool(st.pending_in)
        or st.upstream_eof
        or pipe_unread_bytes(st.in_w) > 0
    )
    if busy and st.active_since_ns is None:
        st.active_since_ns = now
    elif not busy and st.active_since_ns is not None:
        st.active_accum_ns += now - st.active_since_ns
        st.active_since_ns = None


def maybe_settle_stage(st):
    if st.settled or st.exit_code is None or not st.out_eof or not st.err_eof:
        return
    now = time.monotonic_ns()
    if st.k_idx == 0:
        start = st.started_ns if st.started_ns is not None else now
        own_ns = max(0, now - start)
    else:
        wall_ns = st.active_accum_ns
        if st.active_since_ns is not None:
            wall_ns += max(0, now - st.active_since_ns)
            st.active_since_ns = None
        cpu_ns = parse_times_ns(st.times_path)
        own_ns = max(wall_ns, cpu_ns)

    st.settled = True
    if st.in_w >= 0:
        fd_to_in.pop(st.in_w, None)
        st.in_w = close_fd(st.in_w)
        st.in_polling_out = False
        st.downstream_closed = True
        if st.k_idx > 0:
            prev = pipelines[st.p_idx][st.k_idx - 1]
            if prev.out_r >= 0:
                fd_to_out.pop(prev.out_r, None)
                prev.out_r = close_fd(prev.out_r)
                prev.out_eof = True
                maybe_settle_stage(prev)

    payload = struct.pack(">iQ", st.exit_code, own_ns)
    emit_frame(9, st.step, payload)
    if st.done_w >= 0:
        try:
            os.write(st.done_w, b"\n")
        except OSError:
            pass
        st.done_w = close_fd(st.done_w)
    st.done_r_hold = close_fd(st.done_r_hold)


def flush_stage_in(st):
    now = time.monotonic_ns()
    while st.pending_in and st.in_w >= 0:
        try:
            n = os.write(st.in_w, st.pending_in)
            del st.pending_in[:n]
        except BrokenPipeError:
            fd_to_in.pop(st.in_w, None)
            st.in_w = close_fd(st.in_w)
            st.in_polling_out = False
            st.downstream_closed = True
            st.pending_in.clear()
            prev = pipelines[st.p_idx][st.k_idx - 1]
            if prev.out_r >= 0:
                fd_to_out.pop(prev.out_r, None)
                prev.out_r = close_fd(prev.out_r)
                prev.out_eof = True
                maybe_settle_stage(prev)
            break
        except BlockingIOError:
            if not st.in_polling_out:
                wset.add(st.in_w)
                st.in_polling_out = True
            break

    if not st.pending_in and st.in_w >= 0 and st.in_polling_out:
        wset.discard(st.in_w)
        st.in_polling_out = False

    prev = pipelines[st.p_idx][st.k_idx - 1]
    if len(st.pending_in) < BACKPRESSURE_HIGH and prev.out_r >= 0 and not prev.out_polling:
        rset.add(prev.out_r)
        prev.out_polling = True

    if not st.pending_in and prev.out_eof and st.started_ns is not None and st.in_w >= 0:
        fd_to_in.pop(st.in_w, None)
        st.in_w = close_fd(st.in_w)
        st.in_polling_out = False
        st.upstream_eof = True

    update_activity(st, now)


def forward_from_stage(st, chunk, is_eof):
    p_stages = pipelines[st.p_idx]
    is_last = st.k_idx == len(p_stages) - 1
    if is_last:
        out_bytes = (
            (st.out_stripper.flush() if is_eof else st.out_stripper.feed(chunk))
            if st.eff_injected
            else chunk
        )
        if out_bytes:
            emit_frame(1, 0, out_bytes)
        return

    nxt = p_stages[st.k_idx + 1]
    fwd_bytes = (
        (st.out_stripper.flush() if is_eof else st.out_stripper.feed(chunk))
        if st.strip_to_next
        else chunk
    )
    if nxt.downstream_closed or nxt.in_w < 0:
        if not is_eof and st.out_r >= 0:
            fd_to_out.pop(st.out_r, None)
            st.out_r = close_fd(st.out_r)
            st.out_eof = True
            maybe_settle_stage(st)
        return

    if fwd_bytes:
        nxt.pending_in.extend(fwd_bytes)
    flush_stage_in(nxt)
    if len(nxt.pending_in) >= BACKPRESSURE_HIGH and st.out_r >= 0 and st.out_polling:
        rset.discard(st.out_r)
        st.out_polling = False


def drop_unstarted_stages():
    for p_stages in pipelines:
        for st in p_stages:
            if st.started_ns is None:
                fd_to_out.pop(st.out_r, None)
                fd_to_err.pop(st.err_r, None)
                if st.in_w >= 0:
                    fd_to_in.pop(st.in_w, None)
                st.out_r = close_fd(st.out_r)
                st.out_w_hold = close_fd(st.out_w_hold)
                st.err_r = close_fd(st.err_r)
                st.err_w_hold = close_fd(st.err_w_hold)
                st.in_r_hold = close_fd(st.in_r_hold)
                st.in_w = close_fd(st.in_w)
                st.ack_r_hold = close_fd(st.ack_r_hold)
                st.ack_w = close_fd(st.ack_w)
                st.done_r_hold = close_fd(st.done_r_hold)
                st.done_w = close_fd(st.done_w)
                st.settled = True


write_all(READY_FD, (tap_dir + "\n").encode("utf-8"))
os.close(READY_FD)

ctl_buf = b""
shell_opened = False
ctl_eof = False
outer_eof = False

try:
    while True:
        any_fionread_watch = False
        now_tick = time.monotonic_ns()
        for p_stages in pipelines:
            for st in p_stages:
                if st.k_idx > 0 and st.started_ns is not None and not st.settled:
                    update_activity(st, now_tick)
                    if st.active_since_ns is not None and not st.pending_in and not st.upstream_eof:
                        any_fionread_watch = True

        if ctl_eof and outer_eof and not fd_to_out and not fd_to_err:
            break
        if not rset and not wset:
            break

        timeout_sec = 0.01 if any_fionread_watch else None
        r_ready, w_ready, _ = select.select(list(rset), list(wset), [], timeout_sec)

        for fd in r_ready:
            if fd == ctl_r:
                while True:
                    try:
                        chunk = os.read(ctl_r, 4096)
                    except BlockingIOError:
                        break
                    if not chunk:
                        ctl_r = close_fd(ctl_r)
                        ctl_eof = True
                        drop_unstarted_stages()
                        break
                    ctl_buf += chunk
                while b"\n" in ctl_buf:
                    line, ctl_buf = ctl_buf.split(b"\n", 1)
                    parts = line.decode("utf-8", errors="replace").split()
                    if not parts:
                        continue
                    cmd = parts[0]
                    if cmd == "O" and not shell_opened:
                        shell_opened = True
                        ctl_w_hold = close_fd(ctl_w_hold)
                        outer_w_hold = close_fd(outer_w_hold)
                        quit_r_hold = close_fd(quit_r_hold)
                    elif cmd == "S":
                        p_i, k_i = int(parts[1]), int(parts[2])
                        st = pipelines[p_i][k_i]
                        st.started_ns = time.monotonic_ns()
                        st.out_w_hold = close_fd(st.out_w_hold)
                        st.err_w_hold = close_fd(st.err_w_hold)
                        st.in_r_hold = close_fd(st.in_r_hold)
                        emit_frame(4, st.step)
                        if k_i > 0:
                            flush_stage_in(st)
                        update_activity(st, st.started_ns)
                        st.ack_r_hold = close_fd(st.ack_r_hold)
                        st.done_r_hold = close_fd(st.done_r_hold)
                        if st.ack_w >= 0:
                            try:
                                os.write(st.ack_w, b"\n")
                            except OSError:
                                pass
                            st.ack_w = close_fd(st.ack_w)
                    elif cmd == "X":
                        p_i, k_i, rc = int(parts[1]), int(parts[2]), int(parts[3])
                        st = pipelines[p_i][k_i]
                        st.exit_code = rc
                        maybe_settle_stage(st)
                    elif cmd == "Q":
                        ctl_w_hold = close_fd(ctl_w_hold)
                        outer_w_hold = close_fd(outer_w_hold)
                        drop_unstarted_stages()
            elif fd == outer_r:
                while True:
                    try:
                        chunk = os.read(outer_r, 16384)
                    except BlockingIOError:
                        break
                    if not chunk:
                        outer_r = close_fd(outer_r)
                        outer_eof = True
                        break
                    emit_frame(1, 0, chunk)
            elif fd in fd_to_out:
                st = fd_to_out[fd]
                while st.out_r >= 0:
                    try:
                        chunk = os.read(st.out_r, 16384)
                    except BlockingIOError:
                        break
                    if not chunk:
                        fd_to_out.pop(st.out_r, None)
                        st.out_r = close_fd(st.out_r)
                        st.out_eof = True
                        forward_from_stage(st, b"", True)
                        maybe_settle_stage(st)
                        break
                    kind = 6 if st.eff_injected else 5
                    emit_frame(kind, st.step, chunk)
                    forward_from_stage(st, chunk, False)
                    if not st.out_polling:
                        break
            elif fd in fd_to_err:
                st = fd_to_err[fd]
                while st.err_r >= 0:
                    try:
                        chunk = os.read(st.err_r, 16384)
                    except BlockingIOError:
                        break
                    if not chunk:
                        fd_to_err.pop(st.err_r, None)
                        st.err_r = close_fd(st.err_r)
                        st.err_eof = True
                        rem = st.err_stripper.flush() if st.eff_injected else b""
                        if rem:
                            emit_frame(2, 0, rem)
                        maybe_settle_stage(st)
                        break
                    kind = 8 if st.eff_injected else 7
                    emit_frame(kind, st.step, chunk)
                    err_native = st.err_stripper.feed(chunk) if st.eff_injected else chunk
                    if err_native:
                        emit_frame(2, 0, err_native)

        for fd in w_ready:
            if fd in fd_to_in:
                flush_stage_in(fd_to_in[fd])
finally:
    shutil.rmtree(tap_dir, ignore_errors=True)
    close_fd(MUX_FD)
    close_fd(quit_w_hold)
