AITHER_IPC_BIN="$(command -v heel || true)";
if [ -z "$AITHER_IPC_BIN" ]; then echo "heel command not found in container" >&2; exit 127; fi;
AITHER_IPC_DIR="$(mktemp -d)";
cleanup(){ local __aither_ipc_status=$?; wait >/dev/null 2>&1 || true; rm -rf "$AITHER_IPC_DIR"; return $__aither_ipc_status; };
trap cleanup EXIT;
for cmd in {{ escaped_commands }}; do
printf '%s\n' '#!/usr/bin/env bash' "exec \"$AITHER_IPC_BIN\" ipc \"$cmd\" -- \"\$@\"" > "$AITHER_IPC_DIR/$cmd";
chmod +x "$AITHER_IPC_DIR/$cmd";
done;
export PATH="$AITHER_IPC_DIR:$PATH";
hash -r;
export HEEL_IPC_ENDPOINT="tcp://{{ ipc_host }}:{{ ipc_port }}";
