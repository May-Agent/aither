AITHER_IPC_BIN="$(command -v heel || true)";
if [ -z "$AITHER_IPC_BIN" ]; then echo "heel command not found in container" >&2; exit 127; fi;
AITHER_IPC_DIR="$(mktemp -d)";
cleanup(){ local __aither_ipc_status=$?; wait >/dev/null 2>&1 || true; rm -rf "$AITHER_IPC_DIR"; return $__aither_ipc_status; };
trap cleanup EXIT;
"$AITHER_IPC_BIN" shims --dir "$AITHER_IPC_DIR" --commands {{ escaped_commands }} || exit $?;
if [ -f "$AITHER_IPC_DIR/heel-shim" ]; then
# `heel-shim` knows itself by executable name and reads "<name>.json" beside
# any other one: the real binary is renamed so this script can take its place,
# and the name it moves to must keep the "heel-shim" file stem.
mv "$AITHER_IPC_DIR/heel-shim" "$AITHER_IPC_DIR/heel-shim.bin" || exit $?;
cat <<'__AITHER_IPC_SHIM__' > "$AITHER_IPC_DIR/heel-shim" || exit $?;
#!/bin/sh
__aither_cmd="$1"
shift
exec "${0%/*}/heel-shim.bin" "{{ ipc_token }}:$__aither_cmd" "$@"
__AITHER_IPC_SHIM__
chmod 0700 "$AITHER_IPC_DIR/heel-shim" || exit $?;
fi;
export PATH="$AITHER_IPC_DIR:$PATH";
hash -r;
export HEEL_IPC_ENDPOINT="tcp://{{ ipc_host }}:{{ ipc_port }}";
