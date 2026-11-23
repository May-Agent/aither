exec 9>&1
__aither_tap_dir=$(python3 -I -c {{ tap_py }} {{ magic }} 9 8 {{ spec }} 8>&1 >/dev/null 2>/dev/null &)
if [ -z "$__aither_tap_dir" ] || [ ! -d "$__aither_tap_dir" ]; then
  echo "pipeline tap failed to initialize" >&2
  exit 1
fi
exec 7<"$__aither_tap_dir/quit.done"
exec 8>"$__aither_tap_dir/ctl"
exec 1>"$__aither_tap_dir/outer.out"
printf 'O\n' >&8

__aither_stage() {
  local __p="$1" __k="$2" __env_src="$3" __cmd_src="$4" __rc=0
  exec 5<"$__aither_tap_dir/p${__p}_s${__k}.ack" 6<"$__aither_tap_dir/p${__p}_s${__k}.done"
  exec 1>"$__aither_tap_dir/p${__p}_s${__k}.out" 2>"$__aither_tap_dir/p${__p}_s${__k}.err"
  if [ "$__k" -gt 0 ]; then
    exec 0<"$__aither_tap_dir/p${__p}_s${__k}.in"
  fi
  printf 'S %s %s\n' "$__p" "$__k" >&8
  IFS= read -r _ <&5 || true
  exec 5<&-
  (
    exec 6<&- 7<&- 8>&- 9>&-
    if [ -n "$__env_src" ]; then
      eval "$__env_src"
    fi
    eval "$__cmd_src"
  )
  __rc=$?
  times >"$__aither_tap_dir/p${__p}_s${__k}.times" 2>/dev/null || true
  exec 0</dev/null 1>&- 2>&-
  printf 'X %s %s %s\n' "$__p" "$__k" "$__rc" >&8
  IFS= read -r _ <&6 || true
  exec 6<&-
  return "$__rc"
}

eval {{ rewritten_script }}
__aither_status=$?
{% if let Some(report) = report %}pwd -P > {{ report }}
{% endif %}exec 1>&-
printf 'Q\n' >&8
exec 8>&- 9>&-
IFS= read -r _ <&7 || true
exec 7>&-
exit "$__aither_status"
