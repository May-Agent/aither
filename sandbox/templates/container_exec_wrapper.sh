# Records this shell's own in-container pid in the file $1 for the stdin
# probe and the kill path (Docker reports exec pids in the host's namespace),
# announces "<file> pid $$" on stderr so the host registers the real
# in-container PID, and exports AITHER_CONTAINER_JOB=$$ for cgroup/RSS probes.
#
# A job's processes live exactly as long as the job. When the script's shell
# has exited, $3 says what becomes of the processes it left running, found
# through $4, the job script, by their marker:
#   foreground  they are killed, and one line "<file> killed <pid> <name>" per
#               process goes to stderr, the file's path being the nonce that
#               tells it from the script's own output; a kill that fails is
#               reported as "<file> error <message>";
#   background  the wrapper stays until none is left, so the exec ends with its
#               last process; a helper that fails ends the wait with the same
#               error line.
oneline() {
  printf '%s' "$1" | while IFS= read -r oneline_part || [ -n "$oneline_part" ]; do
    printf '%s ' "$oneline_part"
  done
}
wrapper_pid=$$
echo "$wrapper_pid" > "$1" || { echo "cannot record the exec's pid in $1" >&2; exit 125; }
printf '%s pid %s\n' "$1" "$wrapper_pid" >&2
trap 'rm -f "$1"' EXIT
exec 9<&0
AITHER_JOB=$1 AITHER_CONTAINER_JOB=$wrapper_pid sh -c 'exec 0<&9 9<&-; exec sh -c "$1"' sh "$2" &
child_pid=$!
exec 9<&-
forward_signal() {
  sig=$1
  kill "-$sig" "$child_pid" 2>/dev/null || true
  sh -c "$4" sh reap "$1" >/dev/null 2>&1 || true
}
trap 'forward_signal TERM; exit 143' TERM
trap 'forward_signal INT; exit 130' INT
trap 'forward_signal HUP; exit 129' HUP
wait "$child_pid"
status=$?
trap - TERM INT HUP
case $3 in
  foreground)
    if killed=$(sh -c "$4" sh reap "$1" 2>&1); then
      printf '%s\n' "$killed" | while read -r line; do
        [ -n "$line" ] && printf '%s killed %s\n' "$1" "$line" >&2
      done
    else
      printf '%s error %s\n' "$1" "$(oneline "$killed")" >&2
    fi
    ;;
  background)
    while :; do
      left=$(sh -c "$4" sh alive "$1") || {
        printf '%s error %s\n' "$1" "$(oneline "$left")" >&2
        break
      }
      [ -n "$left" ] || break
      sleep 1 || { printf '%s error cannot sleep while the job runs\n' "$1" >&2; break; }
    done
    ;;
esac
exit "$status"
