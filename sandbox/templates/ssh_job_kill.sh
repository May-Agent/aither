[ -r {{ pidfile }} ] || exit 0
group=$(cat {{ pidfile }})
if kill -s TERM -- "-$group" 2>/dev/null; then
  for _ in 1 2 3 4 5; do
    kill -s 0 -- "-$group" 2>/dev/null || break
    sleep 1
  done
  kill -s KILL -- "-$group" 2>/dev/null
fi
rm -f {{ pidfile }}
exit 0
