printf '%s pid %s\n' {{ pidfile }} "$$" >&2; ps -o pgid= -p $$ | tr -d ' ' > {{ pidfile }}; {{ command }}; status=$?; rm -f {{ pidfile }}; exit $status
