ps -o pgid= -p $$ | tr -d ' ' > {{ pidfile }}; {{ command }}; status=$?; rm -f {{ pidfile }}; exit $status
