aither_py="${TMPDIR:-/tmp}/aither-{{ nonce }}.py"
trap 'rm -f "$aither_py"' EXIT
cat > "$aither_py" <<'AITHER_PYTHON_{{ nonce }}'
{{ code }}
AITHER_PYTHON_{{ nonce }}
{{ interpreter }} "$aither_py"

