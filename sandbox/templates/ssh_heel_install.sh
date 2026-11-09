set -eu
bin="$HOME/.local/bin"
mkdir -p "$bin"
cat > "$bin/heel.partial"
chmod 755 "$bin/heel.partial"
"$bin/heel.partial" --version >/dev/null
mv "$bin/heel.partial" "$bin/heel"
printf '%s\n' "$bin/heel"
