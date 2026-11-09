if command -v heel >/dev/null 2>&1; then
  command -v heel
elif [ -x "$HOME/.local/bin/heel" ]; then
  printf '%s\n' "$HOME/.local/bin/heel"
fi
