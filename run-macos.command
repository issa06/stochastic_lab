#!/bin/bash
set -eu
cd -- "$(dirname -- "$0")"

# Finder does not always inherit the shell's user-local executable paths.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
    echo "Python 3.12 と固定済みの依存関係を準備しています。"
    if ! uv sync --locked --python 3.12 --extra fast >/dev/null 2>&1; then
        echo "依存関係を準備できませんでした。ネット接続とMACOS.mdを確認してください。"
        exit 1
    fi
    exec uv run --no-sync python launcher.py "$@"
fi

for candidate in python3.12 python3.13 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(not ((3,11) <= sys.version_info[:2] <= (3,13)))' 2>/dev/null; then
        exec "$candidate" launcher.py --bootstrap "$@"
    fi
done
echo "Python 3.12 または uv をインストールしてください。手順は MACOS.md にあります。"
exit 1
