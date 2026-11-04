"""Local-only launcher with isolated dependencies and sanitized setup diagnostics."""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap',action='store_true')
    parser.add_argument('--check',action='store_true',help='Check dependencies without opening a server.')
    parser.add_argument('--port',type=int,default=8501)
    args=parser.parse_args()
    if not 1024<=args.port<=65535:
        parser.error('Port must be between 1024 and 65535.')
    if not (3,11)<=sys.version_info[:2]<=(3,13):
        raise SystemExit('Python 3.11–3.13 が必要です。推奨は3.12です。')
    if args.bootstrap:
        target=ROOT/'.venv-macos'
        python=target/'bin'/'python'
        if not python.exists():
            try:
                venv.EnvBuilder(with_pip=True).create(target)
            except Exception:
                raise SystemExit('仮想環境を作成できません。Python 3.12のインストールとフォルダの書込権限を確認してください。') from None
        # Package-manager output can include credential-bearing index URLs.
        # Capture it and expose only success/failure; no raw logs are saved.
        result=subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check',
                               '--index-url','https://pypi.org/simple','-r',str(ROOT/'requirements-lock.txt')],
                              cwd=ROOT,capture_output=True,text=True)
        if result.returncode:
            raise SystemExit('依存関係を準備できません。ネット接続・空き容量を確認して再実行してください。')
        command=[str(python),str(ROOT/'launcher.py')]
        if args.check: command.append('--check')
        command.extend(['--port',str(args.port)])
        return subprocess.call(command,cwd=ROOT)
    required=['numpy','scipy','pandas','plotly','streamlit']
    missing=[name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit('依存関係が不足しています。run-macos.commandから起動してください。')
    if args.check:
        print('PASS: Pythonと必要なパッケージを確認しました。')
        return 0
    print(f'ブラウザで http://127.0.0.1:{args.port} を開いてください。終了は Control+C。',flush=True)
    os.chdir(ROOT)
    os.execv(sys.executable,[sys.executable,'-m','streamlit','run',str(ROOT/'app.py'),
                            '--server.address=127.0.0.1',f'--server.port={args.port}',
                            '--browser.gatherUsageStats=false'])


if __name__=='__main__':
    sys.exit(main())
