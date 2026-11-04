# Macで起動する

Apple Silicon（M1〜M4など）とIntel Macを対象にしたPythonアプリです。推奨Pythonは3.12、対応範囲は3.11〜3.13です。macOS実機の動作確認はまだ行っていません。

## uvを使う場合

Homebrewを利用している場合は、最初に `brew install uv` を実行してください。

リポジトリのフォルダで以下を実行します。

```bash
bash run-macos.command
```

Python 3.12と固定済み依存関係を準備し、ブラウザで開くローカルアプリを起動します。初回のみダウンロードが必要です。起動スクリプトをFinderから開く場合は、実行権限が必要です。macOSの確認画面が出る場合はターミナルから上記コマンドを使ってください。

```bash
uv run --locked --extra fast streamlit run app.py --server.address 127.0.0.1
```

## Pythonだけで起動する場合

[Python公式のmacOSインストーラ](https://www.python.org/downloads/macos/)でPython 3.12を準備し、同じ `bash run-macos.command` を実行します。プロジェクト内の `.venv-macos` に依存関係をインストールします。システムのPythonパッケージは変更しません。

## 操作

- `http://127.0.0.1:8501` を開く。終了はターミナルでControl+C。
- 最初は4〜8試行・8観測周期・5探索点・追加細分化0で動作確認。
- 研究用の既定値は30試行・48観測周期。精度検証ではさらに計算量が増えます。
- 「系・ノイズの比較」でノイズ種類と相関時間を比較。「時系列・確率密度を計算」で方向別遷移と滞在時間を表示。
- 「検証」で同じノイズ経路による3段階刻み比較と有限振幅FPの時間相関解析。
- 「カオス」は慣性を加えた別モデル。ノイズによる不規則な軌道とは区別します。

起動前の確認だけなら `bash run-macos.command --check`。ポート使用中の場合は `bash run-macos.command --port 8502` です。Numbaの初回コンパイル中は少し時間がかかります。

## 再計算・テスト

```bash
uv run --locked --extra fast --extra test pytest -q
uv run --locked --extra fast python experiments.py --quick --noise all --preset multi --out results/mac_smoke
uv run --locked --extra fast python experiments.py --validate --precision --out results/mac_full
```

結果の出力先は空のディレクトリを指定します。既存の結果は上書きしません。画像・文字・映像の入力機能は含まれていません。

依存関係は `uv.lock` と `requirements-lock.txt` で固定しています。異なるOS・CPUを含む依存解決を固定しますが、macOS実機での動作確認を代替するものではありません。
