# Streamlit Community Cloud 公開手順

1. https://share.streamlit.io/ でGitHubアカウントを連携する。
2. リポジトリ `issa06-dev/stochastic_lab`、ブランチ `main`、エントリーポイント `app.py` を選択する。
3. Pythonバージョンに `3.12` を選ぶ。依存関係はルートの `requirements.txt` に記載している。
4. 初回は小さな試行数で実行し、SNR(D)グラフとエクスポートを確認する。

計算負荷の大きいMonte CarloやFokker–Planck解析は、Community Cloudの実行資源や制限時間に達する場合がある。初期公開後、スマートフォンで操作とグラフ表示を確認すること。

`.streamlit/config.toml` は `server.address` を固定せず、ホスティング側で設定できるようにしている。
