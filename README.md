# 確率共鳴の比較実験 — v2.1

対称・非対称双安定、多重安定、単安定を、S/N比・方向別遷移・滞在割合で比較する研究用アプリです。文字・画像・映像の処理は含みません。

## Macで使う

[Macの起動手順](MACOS.md) を参照してください。Apple Silicon / Intel向け、推奨Python 3.12です。

```bash
bash run-macos.command
```

uvがある場合はPython 3.12と固定依存関係を用意します。Pythonだけの場合は専用の `.venv-macos` を作ります。アプリは `http://127.0.0.1:8501` にバインドします。終了はControl+C。初回は依存関係のダウンロードとNumbaコンパイルが必要です。

## 今回追加した内容

- 加法ノイズ用Heun法を既定にし、Euler法も比較用に保持。
- 白色ガウス、OU色付きガウス、連続時間二値ノイズ。同じ実現の区間積分を集約した3段階の刻み比較。
- SNR差の95%区間が許容幅内に収まるかを確認。CIの重なりだけではPASSにしない。
- 多重安定系の方向別遷移、井戸の滞在割合、回数/観測時間による経験的率、方向別の位相集中度。
- 有限振幅の周期FPを解き、平均応答と接続二時刻相関からSNRを計算。時間・空間・領域・位相分解能を変更して確認。
- 慣性を追加した決定論的Duffingモデルで、有限時間Lyapunov指数・Poincaré断面を計算。元の過減衰確率系とは別のモデル。
- Mac起動スクリプト、uv.lockと固定requirements、macOS / LinuxのCI。

## 実験画面

1. 「実験」: ポテンシャル、ノイズ・外力・積分法を設定し、ノイズ強度Dを掃引。
2. 「系・ノイズの比較」: 4系、障壁高さ、非対称性、障壁数、高さ差、ノイズ種類、相関時間を一括比較。
3. 「検証」: 独立試行による候補確認、結合経路の刻み比較、観測時間比較、有限振幅FP。
4. 「数式・手法」: 現在のポテンシャルとノイズに応じた式、数値代入、参考資料。
5. 「カオス」: 速度を追加した慣性モデルの別実験。

探索と検証・詳細計算はボタンで実行します。変更前の計算結果にはその旨を表示します。最初は4試行・8周期・5点・追加細分化0で操作を確認してください。少数試行の結果を研究上の結論には使わないでください。

## 数値モデルとノイズの定義

白色ノイズは `dX = [-U'(X)+A cos(2πft)]dt + sqrt(2D)dW`。OUと二値では `dX = [-U'(X)+A cos(2πft)+η]dt` とします。

OU・二値ノイズはともに `Cη(s) = (D/τ) exp(-|s|/τ)`、相関の片側積分をDに揃えます。同じDでも周波数ごとのPSDは一致しません。白色ノイズの瞬時分散と色付きノイズの有限分散を等しいものとして比較しません。

OUは定常初期値から、ノイズの区間積分と終端値の同時ガウス分布を厳密にサンプリングします。二値ノイズは ±sqrt(D/τ)、反転率1/(2τ)とし、指数待ち時間の連続時刻で反転させ、その区間積分を使います。時刻をdtに丸めた反転ではありません。

Heun法では予測子と補正子に同じノイズ積分を使います。ノイズ積分の正確さは、非線形ドリフトの積分や井戸状態qの誤差までゼロにするものではありません。指定dtを超えずに駆動周期とサンプル格子を整合させます。

試行ごとに独立したSeedSequenceを使い、Dと実験段階で乱数を分離します。同じD・同じ段階の比較モデル間には共通乱数を用いるため、モデル間の結果は独立ではありません。

## S/N比と理論の適用範囲

MCの標準指標は試行間の同期パワーをWelch近傍雑音パワーで割る推定量です。同期パワーの有限試行バイアスを除いた符号付き推定を使い、表示だけ0以上に制限します。95%CIは試行削除jackknife-t近似です。平均PSDと各試行SNR平均も選べます。後者2方式の通常CIは試行bootstrapです。雑音床ゼロは未定義のまま扱います。

Welchは周期Hann窓、50%重複、区間平均除去、片側density PSD、ゼロ埋めなし。DC・ガード帯域・高調波近傍を雑音床から除きます。線形SNRは無次元、dBは10 log10(SNR)です。

FP・Kramersの通常の掃引線は白色ガウスノイズの弱外力線形応答です。OU・二値ノイズでは「適用外」と表示し、白色用の理論を流用しません。Kramersは隣接井戸の多状態ネットワークで、井戸内運動を省く高障壁・遅い外力の近似です。ΔU/D≥5などの診断値を必ず確認してください。

有限振幅FPは反射境界のScharfetter–Gummel有限体積法と後退Euler法を使います。周期平均応答m(t)の基本波から同期パワーを計算し、位相平均した接続相関 `Cinc(s) = average_t[E(g(t+s)g(t))-m(t+s)m(t)]` の余弦変換から非同期PSDを求めます。平均波形だけをPSD化する方法ではありません。

周期残差・質量・正値性・端部確率・相関末尾・相関積分の観測長依存を確認します。基準を満たさないSNRは未定義にします。さらに時間半減、空間細分化、領域拡大、位相数倍増で変化5%以内かを調べます。有限領域・有限刻み・有限相関長の数値計算であり、連続方程式の厳密解や誤差の厳密上界ではありません。

## 精度と共鳴の確認

探索候補は調べたD点の最大です。独立乱数で候補・両端・A=0対照を再計算し、1モデル内の3つの片側比較にBonferroni補正を適用します。複数モデル全体の同時保証や真の最適Dの信頼区間ではありません。

刻み比較は、最細格子のノイズ積分を2個・4個ずつ足して同じ経路を共有します。dt→dt/2、dt/2→dt/4のSNR差をpaired jackknife-tで評価。観測時間は同じ長い軌道の前半Tと全体2Tで比較します。各差の95%区間全体が ±max(0.002, 参照SNRの10%) 内ならPASS、点差が許容幅を超えたらFAIL、点差は小さくても区間が広ければINCONCLUSIVEです。これらは有限試行の近似診断で、収束の数学的証明ではありません。

井戸状態q(x)は静的障壁で区切る記憶なしの観測量です。遷移統計は別途core-to-core判定で集計します。最初の未知時間は除外し、滞在区間の左右打切りを保持。回数/滞在時間は経験的な率で、周期駆動下の定常Kramers率や不偏MFPTではありません。中間井戸を飛び越した観測は警告し、架空の中間遷移を補いません。位相集中度は少数イベントでも大きくなるため、回数と一緒に解釈します。

障壁数のChebyshev系列は平均井戸間隔を揃えますが、局所曲率や全体幅は変わります。山の数だけの純粋な因果効果とは扱いません。

## GUIなしの再現実験

```bash
uv run --locked --extra fast python experiments.py --quick --noise all --preset multi --out results/smoke
uv run --locked --extra fast python experiments.py --validate --precision --nonlinear --out results/full
uv run --locked --extra fast python experiments.py --preset multi --noise ou --tau 0.5 --validate --out results/ou
uv run --locked --extra fast python experiments.py --duffing --out results/duffing
uv run --locked --extra fast --extra test python -m pytest -q
```

`--trials`・`--cycles`・`--dt`・`--scheme`で条件を変更できます。出力先は空にし、既存ファイルは上書きしません。CSV/JSON/ZIPにはパラメータ、ソフトウェア版、各D・試行の量、方向別遷移、滞在割合、検証結果を保存。詳細計算を行った場合は時系列・PSD・打切り区間・FPセル確率も保存します。

実行済み結果と環境は [TEST_REPORT.md](TEST_REPORT.md)、集計データは [examples/review](examples/review) を参照してください。旧v2の結果は旧コミットの検証報告に残っており、Heun法による現在の結果とは区別します。

## 参考資料

- Welch: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
- SR review: https://doi.org/10.1103/RevModPhys.70.223
- Finite-amplitude coherent/incoherent correlations: https://doi.org/10.1103/PhysRevE.67.036109
- Multistability and intermediate states: https://doi.org/10.1103/PhysRevE.82.011139
- Colored noise and multistability: https://doi.org/10.1063/1.5030578
- uv cross-platform locking: https://docs.astral.sh/uv/concepts/projects/layout/
