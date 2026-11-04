"""Equation rendering with live numerical substitutions."""
from __future__ import annotations
import numpy as np
from models import PotentialModel,ModelSpec
from simulation import SimConfig
from spectra import SpectralConfig,make_bands
import kramers


REFERENCES = [
    ("Welch PSD / 単位・正規化", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html"),
    ("確率共鳴の理論・評価指標", "https://doi.org/10.1103/RevModPhys.70.223"),
    ("二状態モデル・コヒーレント／非コヒーレント成分", "https://arxiv.org/abs/cond-mat/0308627"),
    ("FP有限体積法・Scharfetter–Gummel", "https://arxiv.org/abs/2002.09385"),
    ("応答とSNRの最適ノイズは同一とは限らない", "https://doi.org/10.1103/PhysRevA.42.2030"),
    ("有限振幅応答・同期成分と非同期相関の分離", "https://doi.org/10.1103/PhysRevE.67.036109"),
    ("多重安定系と中間状態", "https://doi.org/10.1103/PhysRevE.82.011139"),
    ("色付きノイズと多重安定系", "https://doi.org/10.1063/1.5030578"),
]


def render_equations(st, model: PotentialModel, config: SimConfig, spectral: SpectralConfig, D: float):
    omega=2*np.pi*config.f
    dt=config.actual_dt
    bands=make_bands(dt,config.nperseg,config.f,spectral)
    st.subheader("ポテンシャルと時間発展")
    st.latex(r"U(x)="+model.latex())
    st.latex(r"-U'(x)="+model.latex(derivative=True))
    if config.noise_kind=="white":
        st.latex(r"dX_t=[-U'(X_t)+A\cos(\Omega t)]dt+\sqrt{2D}\,dW_t,\qquad\Omega=2\pi f")
        st.latex(rf"dX_t=\left[{model.latex(True)}+{config.A:g}\cos({omega:.6g}t)\right]dt+{np.sqrt(2*D):.6g}\,dW_t")
        st.latex(r"\Delta K_n=\sqrt{2D\Delta t}\,\xi_n,\quad\xi_n\sim\mathcal N(0,1)")
    else:
        st.latex(r"dX_t=[-U'(X_t)+A\cos(\Omega t)+\eta_t]dt,\quad\Delta K_n=\int_{t_n}^{t_{n+1}}\eta_s\,ds")
        st.latex(r"\langle\eta_t\eta_{t+s}\rangle=\frac D\tau e^{-|s|/\tau},\quad\int_0^\infty C_\eta(s)\,ds=D")
        if config.noise_kind=="ou":
            st.latex(r"d\eta_t=-\eta_t\,dt/\tau+\sqrt{2D}\,dW_t/\tau")
        else:
            st.latex(r"\eta_t\in\{-\sqrt{D/\tau},+\sqrt{D/\tau}\},\quad\lambda_{flip}=1/(2\tau)")
        st.write("ノイズの区間積分を直接生成します。以下のFP・Kramers式は白色ノイズ用の参照式で、現在のノイズには適用しません。")
    st.latex(r"b(x,t)=-U'(x)+A\cos(\Omega t),\quad X^*=X_n+b(X_n,t_n)\Delta t+\Delta K_n")
    if config.scheme=="heun":
        st.latex(r"X_{n+1}=X_n+\tfrac12[b(X_n,t_n)+b(X^*,t_{n+1})]\Delta t+\Delta K_n")
    else:
        st.latex(r"X_{n+1}=X^*")
    st.latex(rf"\Delta t={dt:.8g},\quad\sqrt{{2D\Delta t}}={np.sqrt(2*D*dt):.6g},\quad T_\Omega={1/config.f:.6g},\quad T_\Omega/2={1/(2*config.f):.6g}")
    st.subheader("PSDとS/N比")
    st.latex(r"\widehat S_g(\nu_k)=\frac{1}{M}\sum_{m=1}^{M}\frac{c_k}{f_s\sum_n w_n^2}\left|\sum_n w_n(g_{m,n}-\bar g_m)e^{-2\pi i kn/N_s}\right|^2")
    st.write("周期Hann窓、50%重複、区間平均除去。片側PSDの内部binは cₖ=2、DC・Nyquistは1。")
    st.latex(r"\widehat S_N=\frac{1}{|B_N|}\sum_{k\in B_N}\widehat S_g(\nu_k),\quad B=|B_S|\Delta\nu")
    st.latex(r"P_B=\sum_{k\in B_S}\widehat S_g(\nu_k)\Delta\nu,\quad P_N=\widehat S_N B,\quad P_S=\max(P_B-P_N,0)")
    st.latex(r"\mathrm{SNR}=\frac{P_S}{P_N},\qquad\mathrm{SNR}_{dB}=10\log_{10}(\mathrm{SNR})")
    st.latex(rf"\Delta\nu={bands.df:.6g},\quad B={bands.bandwidth:.6g},\quad B_S=[{bands.frequency[bands.signal[0]]:.6g},{bands.frequency[bands.signal[-1]]:.6g}]")
    if spectral.aggregation=="coherent":
        st.latex(r"z_j=\frac{2}{N_t}\sum_n g_j(t_n)e^{-i\Omega t_n},\quad\widehat P_{coh}=\frac{|\sum_jz_j|^2-\sum_j|z_j|^2}{2N_{MC}(N_{MC}-1)}")
        st.latex(r"\mathrm{SNR}_{coh}=\frac{\max(\widehat P_{coh},0)}{\overline P_N}")
        st.write("既知の駆動周波数に位相を合わせた同期成分を、異なる試行間の積から推定します。有限試行による信号パワーの正のバイアスを除き、雑音床はWelchの近傍binから計算します。95%CIは試行削除jackknife標準誤差とt近似。検定は切り捨て前の符号付き推定量を使います。")
    elif spectral.aggregation=="pooled_psd":
        st.latex(r"\widehat S_g=\frac1{N_{MC}}\sum_j\widehat S_{g,j},\qquad\mathrm{SNR}_{pool}=\frac{\max(\overline P_B-\overline P_N,0)}{\overline P_N}")
        st.write("主指標は平均PSDから計算。各試行SNRの算術平均とは別の推定量です。定数列はPSDの平均に含め、全試行で雑音床が0ならSNRは未定義とします。")
    else:
        st.latex(r"\overline{\mathrm{SNR}}=\frac1{N_{MC}}\sum_j\frac{\max(P_{B,j}-P_{N,j},0)}{P_{N,j}}")
        st.write("未定義の試行を除外した条件付き平均は行いません。1試行でも未定義なら、その点の平均とCIは未定義です。")
    if spectral.aggregation!="coherent":
        st.write("95%CIは試行を単位とするpercentile bootstrap。")
    st.write("CIは近似的な点ごとの区間であり、曲線全体の同時信頼帯ではありません。")
    st.subheader("Fokker–Planck")
    st.latex(r"\partial_t p=-\partial_x\{[-U'(x)+A\cos(\Omega t)]p\}+D\partial_x^2p,\qquad J(x_{min},t)=J(x_{max},t)=0")
    st.latex(r"\dot{\mathbf p}=Q(F)\mathbf p,\quad q_{i+1,i}=\frac{D}{\Delta x^2}\mathcal B\!\left(\frac{U_{i+1}-U_i-F\Delta x}{D}\right),\quad \mathcal B(z)=\frac{z}{e^z-1}")
    st.latex(r"[I-\Delta t_{FP}Q(F_{n+1})]\mathbf p_{n+1}=\mathbf p_n")
    st.write("SNR掃引は弱外力線形応答。検証タブでは有限振幅の周期状態と二時刻相関を直接計算できます。空間・時間・位相の分解能と相関の打切りを別途確認します。")
    st.latex(r"C_{inc}(s)=\frac1T\int_0^T\{\langle g(X_{t+s})g(X_t)\rangle-m(t+s)m(t)\}\,dt")
    st.latex(r"S_{inc}^{(1)}(\nu)=4\int_0^\infty C_{inc}(s)\cos(2\pi\nu s)\,ds,\quad P_{coh}=\tfrac12\left|\frac2T\int_0^T m(t)e^{-i\Omega t}\,dt\right|^2")
    st.latex(r"Q_0\mathbf p_0=0,\quad (i\Omega I-Q_0)\mathbf p_1=\left.\frac{\partial Q}{\partial F}\right|_{F=0}\mathbf p_0,\quad\chi_g=\mathbf g^T\mathbf p_1")
    st.latex(r"\mathbf g_c=\mathbf g-\langle g\rangle_0,\quad S_{g,0}^{(1)}(\nu)=4\operatorname{Re}\!\left[\mathbf g_c^T(i2\pi\nu I-Q_0)^{-1}(\mathbf p_0\odot\mathbf g_c)\right]")
    st.latex(r"P_{coh}=\frac{A^2|\chi_g(\Omega)|^2}{2},\quad\mathrm{SNR}_{LR}=\frac{P_{coh}}{B\,\operatorname{mean}_{k\in B_N}S_{g,0}^{(1)}(\nu_k)},\quad\eta=|\chi_g|^2")
    st.write("平均波形だけから雑音PSDを推定しません。MCと同じ帯域幅を使いますが、局所床の曲率・有限標本・非線形応答による差が残ります。")
    st.subheader("Kramersと時間尺度")
    st.latex(r"r_{i\to j}\simeq\frac{\sqrt{U''(x_i)|U''(x_b)|}}{2\pi}\exp\!\left[-\frac{U(x_b)-U(x_i)}D\right],\quad\tau_i=\frac1{\sum_jr_{i\to j}}")
    st.latex(r"\dot p_i=\sum_{j\ne i}(r_{j\to i}p_j-r_{i\to j}p_i)")
    rates=kramers.rate_table(model,D,config.f) if config.noise_kind=="white" else []
    if rates:
        for r in rates:
            st.latex(rf"r_{{{r['source']}\to{r['target']}}}\simeq {r['prefactor']:.6g}\exp\!\left(-\frac{{{r['delta_u']:.6g}}}{{{D:.6g}}}\right)={r['rate']:.6g}")
        st.latex(r"\tau_i\sim T_\Omega/2\quad\text{（時間尺度の目安。SNR最適条件ではない）}")
        st.latex(r"D_{match,ij}=\frac{\Delta U_{ij}}{\log(C_{ij}/2f)}\quad(C_{ij}>2f)")
    else:
        st.info("非退化の井戸間障壁がないためKramers率は適用外です。")
    st.write("弱ノイズ・高障壁・遅い外力が前提。多井戸の滞在時間は出る全方向の率の和から求めます。1本の率の逆数は、多井戸での方向別平均初通過時間とは限りません。")
    st.subheader("共鳴の根拠")
    st.latex(r"D^*_{grid}=\operatorname*{arg\,max}_{D\in\mathcal G}\widehat{\mathrm{SNR}}(D)")
    st.write("探索用データで候補を選び、独立乱数で候補・低D端・高D端・A=0を再計算。3つのSNR差の片側下限をBonferroni補正し、すべて正かを確認します。dt/2、観測時間×2、FP格子・領域変更も別に確認します。")
    st.write("q(x)は静的な障壁位置で区切る記憶なしの井戸番号です。時系列の遷移数・滞在時間だけはcore-to-core判定を使い、端の打切り区間を除外します。")
    with st.expander("参考資料"):
        for title,url in REFERENCES:
            st.markdown(f"[{title}]({url})")

