"""Japanese research UI for stochastic resonance, without hidden auto-runs."""
from __future__ import annotations
from dataclasses import asdict, replace
import io
import json
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import pandas as pd
import streamlit as st
from models import ModelSpec,PotentialModel,LABELS,preset_specs
from simulation import SimConfig,simulate
from spectra import SpectralConfig
from analysis import run_experiment,validate_experiment,detail_simulation,effective_observable
from plotting import (potential_figure,sweep_figure,comparison_figure,time_figure,psd_figure,
                      density_figure,mean_figure,METHOD_LABELS)
from exporting import csv_bytes,parameter_rows,experiment_zip
from equations import render_equations
from noise import NOISE_LABELS
from accuracy import timestep_study
from fokker_planck import make_grid
from spectra import make_bands
from nonlinear_fp import resolution_study
from chaos import DuffingConfig,duffing_refinement
from plotting import transition_figure,phase_figure,chaos_figure

st.set_page_config(page_title="確率共鳴の比較実験",layout="wide")
st.title("確率共鳴")


def number(label,value,step=.01,min_value=None,max_value=None,key=None):
    return st.number_input(label,value=float(value),step=float(step),min_value=min_value,
                           max_value=max_value,format="%.6g",key=key)


def model_controls():
    kind=st.selectbox("系",list(LABELS),format_func=LABELS.get,key="model_kind")
    kwargs={"kind":kind}
    if kind in ("symmetric","asymmetric"):
        cols=st.columns(2)
        with cols[0]: kwargs["a"]=number("a",1,.1,.01,key="a")
        with cols[1]: kwargs["b"]=number("b",1,.1,.01,key="b")
        if kind=="asymmetric": kwargs["c"]=number("非対称性 c",.04,.01,key="c")
    elif kind=="multi":
        for key,value in [("a6",1),("a4",-2),("a2",1),("a1",0)]:
            kwargs[key]=number(key,value,.05,key=key)
    elif kind=="mono":
        kwargs["k"]=number("k",1,.1,0.0,key="k")
        kwargs["b"]=number("4次係数 b",0,.1,0.0,key="b_mono")
    else:
        kwargs["wells"]=st.slider("井戸数",1,5,3,key="wells")
        kwargs["height"]=number("基準障壁高 h",.25,.025,.001,key="height")
        kwargs["skew"]=st.slider("障壁高の左右差",-1.0,1.0,0.0,.1,key="skew")
        kwargs["c"]=number("傾き c",0,.01,key="c_family")
    return ModelSpec(**kwargs)


def safe_error(error):
    if isinstance(error,ValueError):
        st.error(str(error))
    else:
        st.error("計算を完了できませんでした。刻み幅・観測時間・格子数を見直してください。")


def format_number(value):
    return f"{value:.5g}" if value is not None and np.isfinite(value) else "—"


def run_with_progress(spec,config,spectral,grid,observable,nx,refine):
    progress=st.progress(0.0)
    try:
        return run_experiment(spec,config,spectral,grid,observable,nx,refine,
                              progress=lambda done,total:progress.progress(min(1.0,done/total)))
    finally:
        progress.empty()


def evidence_table(experiment):
    e=experiment["evidence"]
    checks=[{"確認項目":"探索範囲の内部に最大","結果":"はい" if e.get("interior") else "いいえ"},
            {"確認項目":"ピーク / 両端の大きい方","結果":format_number(e.get("peak_to_edge",np.nan))},
            {"確認項目":"SNRを推定できなかったD点","結果":str(e.get("missing_points",0))},
            {"確認項目":"低D・高Dの比較対象","結果":f"{format_number(e.get('low_D',np.nan))} / {format_number(e.get('high_D',np.nan))}"}]
    k_rows=[r for r in experiment["rows"] if r["method"]=="Kramers" and np.isfinite(r.get("snr",np.nan))]
    if k_rows and np.isfinite(e.get("D_star",np.nan)):
        near=min(k_rows,key=lambda r:abs(r["D"]-e["D_star"]))
        checks.append({"確認項目":"候補でのKramers高障壁目安（ΔU/D ≥ 5）",
                       "結果":f"{near.get('barrier_over_D',np.nan):.3g}"})
        checks.append({"確認項目":"候補でのKramers近似診断",
                       "結果":"目安を満たす" if near.get("kramers_reliable") else "適用条件に注意"})
    validation=experiment.get("independent_validation")
    if validation:
        checks.extend({"確認項目":r["check"],"結果":r["result"]} for r in validation.get("checks",[]))
    else:
        checks.append({"確認項目":"独立試行での再確認","結果":"未実行"})
    st.dataframe(pd.DataFrame(checks),hide_index=True,use_container_width=True)
    if e.get("missing_points",0):
        st.warning("低Dなどで推定不能な点があります。端部比較は有限のSNRが得られた範囲に限定します。")


def show_details(experiment,prefix):
    ds=sorted({r["D"] for r in experiment["rows"] if r["method"]=="MC"})
    if not ds:
        return
    best=experiment["evidence"].get("D_star",ds[len(ds)//2])
    if not np.isfinite(best): best=ds[len(ds)//2]
    options={"弱いノイズ":ds[0],"候補付近":float(best),"強いノイズ":ds[-1],"指定":float(best)}
    c1,c2=st.columns(2)
    choice=c1.selectbox("表示するノイズ",list(options),index=1,key=f"{prefix}_detail_choice")
    selected=options[choice]
    if choice=="指定":
        selected=c2.selectbox("D",ds,index=int(np.argmin(abs(np.array(ds)-best))),key=f"{prefix}_detail_D")
    with st.expander("確率密度の数値条件"):
        dt_fp=number("FP時間刻み上限",.05,.01,.001,2.0,key=f"{prefix}_dt_fp")
        max_cycles=st.number_input("周期定常への反復上限",min_value=3,max_value=300,value=80,key=f"{prefix}_fp_cycles")
    if st.button("時系列・確率密度を計算",key=f"{prefix}_detail_run"):
        try:
            with st.spinner("計算中"):
                detail=detail_simulation(experiment,float(selected),dt_fp,int(max_cycles))
            st.session_state[f"{prefix}_detail"]=(float(selected),detail)
        except (ValueError,ArithmeticError) as error:
            safe_error(error)
    stored=st.session_state.get(f"{prefix}_detail")
    if stored is None:
        return
    D,detail=stored
    if D!=selected:
        st.info("表示中の結果は前回計算したDです。選択を反映するには再計算してください。")
    cfg=SimConfig(**experiment["simulation"])
    st.subheader(f"時系列 · D = {D:.5g}")
    st.plotly_chart(time_figure(detail["t"],detail["x"],detail["y"],cfg.A,cfg.f),use_container_width=True,key=f"{prefix}_time")
    st.plotly_chart(psd_figure(detail["frequency"],detail["psd"],detail["bands"]),use_container_width=True,key=f"{prefix}_psd")
    periodic=detail["fp_periodic"]
    if periodic is not None:
        st.subheader("FP 確率密度")
        st.plotly_chart(density_figure(periodic),use_container_width=True,key=f"{prefix}_density")
        st.plotly_chart(mean_figure(periodic,detail["kramers_periodic"],cfg.A,cfg.f),use_container_width=True,key=f"{prefix}_means")
        if not periodic["converged"]:
            st.warning("FPの周期定常状態が収束判定を満たしていません。反復上限を増やしてください。")
        with st.expander("保存則・周期収束"):
            st.dataframe(pd.DataFrame([{"質量誤差":periodic["mass_error"],"最小セル確率":periodic["minimum_probability"],
                                        "周期間L1差":periodic["periodic_residual"],"端部確率":periodic["boundary_mass"],
                                        "反復周期数":periodic["cycles"],"実際のFP dt":periodic["dt"]}]),hide_index=True)
    else:
        st.info("色付き・二値ノイズはMCで解析します。白色ノイズ用の1次元FP・Kramers式は適用外です。")
    directed=detail["transition_analysis"]
    if directed["available"]:
        st.subheader("多重安定系の遷移")
        st.plotly_chart(transition_figure(directed),use_container_width=True,key=f"{prefix}_transitions")
        st.plotly_chart(phase_figure(directed),use_container_width=True,key=f"{prefix}_phases")
        st.dataframe(pd.DataFrame(directed["occupancy"]),hide_index=True)
        st.dataframe(pd.DataFrame([{k:v for k,v in edge.items() if k!="phase_counts"} for edge in directed["edges"]]),hide_index=True)
        st.caption("詳細計算の全試行を集計。率は遷移回数 / 出発井戸の観測滞在時間です。位相集中度は方向別で、少数イベントでは大きく出ることがあります。")
        if directed["unresolved_skips"]:
            st.warning("中間井戸を飛び越した観測があります。dtを小さくして遷移経路を再確認してください。")
    st.subheader("遷移と時間尺度")
    trans=detail["transitions"]
    cols=st.columns(3)
    cols[0].metric("半周期",format_number(1/(2*cfg.f)))
    cols[1].metric("完結した滞在時間の平均",format_number(trans["mean_residence"]))
    cols[2].metric("採用した滞在区間",trans["completed"])
    st.caption("core-to-core遷移。最初・最後の打切り区間を除外した標本平均で、平均初通過時間の不偏推定ではありません。")
    if detail["rates"]:
        rate_df=pd.DataFrame(detail["rates"])
        st.dataframe(rate_df[["source","target","delta_u","rate","mean_exit_time","barrier_over_D","edge_time_match_D"]].rename(
            columns={"source":"出発井戸","target":"到着井戸","delta_u":"障壁高 ΔU","rate":"Kramers率",
                     "mean_exit_time":"全方向への平均脱出時間","barrier_over_D":"ΔU / D","edge_time_match_D":"D_match（目安）"}),hide_index=True,use_container_width=True)
    else:
        st.info("この系にはKramers解析の対象となる井戸間障壁がありません。")


def show_results(experiment,prefix="main"):
    e=experiment["evidence"]
    validation=experiment.get("independent_validation")
    status=validation["status"] if validation else e["status"]
    st.subheader(status)
    cols=st.columns(3)
    cols[0].metric("範囲内の候補 D*",format_number(e.get("D_star",np.nan)))
    cols[1].metric("候補のS/N比",format_number(e.get("snr_star",np.nan)))
    cols[2].metric("試行数",experiment["simulation"]["trials"])
    db=st.toggle("dB表示",key=f"{prefix}_db")
    st.plotly_chart(sweep_figure(experiment["rows"],db),use_container_width=True,key=f"{prefix}_sweep")
    with st.expander("判定の根拠",expanded=True):
        evidence_table(experiment)
    with st.expander("S/N比と手法の適用範囲"):
        st.write("MCの95%CIは、同期成分ではjackknife-t、帯域差分では試行bootstrap。FP・Kramersの線は弱外力線形応答で、信頼区間ではありません。")
        st.write("局所雑音床が0ならSNRは未定義です。欠測点を0に置き換えてピークを作る処理は行いません。")
        st.write("Kramers曲線は高障壁・遅い外力を仮定します。x表示では井戸内運動を省いた近似です。")
        st.dataframe(pd.DataFrame(experiment["rows"]),hide_index=True,use_container_width=True)
    with st.expander("計算条件"):
        st.dataframe(pd.DataFrame(parameter_rows(experiment)),hide_index=True,use_container_width=True)
    if st.button("弱・候補・強の時系列を比較",key=f"{prefix}_three_traces"):
        cfg=replace(SimConfig(**experiment["simulation"]),trials=1)
        m=PotentialModel(ModelSpec(**experiment["model"]))
        trace_D=[min(r["D"] for r in experiment["rows"]), e.get("D_star",np.nan), max(r["D"] for r in experiment["rows"])]
        traces=[]
        for label,D in zip(["弱いノイズ","候補付近","強いノイズ"],trace_D):
            if not np.isfinite(D): continue
            try:
                t,x=simulate(m,cfg,float(D),stage=501)
                traces.append((label,float(D),t,x[0],m.observable(x[0],experiment["observable"])))
            except (ValueError,ArithmeticError) as error:
                safe_error(error)
        st.session_state[f"{prefix}_traces"]=traces
    for i,(label,D,t,x,y) in enumerate(st.session_state.get(f"{prefix}_traces",[])):
        cfg=SimConfig(**experiment["simulation"])
        st.markdown(f"**{label} · D = {D:.5g}**")
        st.plotly_chart(time_figure(t,x,y,cfg.A,cfg.f),use_container_width=True,key=f"{prefix}_trace_{i}")
    show_details(experiment,prefix)
    stored=st.session_state.get(f"{prefix}_detail")
    detail=stored[1] if stored else None
    st.download_button("結果一式を保存",experiment_zip(experiment,detail),"stochastic_resonance_results.zip",
                       "application/zip",key=f"{prefix}_download")
    st.download_button("S/N比 CSV",csv_bytes(experiment["rows"]),"noise_sweep.csv","text/csv",key=f"{prefix}_csv")


with st.sidebar:
    spec=model_controls()
    with st.expander("外力・観測",expanded=True):
        A=number("入力振幅 A",.06,.01,0.0,key="A")
        f=number("周波数 f",.02,.005,.001,key="f")
        observable=st.selectbox("観測量",["q","x"],format_func=lambda x:"井戸状態 q(x)" if x=="q" else "位置 x(t)",key="observable")
    with st.expander("モンテカルロ",expanded=True):
        scheme=st.selectbox("積分法",["heun","euler"],format_func=lambda x:{"heun":"Heun（既定）","euler":"Euler（比較用）"}[x],key="scheme")
        trials=st.number_input("試行数",min_value=3,max_value=200,value=30,key="trials")
        dt=number("時間刻み上限 dt",.02,.005,.0001,1.0,key="dt")
        cycles=st.number_input("観測周期数",min_value=8,max_value=400,value=48,step=8,key="cycles")
        burn=st.number_input("過渡除外周期数",min_value=0,max_value=200,value=8,key="burn")
        x0=number("初期位置 x₀",-1,.1,key="x0")
        seed=st.number_input("乱数 seed",min_value=0,max_value=2**31-1,value=2026,key="seed")
    with st.expander("ノイズの種類",expanded=True):
        noise_kind=st.selectbox("ノイズ",list(NOISE_LABELS),format_func=NOISE_LABELS.get,key="noise_kind")
        noise_tau=number("相関時間 τ",.5,.1,.001,key="noise_tau")
        if noise_kind!="white":
            st.caption("OU・二値の相関は (D/τ) exp(−|時間差|/τ)。相関の片側積分をDに揃えます。")
    with st.expander("ノイズ探索",expanded=True):
        dmin=number("D 最小",.04,.01,.0001,key="dmin")
        dmax=number("D 最大",.4,.05,.001,key="dmax")
        points=st.slider("粗探索点数",5,41,13,key="points")
        refine=st.slider("候補近傍の追加点数",0,15,5,key="refine")
        logarithmic=st.checkbox("対数刻み",value=False,key="log_grid")
    with st.expander("S/N比・FP格子"):
        segment_cycles=st.selectbox("Welch区間の周期数",[8,12,16,24],key="segment_cycles")
        half_width=st.selectbox("信号帯域の片側 bin 数",[1,2],key="half_width")
        aggregation=st.selectbox("S/N比の集約",["coherent","pooled_psd","trial_mean"],
                                  format_func=lambda x:{"coherent":"同期成分 / 局所雑音床","pooled_psd":"平均PSDから計算","trial_mean":"各試行SNRの平均"}[x],key="aggregation")
        nx=st.select_slider("FP格子点数",[121,241,481,961],value=241,key="nx")

try:
    model=PotentialModel(spec)
    config=SimConfig(A=A,f=f,dt=dt,cycles=int(cycles),burn_cycles=int(burn),trials=int(trials),
                     seed=int(seed),x0=x0,segment_cycles=int(segment_cycles),scheme=scheme,noise_kind=noise_kind,noise_tau=noise_tau)
    config.validate()
    spectral=SpectralConfig(half_width=half_width,aggregation=aggregation)
    if dmax<=dmin: raise ValueError("D最大はD最小より大きくしてください。")
    d_grid=np.geomspace(dmin,dmax,points) if logarithmic else np.linspace(dmin,dmax,points)
except ValueError as error:
    safe_error(error)
    st.stop()

experiment_tab,comparison_tab,validation_tab,theory_tab,chaos_tab=st.tabs(["実験","系・ノイズの比較","検証","数式・手法","カオス"])
with experiment_tab:
    phase=st.slider("ポテンシャルの位相 [周期]",0.0,1.0,0.0,.05,key="phase")
    st.plotly_chart(potential_figure(model,A,phase),use_container_width=True,key="potential")
    extrema=model.extrema()
    cols=st.columns(3)
    cols[0].metric("井戸（谷）",sum(e.kind=="minimum" for e in extrema))
    cols[1].metric("障壁（山）",sum(e.kind=="maximum" for e in extrema))
    cols[2].metric("実際の dt",format_number(config.actual_dt))
    if effective_observable(model,observable)!=observable:
        st.info("単安定のため観測量は x(t) を使います。")
    confirm_now=st.checkbox("独立試行・収束検証も実行",value=False,key="confirm_now")
    if st.button("解析する",type="primary",key="run_main"):
        try:
            exp=run_with_progress(spec,config,spectral,d_grid,observable,nx,refine)
            if confirm_now:
                progress=st.progress(0.0)
                exp["independent_validation"]=validate_experiment(exp,lambda n,total:progress.progress(n/total))
                progress.empty()
            st.session_state["experiment"]=exp
            st.session_state.pop("main_detail",None)
            st.session_state.pop("main_traces",None)
        except (ValueError,ArithmeticError) as error:
            safe_error(error)
    exp=st.session_state.get("experiment")
    if exp:
        changed=(exp["model"]!=asdict(spec) or exp["simulation"]!=asdict(config) or exp["spectral"]!=asdict(spectral)
                 or exp["nx"]!=nx or exp["observable"]!=effective_observable(model,observable)
                 or exp.get("scan_settings")!={"coarse_D":d_grid.tolist(),"refine":int(refine)}
                 or min(r["D"] for r in exp["rows"])!=float(d_grid.min()) or max(r["D"] for r in exp["rows"])!=float(d_grid.max()))
        if changed: st.info("下の結果は前回の条件です。変更を反映するには再計算してください。")
        show_results(exp)

with comparison_tab:
    mode=st.selectbox("比較軸",["4つの系","障壁の高さ","非対称性","障壁の数","障壁間の高さ差","ノイズ種類","相関時間"],key="comparison_mode")
    if mode=="4つの系":
        specs=[(LABELS[s.kind],s) for s in preset_specs()]
    elif mode=="障壁の高さ":
        specs=[(f"ΔU={h:g}",ModelSpec(a=4*h,b=4*h)) for h in (.125,.25,.5)]
    elif mode=="非対称性":
        specs=[(f"c={c:g}",ModelSpec(kind="asymmetric",c=c)) for c in (0,.04,.08,.12)]
    elif mode=="障壁の数":
        specs=[(f"山{n-1}・谷{n}",ModelSpec(kind="barriers",wells=n,c=0,height=.25)) for n in (1,2,3,4)]
    elif mode=="障壁間の高さ差":
        specs=[(f"高さ差係数={s:g}",ModelSpec(kind="barriers",wells=3,c=0,height=.25,skew=s)) for s in (-.7,0,.7)]
    elif mode=="ノイズ種類":
        specs=[(NOISE_LABELS[kind],spec) for kind in NOISE_LABELS]
    else:
        specs=[(f"τ={tau:g}",spec) for tau in (.1,.5,2.)]
    cases=[]
    for i,(label,s) in enumerate(specs):
        cc=config
        if mode=="ノイズ種類": cc=replace(config,noise_kind=list(NOISE_LABELS)[i])
        if mode=="相関時間": cc=replace(config,noise_kind="ou" if config.noise_kind=="white" else config.noise_kind,noise_tau=(.1,.5,2.)[i])
        cases.append((label,s,cc))
    with st.expander("比較条件"):
        table=[]
        for label,s,cc in cases:
            m=PotentialModel(s)
            es=m.extrema(); barriers=m.barriers()
            table.append({"系":label,"谷":sum(e.kind=="minimum" for e in es),"山":sum(e.kind=="maximum" for e in es),
                          "最小ΔU":min((e.delta_u for e in barriers),default=np.nan),
                          "最大ΔU":max((e.delta_u for e in barriers),default=np.nan),
                          "観測量":effective_observable(m,observable),"ノイズ":NOISE_LABELS[cc.noise_kind],"τ":cc.noise_tau})
        st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
        st.write("外力・ノイズ範囲・観測時間・帯域は共通です。単安定のみqを定義できないためxを使います。曲線の絶対値より、各系での非単調性を比較します。")
        st.write("障壁数の系列は平均井戸間隔を固定したChebyshev多項式です。局所曲率まで固定されるわけではありません。")
    current_comparison_settings={"mode":mode,"simulation":asdict(config),"spectral":asdict(spectral),
                                 "coarse_D":d_grid.tolist(),"nx":nx,"refine":refine,"observable":observable}
    if st.button("比較を一括実行",type="primary",key="run_comparison"):
        st.session_state["comparison_settings"]=current_comparison_settings
        comparisons=[]
        try:
            for label,s,cc in cases:
                with st.spinner(f"{label} を計算中"):
                    result=run_with_progress(s,cc,spectral,d_grid,observable,nx,refine)
                comparisons.append((label,result))
            st.session_state["comparisons"]=comparisons
        except (ValueError,ArithmeticError) as error:
            safe_error(error)
            if comparisons: st.session_state["comparisons"]=comparisons
    comparisons=st.session_state.get("comparisons")
    if comparisons:
        if st.session_state.get("comparison_settings")!=current_comparison_settings:
            st.info("比較グラフは前回の条件です。変更を反映するには一括実行してください。")
        method=st.selectbox("表示する手法",list(METHOD_LABELS),format_func=METHOD_LABELS.get,key="comparison_method")
        st.plotly_chart(comparison_figure(comparisons,method),use_container_width=True,key="comparison_plot")
        st.dataframe(pd.DataFrame([{"系":label,"観測量":e["observable"],"候補D*":e["evidence"].get("D_star",np.nan),
                                    "判定":e["evidence"]["status"]} for label,e in comparisons]),hide_index=True,use_container_width=True)
        selected_label=st.selectbox("検証する比較結果",[label for label,_ in comparisons],key="comparison_select")
        if st.button("選んだ系を検証対象にする",key="use_comparison"):
            st.session_state["experiment"]=next(e for label,e in comparisons if label==selected_label)
            st.session_state.pop("main_detail",None)
            st.session_state.pop("main_traces",None)
            st.success("「検証」タブで独立試行を実行できます。")
        buffer=io.BytesIO()
        with ZipFile(buffer,"w",ZIP_DEFLATED) as archive:
            for i,(label,e) in enumerate(comparisons):
                archive.writestr(f"system_{i+1}.zip",experiment_zip(e))
            archive.writestr("systems.csv",csv_bytes([{"index":i+1,"name":label} for i,(label,e) in enumerate(comparisons)]))
        st.download_button("比較結果を保存",buffer.getvalue(),"system_comparison.zip","application/zip",key="comparison_download")

with validation_tab:
    exp=st.session_state.get("experiment")
    if exp is None:
        st.info("実験または系の比較を実行してください。")
    else:
        st.subheader(PotentialModel(ModelSpec(**exp["model"])).name)
        st.write(f"候補 D* = {format_number(exp['evidence'].get('D_star',np.nan))}")
        if st.button("独立試行と収束を検証",type="primary",key="run_validation"):
            progress=st.progress(0.0)
            try:
                exp["independent_validation"]=validate_experiment(exp,lambda n,total:progress.progress(n/total))
                st.session_state["experiment"]=exp
            except (ValueError,ArithmeticError) as error:
                safe_error(error)
            finally:
                progress.empty()
        validation=exp.get("independent_validation")
        if validation:
            st.subheader(validation["status"])
            if validation.get("checks"):
                st.dataframe(pd.DataFrame(validation["checks"]).rename(columns={"check":"比較","difference":"SNR差",
                            "lower_bound":"補正後の片側下限","one_sided_confidence":"片側信頼水準","result":"結果"}),hide_index=True,use_container_width=True)
                st.write("候補の選択と確認で乱数を分離。1モデル内の3つの片側比較の全体α=0.05をBonferroni補正しています。")
            if validation.get("convergence"):
                st.subheader("数値条件を変えた比較")
                st.dataframe(pd.DataFrame(validation["convergence"]),hide_index=True,use_container_width=True)
                st.caption("MCは差の95%区間が±max(0.002, SNRの10%)内ならPASS。区間が広い場合はINCONCLUSIVEです。FP線形応答は変化5%以内を確認します。")
            if validation.get("controls"):
                controls=validation["controls"]
                st.metric("D=0でのcore-to-core遷移",controls["D_zero_transitions"])
                if controls["D_zero_transitions"]: st.warning("ノイズなしでも遷移しています。閾値以下のノイズ誘起遷移とは区別して解釈してください。")
            st.write(validation["conclusion"])
            st.download_button("検証込みの結果を保存",experiment_zip(exp),"validated_experiment.zip","application/zip",key="validation_download")
        else:
            evidence_table(exp)

        st.subheader("指定Dで精度を確認")
        precision_D=st.selectbox("精度確認するD",sorted({r["D"] for r in exp["rows"]}),key="precision_D")
        cfg=SimConfig(**exp["simulation"])
        m=PotentialModel(ModelSpec(**exp["model"]))
        sc=SpectralConfig(**exp["spectral"])
        if st.button("結合経路で刻み幅を比較",key="run_precision"):
            try:
                with st.spinner("同じノイズ経路で3段階を比較中"):
                    exp["precision_study"]={"D":precision_D,"checks":timestep_study(m,cfg,sc,precision_D,exp["observable"])}
            except (ValueError,ArithmeticError) as error:
                safe_error(error)
        if exp.get("precision_study"):
            st.write(f"計算済み D = {exp['precision_study']['D']:.5g}")
            st.dataframe(pd.DataFrame(exp["precision_study"]["checks"]),hide_index=True)
        if cfg.noise_kind=="white":
            with st.expander("有限振幅のFP S/N比"):
                fp_steps=st.select_slider("1周期の時間分割",[64,128,256,512,1024],value=256,key="nonlinear_steps")
                fp_lags=st.select_slider("相関の観測周期数",[4,8,16,32,64],value=8,key="nonlinear_lags")
                if st.button("時間相関からFP S/N比を計算",key="run_nonlinear"):
                    try:
                        with st.spinner("周期定常相関と分解能を確認中"):
                            bands=make_bands(cfg.actual_dt,cfg.nperseg,cfg.f,sc)
                            exp["nonlinear_fp"]={"selected_D":precision_D,"result":resolution_study(m,precision_D,cfg.A,bands,exp["observable"],make_grid(tuple(exp["domain"]),exp["nx"]),steps=fp_steps,lag_cycles=fp_lags)}
                    except (ValueError,ArithmeticError) as error:
                        safe_error(error)
                if exp.get("nonlinear_fp"):
                    r=exp["nonlinear_fp"]["result"]
                    st.write(f"計算済み D = {exp['nonlinear_fp']['selected_D']:.5g}")
                    st.metric("FP S/N比（有限振幅）",format_number(r["snr"]))
                    st.dataframe(pd.DataFrame(r["resolution_checks"]),hide_index=True)
                    if not r["valid"] or not r["resolution_supported"]:
                        st.warning("周期収束・相関の打切り・境界・分解能の確認が完了していません。")
                    st.dataframe(pd.DataFrame([{k:r[k] for k in ["periodic_residual","tail_ratio","spectrum_truncation_change","boundary_mass","dt"]}]),hide_index=True)
        st.download_button("精度確認を含めて保存",experiment_zip(exp),"precision_results.zip","application/zip",key="precision_download")

with theory_tab:
    st.dataframe(pd.DataFrame([
        {"手法":"モンテカルロ","計算対象":"確率軌道","SNR":"Welch帯域推定","誤差・制約":"標本誤差・時間離散化"},
        {"手法":"Kramers","計算対象":"隣接井戸の遷移率・人口動態","SNR":"弱外力の多状態線形応答","誤差・制約":"高障壁・遅い外力・井戸内運動の省略"},
        {"手法":"Fokker–Planck","計算対象":"確率密度・平衡相関","SNR":"弱外力線形応答／密度は周期外力の直接解","誤差・制約":"空間・時間離散化・有限領域・線形応答近似"}
    ]),hide_index=True,use_container_width=True)
    theory_D=number("数値代入する D",.1,.01,.0001,key="theory_D")
    st.subheader("現在の入力値")
    render_equations(st,model,config,spectral,theory_D)

with chaos_tab:
    st.subheader("慣性を加えたDuffingモデル")
    st.latex(r"\ddot x+\gamma\dot x=-U'(x)+A\cos(2\pi f t)")
    st.caption("ノイズなしの別モデルです。有限時間Lyapunov指数とPoincaré断面を調べ、刻み幅半減と比較します。元の過減衰系の確率共鳴とは分けて評価します。")
    ca=number("Duffing入力振幅",.3,.05,0.,key="chaos_A")
    cf=number("Duffing周波数",1.2/(2*np.pi),.01,.001,key="chaos_f")
    damping=number("減衰 γ",.2,.05,.001,key="chaos_damping")
    cdt=number("Duffing時間刻み",.02,.005,.001,key="chaos_dt")
    ccycles=st.number_input("Duffing観測周期",min_value=4,max_value=300,value=100,key="chaos_cycles")
    cburn=st.number_input("Duffing過渡除外周期",min_value=0,max_value=200,value=40,key="chaos_burn")
    chaos_settings={"model":asdict(spec),"config":asdict(DuffingConfig(ca,cf,damping,cdt,int(ccycles),int(cburn)))}
    if st.button("カオス指標を計算",key="run_chaos"):
        try:
            with st.spinner("軌道・変分方程式を計算中"):
                st.session_state["chaos_result"]={"settings":chaos_settings,"result":duffing_refinement(model,DuffingConfig(**chaos_settings["config"]))}
        except (ValueError,ArithmeticError) as error:
            safe_error(error)
    if st.session_state.get("chaos_result"):
        stored=st.session_state["chaos_result"]
        if stored["settings"]!=chaos_settings: st.info("表示中は前回の計算条件です。")
        r=stored["result"]; fine=r["fine"]
        st.write(fine["classification"])
        st.metric("有限時間Lyapunov指数",format_number(fine["finite_time_exponent"]))
        st.plotly_chart(chaos_figure(fine),use_container_width=True,key="chaos_plot")
        st.write(f"刻み半減による指数差：{r['exponent_difference']:.5g}")
        if not r["precision_supported"]: st.warning("刻み幅による指数の変動が大きいため、追加検証が必要です。")
        from exporting import json_safe
        st.download_button("カオス解析を保存",json.dumps(json_safe(stored),ensure_ascii=False,allow_nan=False),"duffing_diagnostic.json","application/json",key="chaos_download")

