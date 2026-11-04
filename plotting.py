"""Reusable interactive plots; visual downsampling never changes analysis data."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models import PotentialModel


METHOD_LABELS = {"MC": "MC", "FP": "FP（線形応答）", "Kramers": "Kramers（線形応答）"}


def finish(fig, x_title, y_title, height=410):
    fig.update_layout(height=height, margin=dict(l=30,r=25,t=30,b=45),
                      xaxis_title=x_title, yaxis_title=y_title,
                      legend=dict(orientation="h", y=-.22), hovermode="closest")
    return fig


def potential_figure(model: PotentialModel, A=0.0, phase=0.0):
    extrema = model.extrema()
    minimum = [e for e in extrema if e.kind in ("minimum","flat")]
    if len(minimum) > 1:
        lo, hi = minimum[0].x, minimum[-1].x
        pad = max(.4, .15*(hi-lo))
        lo, hi = lo-pad, hi+pad
    else:
        center = minimum[0].x if minimum else 0
        lo, hi = center-1.5, center+1.5
    x = np.linspace(lo,hi,1001)
    force = A*np.cos(2*np.pi*phase)
    fig = go.Figure()
    fig.add_scatter(x=x,y=model.potential(x),name="U(x)")
    fig.add_scatter(x=x,y=model.potential(x)-force*x,name="瞬時ポテンシャル",line=dict(dash="dash"))
    for kind,label,symbol in [("minimum","谷（静的）","circle"),("maximum","山（静的）","x")]:
        es = [e for e in extrema if e.kind==kind]
        fig.add_scatter(x=[e.x for e in es],y=[e.u for e in es],mode="markers",name=label,
                        marker=dict(symbol=symbol,size=10))
    tops = [e.u for e in extrema if e.kind=="maximum"]
    if tops:
        bottom = min(e.u for e in extrema)
        scale = max(max(tops)-bottom,.05)
        fig.update_yaxes(range=[bottom-.6*scale-A*max(abs(lo),abs(hi)), bottom+3*scale])
    return finish(fig,"x","U(x)",350)


def sweep_figure(rows, db=False):
    fig=go.Figure()
    for method,dash in [("MC","solid"),("FP","dash"),("Kramers","dot")]:
        subset=sorted([r for r in rows if r["method"]==method],key=lambda r:r["D"])
        if not subset:
            continue
        x=np.array([r["D"] for r in subset])
        y=np.array([r.get("snr",np.nan) for r in subset])
        lo=np.array([r.get("ci_low",np.nan) for r in subset])
        hi=np.array([r.get("ci_high",np.nan) for r in subset])
        if db:
            with np.errstate(divide="ignore",invalid="ignore"):
                y,lo,hi=[np.where(z>0,10*np.log10(z),np.nan) for z in (y,lo,hi)]
        error=dict(type="data",symmetric=False,array=hi-y,arrayminus=y-lo,visible=method=="MC")
        fig.add_scatter(x=x,y=y,name=METHOD_LABELS[method],mode="lines+markers",
                        line=dict(dash=dash),error_y=error,connectgaps=False)
    return finish(fig,"ノイズ強度 D","S/N比 [dB]" if db else "S/N比（線形）")


def comparison_figure(experiments, method="MC"):
    fig=go.Figure()
    for name, exp in experiments:
        rows=sorted([r for r in exp["rows"] if r["method"]==method],key=lambda r:r["D"])
        y=np.array([r.get("snr",np.nan) for r in rows])
        lo=np.array([r.get("ci_low",np.nan) for r in rows])
        hi=np.array([r.get("ci_high",np.nan) for r in rows])
        obs=exp["observable"]
        fig.add_scatter(x=[r["D"] for r in rows],y=y,name=f"{name} · {obs}",mode="lines+markers",
                        error_y=dict(type="data",symmetric=False,array=hi-y,arrayminus=y-lo,visible=method=="MC"))
    return finish(fig,"ノイズ強度 D","S/N比（線形）",470)


def time_figure(t,x,y,A,f,periods=6):
    keep=t>=t[-1]-periods/f
    tt=t[keep]
    # Min/max envelope points preserve spikes better than uniform decimation.
    ids=np.flatnonzero(keep)
    if len(ids)>5000:
        chunks=np.array_split(ids,2400)
        ids=np.unique(np.array([[c[np.argmin(x[c])],c[np.argmax(x[c])]] for c in chunks]).ravel())
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_scatter(x=t[ids],y=x[ids],name="x(t)",secondary_y=False)
    if not np.allclose(x,y):
        fig.add_scatter(x=t[ids],y=y[ids],name="q(x)",line=dict(dash="dot"),secondary_y=False)
    input_times=np.linspace(tt[0],tt[-1],2000)
    fig.add_scatter(x=input_times,y=A*np.cos(2*np.pi*f*input_times),name="周期入力",secondary_y=True)
    fig.update_yaxes(title_text="入力",secondary_y=True)
    return finish(fig,"時間 t","出力",350)


def psd_figure(frequency,psd,bands):
    mask=(frequency>0)&(frequency<=bands.f0*4)
    fig=go.Figure()
    fig.add_scatter(x=frequency[mask],y=np.maximum(psd[mask],np.finfo(float).tiny),name="Welch PSD")
    for ids,name,symbol in [(bands.signal,"信号帯域","circle"),(bands.noise,"雑音床の推定帯域","diamond")]:
        fig.add_scatter(x=frequency[ids],y=np.maximum(psd[ids],np.finfo(float).tiny),mode="markers",name=name,
                        marker=dict(symbol=symbol,size=8))
    fig.add_vline(x=bands.f0,line_dash="dash")
    fig.update_yaxes(type="log")
    return finish(fig,"周波数 ν","PSD [出力² / Hz]",350)


def density_figure(result):
    grid=result["grid"]
    dx=grid[1]-grid[0]
    fig=go.Figure(go.Heatmap(x=result["time"],y=grid,z=result["probability"].T/dx,
                            colorbar=dict(title="p(x,t)")))
    return finish(fig,"周期内の時間 t","x",380)


def mean_figure(fp_result,k_result,A,f):
    fig=go.Figure()
    fig.add_scatter(x=fp_result["time"],y=fp_result["mean"],name="FP 周期平均応答")
    if k_result:
        fig.add_scatter(x=k_result["time"],y=k_result["mean"],name="Kramers 状態平均",line=dict(dash="dash"))
    return finish(fig,"周期内の時間 t","期待値 〈g(X)〉",320)
