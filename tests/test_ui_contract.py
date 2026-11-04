"""Headless control-flow checks using a strict UI stand-in, not a browser test."""
from __future__ import annotations
import runpy
import sys
import types
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class StopRender(Exception):
    pass


class UI:
    def __init__(self,inputs=None,buttons=()):
        self.session_state={}
        self.inputs=inputs or {}
        self.buttons=set(buttons)
        self.errors=[]
        self.figures=[]
        self.downloads=[]
        self.sidebar=self

    def __enter__(self):return self
    def __exit__(self,*args):return False
    def set_page_config(self,**kwargs):pass
    def title(self,*args,**kwargs):pass
    def subheader(self,*args,**kwargs):pass
    def markdown(self,*args,**kwargs):pass
    def write(self,*args,**kwargs):pass
    def caption(self,*args,**kwargs):pass
    def latex(self,*args,**kwargs):pass
    def metric(self,*args,**kwargs):pass
    def dataframe(self,*args,**kwargs):pass
    def info(self,*args,**kwargs):pass
    def warning(self,*args,**kwargs):pass
    def success(self,*args,**kwargs):pass
    def error(self,text,*args,**kwargs):self.errors.append(text)
    def stop(self):raise StopRender()
    def columns(self,n,*args,**kwargs):return [self]*n
    def tabs(self,labels):return [self for _ in labels]
    def expander(self,*args,**kwargs):return self
    def spinner(self,*args,**kwargs):return self
    def progress(self,*args,**kwargs):return self
    def empty(self):pass
    def _value(self,key,default):return self.inputs.get(key,default)
    def number_input(self,label,value=None,min_value=None,max_value=None,key=None,**kwargs):
        v=self._value(key,value)
        assert min_value is None or v>=min_value
        assert max_value is None or v<=max_value
        return v
    def selectbox(self,label,options,index=0,key=None,**kwargs):
        v=self._value(key,options[index]);assert v in options;return v
    def select_slider(self,label,options,value=None,key=None,**kwargs):
        v=self._value(key,options[0] if value is None else value);assert v in options;return v
    def slider(self,label,min_value,max_value,value=None,*args,key=None,**kwargs):
        v=self._value(key,min_value if value is None else value)
        assert min_value<=v<=max_value
        return v
    def checkbox(self,label,value=False,key=None,**kwargs):return self._value(key,value)
    def toggle(self,label,value=False,key=None,**kwargs):return self._value(key,value)
    def button(self,label,key=None,**kwargs):
        if key in self.buttons:
            self.buttons.remove(key);return True
        return False
    def plotly_chart(self,figure,**kwargs):
        assert len(figure.to_json())>10
        self.figures.append(figure)
    def download_button(self,label,data,*args,**kwargs):
        assert isinstance(data,(bytes,str)) and len(data)>0
        self.downloads.append(label)
        return False


def execute(monkeypatch,inputs=None,buttons=()):
    defaults={'trials':4,'cycles':8,'burn':1,'dt':.05,'points':5,'refine':0,'nx':121}
    defaults.update(inputs or {})
    ui=UI(defaults,buttons)
    module=types.ModuleType('streamlit')
    for name in dir(ui):
        if not name.startswith('_'):setattr(module,name,getattr(ui,name))
    monkeypatch.setitem(sys.modules,'streamlit',module)
    try:runpy.run_path(str(ROOT/'app.py'),run_name='__main__')
    except StopRender:pass
    return ui


@pytest.mark.parametrize('kind',['symmetric','asymmetric','multi','mono','barriers'])
def test_initial_control_paths(monkeypatch,kind):
    ui=execute(monkeypatch,{'model_kind':kind})
    assert not ui.errors and ui.figures


def test_calculation_downloads_and_three_traces(monkeypatch):
    ui=execute(monkeypatch,buttons=['run_main','main_three_traces'])
    assert not ui.errors
    assert len(ui.session_state['experiment']['rows'])==15
    assert len(ui.session_state['main_traces'])==3
    assert len(ui.downloads)>=2


def test_comparison_control_path(monkeypatch):
    ui=execute(monkeypatch,{'comparison_mode':'障壁の数'},['run_comparison'])
    assert not ui.errors
    assert len(ui.session_state['comparisons'])==4


def test_invalid_input_stops_without_execution(monkeypatch):
    ui=execute(monkeypatch,{'dmax':.01},['run_main'])
    assert ui.errors and 'experiment' not in ui.session_state


def test_detail_probability_and_exports(monkeypatch):
    ui=execute(monkeypatch,{'main_dt_fp':.2,'main_fp_cycles':20},['run_main','main_detail_run'])
    assert not ui.errors
    assert 'main_detail' in ui.session_state
    assert len(ui.figures)>=6
