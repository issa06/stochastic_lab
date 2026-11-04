"""UI tests; skipped explicitly when the optional test runtime is unavailable."""
from pathlib import Path
import pytest
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'app.py'


def test_initial_render_and_equations():
    app=AppTest.from_file(str(APP),default_timeout=60).run()
    assert not app.exception
    assert len(app.tabs)==5
    assert len(app.latex)>10


def test_calculate_and_invalid_input_recovery():
    app=AppTest.from_file(str(APP),default_timeout=60).run()
    app.number_input(key='trials').set_value(4)
    app.number_input(key='cycles').set_value(8)
    app.number_input(key='burn').set_value(1)
    app.number_input(key='dt').set_value(.05)
    app.slider(key='points').set_value(5)
    app.slider(key='refine').set_value(0)
    app.button(key='run_main').click().run()
    assert not app.exception
    assert len(app.session_state['experiment']['rows'])==15
    app.number_input(key='dmax').set_value(.01).run()
    assert app.error and not app.exception
    app.number_input(key='dmax').set_value(.4).run()
    assert not app.exception


def test_monostable_and_barrier_family():
    app=AppTest.from_file(str(APP),default_timeout=60).run()
    app.selectbox(key='model_kind').select('mono').run()
    assert not app.exception
    app.selectbox(key='model_kind').select('barriers').run()
    app.slider(key='wells').set_value(4).run()
    assert not app.exception


def test_colored_noise_detail_export_and_precision_controls():
    app=AppTest.from_file(str(APP),default_timeout=120).run()
    app.selectbox(key='noise_kind').select('ou')
    app.number_input(key='trials').set_value(4)
    app.number_input(key='cycles').set_value(8)
    app.number_input(key='burn').set_value(1)
    app.number_input(key='dt').set_value(.05)
    app.slider(key='points').set_value(5)
    app.slider(key='refine').set_value(0)
    app.button(key='run_main').click().run()
    assert not app.exception
    rows=app.session_state['experiment']['rows']
    assert all(r['status']=='適用外' for r in rows if r['method']!='MC')
    app.button(key='main_detail_run').click().run()
    assert not app.exception
    assert app.session_state['main_detail'][1]['fp_periodic'] is None
    app.button(key='run_precision').click().run()
    assert not app.exception
    assert len(app.session_state['experiment']['precision_study']['checks'])==2

