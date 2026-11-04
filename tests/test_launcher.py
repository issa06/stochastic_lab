"""Launcher behavior without touching the host's Python installation."""
from pathlib import Path
import subprocess
import sys
import pytest
import launcher


def test_launcher_dependency_check():
    r=subprocess.run([sys.executable,str(Path(launcher.__file__)),'--check'],capture_output=True,text=True)
    assert r.returncode==0 and 'PASS' in r.stdout


def test_launcher_rejects_invalid_port():
    r=subprocess.run([sys.executable,str(Path(launcher.__file__)),'--port','80'],capture_output=True,text=True)
    assert r.returncode!=0 and '1024' in r.stderr


def test_launcher_binds_only_to_localhost(monkeypatch):
    calls=[]
    monkeypatch.setattr(sys,'argv',['launcher.py','--port','8502'])
    monkeypatch.setattr(launcher.os,'execv',lambda path,args:calls.append((path,args)))
    launcher.main()
    assert '--server.address=127.0.0.1' in calls[0][1]
    assert '--server.port=8502' in calls[0][1]


def test_failed_bootstrap_suppresses_package_manager_output(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(launcher,'ROOT',tmp_path)
    target=tmp_path/'.venv-macos'/'bin'
    target.mkdir(parents=True)
    (target/'python').touch()
    monkeypatch.setattr(sys,'argv',['launcher.py','--bootstrap'])
    def fail(*args,**kwargs):
        assert kwargs['capture_output']
        return subprocess.CompletedProcess(args[0],1,'private-diagnostic-marker','private-diagnostic-marker')
    monkeypatch.setattr(launcher.subprocess,'run',fail)
    with pytest.raises(SystemExit) as error: launcher.main()
    assert 'private-diagnostic-marker' not in str(error.value)
    assert 'private-diagnostic-marker' not in str(capsys.readouterr())
