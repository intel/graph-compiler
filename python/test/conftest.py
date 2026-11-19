import pytest


def pytest_addoption(parser):
    group = parser.getgroup("graph-compiler")
    group.addoption("--gc-dump-input", action="store_true", help="Print input MLIR")
    group.addoption("--gc-dump", action="store_true", help="Dump compiler pipeline IR")


@pytest.fixture(autouse=True)
def disable_cache(monkeypatch):
    monkeypatch.setenv("GC_FILE_CACHE_ON", "0")


@pytest.fixture
def enable_cache(disable_cache, monkeypatch):
    monkeypatch.setenv("GC_FILE_CACHE_ON", "1")