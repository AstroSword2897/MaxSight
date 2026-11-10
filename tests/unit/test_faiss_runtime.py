from __future__ import annotations

from ml.retrieval.indexing.faiss_runtime import configure_environment, configure_faiss


class FakeFaiss:
    def __init__(self) -> None:
        self.threads: int | None = None

    def omp_set_num_threads(self, threads: int) -> None:
        self.threads = threads


def test_macos_runtime_limits_environment_and_faiss_threads(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    fake = FakeFaiss()

    configure_environment("darwin")
    configure_faiss(fake, "darwin")

    assert fake.threads == 1
    assert __import__("os").environ["OMP_NUM_THREADS"] == "1"


def test_non_macos_runtime_is_unchanged(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    fake = FakeFaiss()

    configure_environment("linux")
    configure_faiss(fake, "linux")

    assert fake.threads is None
    assert "OMP_NUM_THREADS" not in __import__("os").environ
