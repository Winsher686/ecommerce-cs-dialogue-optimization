"""灰度路由单元测试。"""

from src.inference.router import GrayRouter


def test_ratio_bounds():
    r = GrayRouter(0.0)
    assert all(r.route() == "base" for _ in range(20))
    r.set_ratio(1.0)
    assert all(r.route() == "finetuned" for _ in range(20))


def test_rollback():
    r = GrayRouter(1.0)
    r.rollback()
    assert r.route() == "base"
    assert r.status()["rollback"] is True
    r.recover()
    assert r.route() == "finetuned"
