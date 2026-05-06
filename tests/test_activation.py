"""
测试联动：activation/fixture.json
在 fixture 指定的 x 处校验 Sigmoid 函数值和导数值；
同时参数化验证 samplePoints 上 Sigmoid 解析导数与数值微分的一致性。
"""
import math
import pytest
from conftest import load_fixture


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _sigmoid_grad(x: float) -> float:
    s = _sigmoid(x)
    return s * (1.0 - s)


def _numeric_grad(fn, x: float, h: float = 1e-4) -> float:
    return (fn(x + h) - fn(x - h)) / (2 * h)


def test_sigmoid_gradient_check():
    fixture = load_fixture("activation")
    gc = fixture["gradientCheck"]
    tol = gc["tolerance"]

    assert abs(_sigmoid(gc["x"]) - gc["sigmoid"]) <= tol
    assert abs(_sigmoid_grad(gc["x"]) - gc["sigmoidGradient"]) <= tol


@pytest.mark.parametrize("x", load_fixture("activation")["samplePoints"])
def test_sigmoid_analytic_vs_numeric(x):
    """解析导数与数值微分在样本点上误差 < 1e-5。"""
    analytic = _sigmoid_grad(x)
    numeric = _numeric_grad(_sigmoid, x)
    assert abs(analytic - numeric) < 1e-5, (
        f"x={x}: analytic={analytic:.8f}, numeric={numeric:.8f}"
    )
