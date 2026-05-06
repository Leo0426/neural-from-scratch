"""
测试联动：gradient_descent/fixture.json
损失函数 f(x,y) = x^2 + 0.2*y^2 + 0.5*sin(3x)*cos(2y)
在 fixture 指定的检查点处计算解析梯度，与 expectedGradient 对比。
"""
import math
from conftest import load_fixture


def _grad(theta: list[float]) -> list[float]:
    x, y = theta
    return [
        2 * x + 1.5 * math.cos(3 * x) * math.cos(2 * y),
        0.4 * y - math.sin(3 * x) * math.sin(2 * y),
    ]


def test_gradient_check():
    fixture = load_fixture("gradient_descent")
    gc = fixture["gradientCheck"]
    computed = _grad(gc["theta"])
    expected = gc["expectedGradient"]
    tol = gc["tolerance"]
    for i, (c, e) in enumerate(zip(computed, expected)):
        assert abs(c - e) <= tol, (
            f"gradient[{i}] mismatch: computed={c:.8f}, expected={e:.8f}, tol={tol}"
        )
