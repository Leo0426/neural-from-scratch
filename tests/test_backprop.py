"""
测试联动：backprop/fixture.json
用与 notebook 相同的随机种子（default_rng(1)）构造两层网络，
手写前向 + 反向传播，校验 MSE 损失值和各参数梯度 L2 范数。
"""
import numpy as np
from conftest import load_fixture


def _build_network():
    """与 notebook 04 完全相同的网络初始化方式。"""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 3))
    y = rng.normal(size=(5, 2))
    params = {
        "W1": rng.normal(scale=0.2, size=(3, 4)),
        "b1": np.zeros(4),
        "W2": rng.normal(scale=0.2, size=(4, 2)),
        "b2": np.zeros(2),
    }
    return X, y, params


def _forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    return z2, {"a1": a1, "z2": z2}


def _backward(params, X, y, cache):
    n = y.size
    dz2 = 2 * (cache["z2"] - y) / n
    dW2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (1 - cache["a1"] ** 2)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def test_loss_value():
    fixture = load_fixture("backprop")
    X, y, params = _build_network()
    pred, _ = _forward(params, X)
    loss = np.mean((pred - y) ** 2)
    assert abs(loss - fixture["loss"]) < fixture["tolerance"], (
        f"loss mismatch: {loss:.10f} vs {fixture['loss']:.10f}"
    )


def test_gradient_norms():
    fixture = load_fixture("backprop")
    X, y, params = _build_network()
    pred, cache = _forward(params, X)
    grads = _backward(params, X, y, cache)

    expected_norms = fixture["expectedGradientNorms"]
    tol = fixture["tolerance"]
    for name, g in grads.items():
        norm = float(np.linalg.norm(g))
        expected_norm = expected_norms[name]["norm"]
        assert abs(norm - expected_norm) < tol, (
            f"{name}: norm={norm:.8f}, expected={expected_norm:.8f}, tol={tol}"
        )
