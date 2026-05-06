"""
测试联动：attention/fixture.json
从 fixture 读取 Q、K、V，计算 softmax(Q·Kᵀ / √d_k)，
与 expectedWeights 逐元素对比；同时校验权重行和为 1。
"""
import numpy as np
from conftest import load_fixture


def _attention_weights(Q: list, K: list) -> np.ndarray:
    Q = np.array(Q, dtype=float)
    K = np.array(K, dtype=float)
    dk = K.shape[1]
    scores = Q @ K.T / np.sqrt(dk)
    scores -= scores.max(axis=1, keepdims=True)   # 数值稳定
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def test_attention_weights_match_expected():
    fixture = load_fixture("attention")
    weights = _attention_weights(fixture["q"], fixture["k"])
    expected = np.array(fixture["expectedWeights"], dtype=float)
    assert np.allclose(weights, expected, atol=fixture["tolerance"]), (
        f"max deviation: {np.abs(weights - expected).max():.2e}"
    )


def test_attention_weights_sum_to_one():
    fixture = load_fixture("attention")
    weights = _attention_weights(fixture["q"], fixture["k"])
    row_sums = weights.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), (
        f"row sums not 1: {row_sums}"
    )
