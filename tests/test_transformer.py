"""
测试联动：transformer/fixture.json
校验两部分：
1. 正弦位置编码：PE[pos, 2i] = sin(pos/10000^(2i/d))，PE[pos, 2i+1] = cos(...)
2. Causal mask：seq_len × seq_len 的下三角布尔矩阵
"""
import numpy as np
from conftest import load_fixture


def _positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angles = pos / np.power(10000.0, (i // 2 * 2) / d_model)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


def _causal_mask(seq_len: int) -> np.ndarray:
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def test_positional_encoding():
    fixture = load_fixture("transformer")
    pe = _positional_encoding(fixture["sequenceLength"], fixture["dModel"])
    expected = np.array(fixture["positionalEncoding"], dtype=float)
    assert np.allclose(pe, expected, atol=fixture["tolerance"]), (
        f"max deviation: {np.abs(pe - expected).max():.2e}"
    )


def test_causal_mask():
    fixture = load_fixture("transformer")
    mask = _causal_mask(fixture["sequenceLength"])
    expected = np.array(fixture["causalMask"], dtype=bool)
    assert np.array_equal(mask, expected), (
        f"causal mask mismatch:\n{mask}\nvs\n{expected}"
    )


def test_causal_mask_is_lower_triangular():
    """下三角性质：位置 i 只能看到 0..i，不能看到 i+1 以后。"""
    fixture = load_fixture("transformer")
    mask = np.array(fixture["causalMask"])
    n = fixture["sequenceLength"]
    for i in range(n):
        assert all(mask[i, : i + 1]), f"position {i} should see 0..{i}"
        assert not any(mask[i, i + 1 :]), f"position {i} should not see {i+1}..{n-1}"
