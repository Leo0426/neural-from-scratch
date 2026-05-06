"""
测试联动：cnn/fixture.json
手写 2D 卷积（支持 padding 和 stride），将输出与 fixture 中
PyTorch conv2d 预先计算的 expectedOutput 逐元素对比。
"""
import numpy as np
from conftest import load_fixture


def _conv2d(input_arr: list, kernel: list, padding: int, stride: int) -> np.ndarray:
    inp = np.array(input_arr, dtype=float)
    k = np.array(kernel, dtype=float)
    if padding > 0:
        inp = np.pad(inp, padding)
    kH, kW = k.shape
    H, W = inp.shape
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1
    out = np.zeros((out_h, out_w))
    for r in range(out_h):
        for c in range(out_w):
            out[r, c] = np.sum(
                inp[r * stride : r * stride + kH, c * stride : c * stride + kW] * k
            )
    return out


def test_conv_output_shape():
    fixture = load_fixture("cnn")
    out = _conv2d(fixture["input"], fixture["kernel"], fixture["padding"], fixture["stride"])
    expected = np.array(fixture["expectedOutput"])
    assert out.shape == expected.shape, f"shape mismatch: {out.shape} vs {expected.shape}"


def test_conv_output_values():
    fixture = load_fixture("cnn")
    out = _conv2d(fixture["input"], fixture["kernel"], fixture["padding"], fixture["stride"])
    expected = np.array(fixture["expectedOutput"], dtype=float)
    assert np.allclose(out, expected, atol=fixture["tolerance"]), (
        f"max deviation: {np.abs(out - expected).max()}"
    )
