import json
from pathlib import Path

_VISUALS = Path(__file__).parent.parent / "visualizations"


def load_fixture(name: str) -> dict:
    """从 visualizations/<name>/fixture.json 加载测试数据。"""
    return json.loads((_VISUALS / name / "fixture.json").read_text())
