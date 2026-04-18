import json
import os
import re
from typing import Dict, List

import plotly.graph_objects as go
import plotly.io as pio


PEAK_LINE_PATTERN = re.compile(r"^[\d.E+-]+[\s\t]+[\d.E+-]+$")


def parse_ms2_from_mgf(mgf_path: str) -> List[Dict]:
    """解析 MGF 文件以提取所有 MS2 谱图信息。"""
    if not os.path.exists(mgf_path):
        raise FileNotFoundError(f"找不到文件: {mgf_path}")

    with open(mgf_path, "r", encoding="utf-8") as f:
        mgf_content = f.read()

    ion_blocks = re.findall(r"BEGIN IONS(.*?)END IONS", mgf_content, re.DOTALL)
    ms2_results: List[Dict] = []

    for block in ion_blocks:
        parsed = _parse_ion_block(block)
        if parsed is not None:
            ms2_results.append(parsed)

    return ms2_results


def parse_single_ms2_by_title(mgf_path: str, target_title: str) -> Dict | None:
    """按 title 解析单个 MS2 谱图，未命中时返回 None。"""
    if not os.path.exists(mgf_path):
        raise FileNotFoundError(f"找不到文件: {mgf_path}")

    normalized_title = (target_title or "").strip()
    if not normalized_title:
        return None

    in_block = False
    block_lines: List[str] = []

    with open(mgf_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "BEGIN IONS":
                in_block = True
                block_lines = []
                continue

            if line == "END IONS":
                if in_block:
                    parsed = _parse_ion_block("\n".join(block_lines))
                    if parsed is not None and parsed["title"] == normalized_title:
                        return parsed
                in_block = False
                block_lines = []
                continue

            if in_block:
                block_lines.append(line)

    return None


def _parse_ion_block(block: str) -> Dict | None:
    lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
    title, precursor_mz, ms_level = None, None, None
    peaks: List[tuple[float, float]] = []

    for line in lines:
        if line.startswith("TITLE="):
            title = line.split("=", 1)[1].strip()
        elif line.startswith("PEPMASS="):
            precursor_mz = float(line.split("=", 1)[1].split()[0])
        elif line.startswith("MSLEVEL="):
            try:
                ms_level = int(line.split("=", 1)[1])
            except Exception:
                ms_level = 2
        elif PEAK_LINE_PATTERN.match(line):
            parts = line.split()
            peaks.append((float(parts[0]), float(parts[1])))

    if (ms_level == 2 or ms_level is None) and precursor_mz:
        return {
            "title": title or "Unknown",
            "precursor_mz": precursor_mz,
            "ms2_peaks": peaks,
        }

    return None


def build_plot_payload(spec: Dict) -> Dict:
    """将单个谱图转换为 Plotly payload。"""
    mzs = [p[0] for p in spec["ms2_peaks"]]
    intensities = [p[1] for p in spec["ms2_peaks"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mzs,
            y=intensities,
            mode="markers",
            marker={"size": 4, "color": "red"},
            error_y={
                "type": "data",
                "array": [0] * len(intensities),
                "symmetric": False,
                "arrayminus": intensities,
                "width": 0,
                "thickness": 1,
                "color": "blue",
            },
            hovertemplate="m/z: %{x}<br>Int: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"MS2 Spectrum: {spec['title']}<br>Precursor m/z: {spec['precursor_mz']}",
        xaxis_title="m/z (Mass-to-Charge Ratio)",
        yaxis_title="Intensity",
        template="plotly_white",
        yaxis={"rangemode": "tozero"},
    )

    return {
        "title": spec["title"],
        "precursor_mz": spec["precursor_mz"],
        "plotly_data": json.loads(pio.to_json(fig)),
    }


def generate_single_ms2_plot(mgf_path: str, title: str) -> Dict | None:
    spec = parse_single_ms2_by_title(mgf_path, title)
    if spec is None:
        return None
    return build_plot_payload(spec)


def get_single_spectrum_plot_by_title(mgf_path: str, title: str) -> Dict:
    payload = generate_single_ms2_plot(mgf_path, title)
    if payload is None:
        raise ValueError(f"未找到标题为 {title} 的谱图")
    return payload


def generate_ms2_plot_json(mgf_path: str) -> str:
    """兼容旧逻辑：生成全部谱图 Plotly 数据 JSON 字符串。"""
    spectra_data = parse_ms2_from_mgf(mgf_path)
    output_data = [build_plot_payload(spec) for spec in spectra_data]
    return json.dumps(output_data, indent=4)
