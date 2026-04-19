import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import os
import traceback

# ======================== 全局参数配置（按要求设置）========================
# 谱图文件解析参数
SUPPORTED_SPECTRUM_FORMATS = ["mgf", "txt"]  # 支持的谱图文件格式
INTENSITY_SCALE = 100.0  # 强度归百化的最大值
MIN_MZ = 0.0  # 保留的最小m/z值
MAX_MZ = 5000.0  # 保留的最大m/z值
INTENSITY_DECIMALS = 2  # 归百化后强度保留小数位数

# 文件输出路径配置（按要求设置）
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_TEMP_DIR = _BACKEND_DIR / "temp"
SPECTRUM_OUTPUT_DIR = str(_DEFAULT_TEMP_DIR)  # 处理后的谱图文件保存路径
FRAGTREE_OUTPUT_DIR = str(_DEFAULT_TEMP_DIR)  # 处理后的碎裂树文件保存路径
STATS_OUTPUT_PATH = str(_DEFAULT_TEMP_DIR / "statas.json")  # 统计信息输出路径
VALID_PAIRS_MGF = str(_DEFAULT_TEMP_DIR / "valid_pairs_spectra.mgf")  # 有效对谱图文件
VALID_PAIRS_JSON = str(_DEFAULT_TEMP_DIR / "valid_pairs_fragtrees.json")  # 有效对碎裂树文件

def set_output_base(base_dir: str | Path) -> None:
    """基于指定目录重置所有输出路径，确保任务隔离。"""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    global SPECTRUM_OUTPUT_DIR, FRAGTREE_OUTPUT_DIR, STATS_OUTPUT_PATH, VALID_PAIRS_MGF, VALID_PAIRS_JSON
    SPECTRUM_OUTPUT_DIR = str(base_path)
    FRAGTREE_OUTPUT_DIR = str(base_path)
    STATS_OUTPUT_PATH = str(base_path / "statas.json")
    VALID_PAIRS_MGF = str(base_path / "valid_pairs_spectra.mgf")
    VALID_PAIRS_JSON = str(base_path / "valid_pairs_fragtrees.json")

# 碎裂树解析参数
FRAGTREE_VALID_KEY = "frag_tree"  # 判断碎裂树有效的关键字段
ROOT_FRAGMENT_ID = 0  # 碎裂树根节点的fragmentId值

# 示例输入文件路径（按要求设置）
EXAMPLE_SPECTRUM_FILE = str(_BACKEND_DIR / "testdata" / "C9H18N2O2_4.mgf")  # 输入谱图文件
EXAMPLE_FRAGTREE_FILE = str(_BACKEND_DIR / "testdata" / "C9H18N2O2_4.json")  # 输入碎裂树文件


class SpectrumNormalizer:
    """谱图数据标准化处理器（统一MGF/TXT解析逻辑，移除matchms依赖）"""
    
    def __init__(self):
        self.supported_formats = SUPPORTED_SPECTRUM_FORMATS
        # 创建输出目录
        Path(SPECTRUM_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        # 缓存解析后的所有谱图（key: title, value: 谱图数据）
        self.all_spectra: Dict[str, Dict[str, Any]] = {}
        # 缓存原始精度数据（保留precursor_mz和m/z的原始字符串格式）
        self.raw_precision_cache: Dict[str, Dict[str, Any]] = {}

    def parse_spectrum_file(self, file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        统一解析MGF/TXT谱图文件（基于原TXT解析逻辑）
        修复：合并同title的MS1/MS2块，统计真实谱图数量
        新增：缓存precursor_mz和m/z的原始字符串，保留精度
        """
        file_format = Path(file_path).suffix.lstrip('.').lower()
        if file_format not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {file_format}，仅支持{self.supported_formats}")
        
        # 第一步：解析所有BEGIN/END IONS块（含MS1/MS2），保留原始精度数据
        raw_blocks, raw_precision_data = self._parse_spectrum_unified(file_path)
        # 第二步：按TITLE合并MS1/MS2块为单个谱图（核心修复）
        spectra_list = self._merge_ms1_ms2_blocks(raw_blocks)
        
        # 批量归一化峰强度（仅调整强度，保留m/z原始精度）
        normalized_spectra = self._batch_normalize_intensity(spectra_list)
        
        # 缓存所有谱图（按title索引）
        self.all_spectra = {spec["title"]: spec for spec in normalized_spectra}
        # 缓存原始精度数据
        self.raw_precision_cache = raw_precision_data
        
        # 生成基础统计信息（修复：统计合并后的真实数量）
        stats = {
            "原始文件中的谱图数量": len(spectra_list),
            "原始谱图title": [s.get("title", "") for s in spectra_list],
            "解析的原始块数量": len(raw_blocks)  # 可选：保留块数量统计，便于验证
        }
        
        return self.all_spectra, stats

    def _parse_spectrum_unified(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        统一解析MGF/TXT文件的所有BEGIN/END IONS块（不合并，仅提取原始块）
        关键：缓存precursor_mz和m/z的原始字符串，保留原始精度
        """
        parsed_blocks = []
        raw_precision_data = {}  # key: title, value: {precursor_mz_str, mz_raw_list}
        current_title = ""
        current_precursor_mz_str = ""
        current_mz_raw_list = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]  # 保留原始行，不strip
        
        current_block = None
        peaks = []
        
        for line in lines:
            stripped_line = line.strip()
            # 兼容MGF/TXT的BEGIN IONS标识
            if stripped_line == "BEGIN IONS":
                current_block = {"peaks": [], "metadata": {}, "ms_level": 2}  # 默认MS2
                peaks = []
                current_title = ""
                current_precursor_mz_str = ""
                current_mz_raw_list = []
            elif stripped_line == "END IONS":
                if current_block and current_title:
                    # 过滤m/z范围（向量化）
                    peaks_np = np.array(peaks) if peaks else np.array([])
                    if len(peaks_np) > 0:
                        mask = (peaks_np[:, 0] >= MIN_MZ) & (peaks_np[:, 0] <= MAX_MZ)
                        peaks_np = peaks_np[mask]
                        # 修复语法错误：正确过滤原始m/z字符串列表
                        filtered_mz_raw = []
                        for i in range(len(current_mz_raw_list)):
                            # 正确的条件判断：先判断索引是否越界，再判断mask值
                            if i < len(mask) and mask[i]:
                                filtered_mz_raw.append(current_mz_raw_list[i])
                    else:
                        filtered_mz_raw = []
                    
                    current_block["peaks"] = peaks_np
                    parsed_blocks.append(current_block)
                    
                    # 缓存该title的原始精度数据（仅保留MS2的m/z原始数据）
                    if current_title not in raw_precision_data:
                        raw_precision_data[current_title] = {
                            "precursor_mz_str": current_precursor_mz_str,
                            "mz_raw_list": filtered_mz_raw
                        }
                    # 仅更新MS2块的m/z原始数据
                    elif current_block["ms_level"] == 2:
                        raw_precision_data[current_title]["mz_raw_list"] = filtered_mz_raw
                        # 保留MS1的precursor_mz原始字符串
                        if current_precursor_mz_str:
                            raw_precision_data[current_title]["precursor_mz_str"] = current_precursor_mz_str
                
                current_block = None
                peaks = []
            elif stripped_line.startswith("TITLE="):
                if current_block:
                    current_title = stripped_line.split("=", 1)[1].strip()
                    current_block["title"] = current_title
            elif stripped_line.startswith("PEPMASS="):
                if current_block:
                    # 保留原始字符串（不转换为浮点数）
                    current_precursor_mz_str = stripped_line.split("=", 1)[1].strip()
                    try:
                        current_block["precursor_mz"] = float(current_precursor_mz_str)
                    except ValueError:
                        current_block["precursor_mz"] = 0.0
                        current_precursor_mz_str = "0.0"
            elif stripped_line.startswith("MSLEVEL="):
                if current_block:
                    try:
                        current_block["ms_level"] = int(stripped_line.split("=", 1)[1].strip())
                    except ValueError:
                        current_block["ms_level"] = 2
            elif stripped_line.startswith("ADDUCTIONNAME="):
                if current_block:
                    current_block["adduct"] = stripped_line.split("=", 1)[1].strip()
            else:
                # 解析峰数据（保留m/z原始字符串，仅转换数值用于计算）
                if current_block and not stripped_line.startswith("#") and stripped_line:
                    parts = stripped_line.split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            peaks.append([mz, intensity])
                            # 保存m/z的原始字符串（保留精度）
                            current_mz_raw_list.append(parts[0])
                        except ValueError:
                            continue
        
        return parsed_blocks, raw_precision_data

    def _merge_ms1_ms2_blocks(self, raw_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        核心修复：按TITLE合并MS1/MS2块为单个谱图
        逻辑：
        1. 遍历所有块，按title分组
        2. 每个title保留MS1的元数据 + MS2的峰数据（核心谱图信息）
        3. 无MS2的块仅保留基础信息，无MS1的块补充默认MS1信息
        """
        merged_spectra = {}
        
        for block in raw_blocks:
            title = block.get("title", "")
            if not title:  # 无title的块跳过
                continue
            
            ms_level = block.get("ms_level", 2)
            # 初始化该title的谱图数据
            if title not in merged_spectra:
                merged_spectra[title] = {
                    "title": title,
                    "precursor_mz": block.get("precursor_mz", 0.0),
                    "adduct": block.get("adduct", ""),
                    "ms1_metadata": {},  # 存储MS1块的元数据
                    "ms2_peaks": np.array([]),  # 存储MS2块的峰数据（核心）
                    "metadata": block.get("metadata", {})
                }
            
            # 区分MS1/MS2块，分别存储信息
            if ms_level == 1:
                # MS1块：保留元数据（precursor_mz/adduct等）
                merged_spectra[title]["precursor_mz"] = block.get("precursor_mz", merged_spectra[title]["precursor_mz"])
                merged_spectra[title]["adduct"] = block.get("adduct", merged_spectra[title]["adduct"])
                merged_spectra[title]["ms1_metadata"] = block.copy()
            elif ms_level == 2:
                # MS2块：保留峰数据（核心谱图信息）
                merged_spectra[title]["ms2_peaks"] = block.get("peaks", np.array([]))
        
        # 转换为列表，并统一峰数据字段为"peaks"（兼容后续逻辑）
        final_spectra = []
        for title, spec in merged_spectra.items():
            final_spec = {
                "title": title,
                "precursor_mz": spec["precursor_mz"],
                "adduct": spec["adduct"],
                "peaks": spec["ms2_peaks"],  # 后续归一化/导出使用MS2峰数据
                "metadata": spec["metadata"],
                "ms1_metadata": spec["ms1_metadata"]  # 保留MS1信息，便于导出
            }
            final_spectra.append(final_spec)
        
        return final_spectra

    def _batch_normalize_intensity(self, spectra_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量归一化峰强度（仅调整强度保留两位小数，m/z完全保留原始精度）
        """
        if not spectra_list:
            return []
        
        # 步骤1：收集所有峰数据和索引
        all_peaks = []
        spec_indices = []
        normalized_spectra = [spec.copy() for spec in spectra_list]
        
        for i, spec in enumerate(spectra_list):
            peaks = spec.get("peaks", np.array([]))
            if len(peaks) > 0:
                all_peaks.append(peaks)
                spec_indices.extend([i] * len(peaks))
        
        if not all_peaks:
            return normalized_spectra
        
        # 步骤2：合并为二维数组（向量化处理）
        all_peaks_np = np.vstack(all_peaks)
        spec_indices_np = np.array(spec_indices)
        
        # 步骤3：按谱图分组计算每个谱图的最大强度
        max_intensities = np.zeros(len(spectra_list))
        for i in range(len(spectra_list)):
            mask = spec_indices_np == i
            if np.any(mask):
                max_intensities[i] = all_peaks_np[mask, 1].max()
        
        # 步骤4：批量缩放强度（向量化）
        # 避免除零
        max_intensities[max_intensities == 0] = 1.0
        # 按谱图索引获取对应最大值
        peak_max_intensities = max_intensities[spec_indices_np]
        # 归百化 + 保留两位小数（仅调整强度）
        all_peaks_np[:, 1] = np.round((all_peaks_np[:, 1] / peak_max_intensities) * INTENSITY_SCALE, INTENSITY_DECIMALS)
        
        # 步骤5：将归一化后的峰数据放回原谱图（仅更新强度值）
        ptr = 0
        for i, spec in enumerate(normalized_spectra):
            peaks = spec.get("peaks", np.array([]))
            if len(peaks) > 0:
                spec["peaks"] = all_peaks_np[ptr:ptr+len(peaks)]
                ptr += len(peaks)
        
        return normalized_spectra

    def export_valid_pairs_mgf(self, valid_titles: List[str]) -> None:
        """
        导出有效对的谱图为总MGF文件
        核心规则：
        1. precursor_mz使用原始字符串，保留原始精度
        2. m/z使用原始字符串，保留原始精度
        3. 强度使用归百化后的值，保留两位小数
        """
        # 确保输出目录存在
        output_path = Path(VALID_PAIRS_MGF)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for title in valid_titles:
                spec = self.all_spectra.get(title)
                precision_data = self.raw_precision_cache.get(title, {})
                if not spec or not precision_data:
                    continue
                
                # 获取原始精度数据
                precursor_mz_str = precision_data.get("precursor_mz_str", "0.0")
                mz_raw_list = precision_data.get("mz_raw_list", [])
                
                # 写入MS1部分（使用原始precursor_mz字符串）
                f.write("BEGIN IONS\n")
                f.write(f"TITLE={title}\n")
                f.write(f"PEPMASS={precursor_mz_str}\n")
                f.write("MSLEVEL=1\n")
                f.write(f"ADDUCTIONNAME={spec.get('adduct', '')}\n")
                f.write("END IONS\n\n")
                
                # 写入MS2部分（m/z用原始字符串，强度用归一化后的值）
                f.write("BEGIN IONS\n")
                f.write(f"TITLE={title}\n")
                f.write(f"PEPMASS={precursor_mz_str}\n")
                f.write("MSLEVEL=2\n")
                
                # 遍历峰数据：m/z用原始字符串，强度保留两位小数
                peaks = spec.get("peaks", np.array([]))
                for idx, peak in enumerate(peaks):
                    # 确保m/z原始字符串列表长度匹配
                    mz_str = mz_raw_list[idx] if idx < len(mz_raw_list) else f"{peak[0]}"
                    # 强度保留两位小数
                    intensity = round(peak[1], INTENSITY_DECIMALS)
                    f.write(f"{mz_str} {intensity:.{INTENSITY_DECIMALS}f}\n")
                
                f.write("END IONS\n\n")


class FragTreeParser:
    """碎裂树JSON文件解析器（仅生成总文件，无单个小文件）"""
    
    def __init__(self):
        Path(FRAGTREE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        # 缓存所有碎裂树（key: title, value: 碎裂树数据）
        self.all_fragtrees: Dict[str, Any] = {}

    def parse_fragtree_file(self, file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        解析碎裂树JSON文件，验证有效性
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 验证每个碎裂树有效性
        valid_titles = []
        root_info_list = []
        self.all_fragtrees = raw_data.copy()
        
        for title, frag_data in raw_data.items():
            # 验证碎裂树有效性
            is_valid = FRAGTREE_VALID_KEY in frag_data
            if is_valid:
                valid_titles.append(title)
                # 提取根节点信息
                root_info = self._extract_root_fragment_info(frag_data[FRAGTREE_VALID_KEY])
                if root_info:
                    root_info_list.append({
                        "title": title,
                        "adduct": root_info.get("adduct", ""),
                        "mz": root_info.get("mz", 0.0)
                    })
        
        # 生成统计信息
        stats = {
            "有效谱图title": valid_titles,
            "有效碎裂树根节点信息": root_info_list
        }
        
        return self.all_fragtrees, stats

    def _extract_root_fragment_info(self, frag_tree: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提取碎裂树根节点（fragmentId=0）的adduct和m/z信息"""
        fragments = frag_tree.get("fragments", [])
        for fragment in fragments:
            if fragment.get("fragmentId") == ROOT_FRAGMENT_ID:
                return {
                    "adduct": fragment.get("adduct", ""),
                    "mz": fragment.get("mz", 0.0)
                }
        return None

    def export_valid_pairs_json(self, valid_titles: List[str]) -> None:
        """
        导出有效对的碎裂树为总JSON文件（仅生成总文件，无单个小文件）
        """
        output_path = Path(VALID_PAIRS_JSON)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        valid_fragtrees = {}
        for title in valid_titles:
            frag_data = self.all_fragtrees.get(title)
            if frag_data:
                valid_fragtrees[title] = frag_data
        
        # 写入总JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(valid_fragtrees, f, ensure_ascii=False, indent=2)


class SpectrumFragTreeMatcher:
    """谱图-碎裂树匹配器，统计最终有效对并导出总文件"""
    
    @staticmethod
    def match(spectrum_stats: Dict[str, Any], fragtree_stats: Dict[str, Any],
              spectrum_normalizer: SpectrumNormalizer, fragtree_parser: FragTreeParser) -> Dict[str, Any]:
        """
        匹配谱图和碎裂树的title，计算交集并导出总文件
        """
        # 提取原始title列表（保留顺序）
        spectrum_titles = spectrum_stats.get("原始谱图title", [])
        fragtree_valid_titles = set(fragtree_stats.get("有效谱图title", []))

        # 建立快速索引：MGF 信息与碎裂树根节点信息
        mgf_info_map = spectrum_normalizer.all_spectra
        fragtree_root_map = {item.get("title"): item for item in fragtree_stats.get("有效碎裂树根节点信息", [])}

        # 计算交集并保留原始谱图的顺序
        final_valid_titles = [title for title in spectrum_titles if title in fragtree_valid_titles]

        # 生成合并后的根节点信息（MGF 优先，缺失时回落碎裂树）
        merged_root_info = []
        for title in final_valid_titles:
            mgf_info = mgf_info_map.get(title, {}) if mgf_info_map else {}
            fragtree_info = fragtree_root_map.get(title, {}) if fragtree_root_map else {}

            mz_from_mgf = mgf_info.get("precursor_mz") if mgf_info else None
            adduct_from_mgf = mgf_info.get("adduct") if mgf_info else ""

            merged_root_info.append({
                "title": title,
                "adduct": adduct_from_mgf if adduct_from_mgf else fragtree_info.get("adduct", ""),
                "mz": mz_from_mgf if mz_from_mgf else fragtree_info.get("mz", 0.0)
            })

        # 导出有效对的总MGF和总JSON文件（仅总文件）
        spectrum_normalizer.export_valid_pairs_mgf(final_valid_titles)
        fragtree_parser.export_valid_pairs_json(final_valid_titles)

        # 整合所有统计信息（覆盖碎裂树根节点信息为合并后的结果）
        merged_fragtree_stats = dict(fragtree_stats)
        merged_fragtree_stats["有效碎裂树根节点信息"] = merged_root_info

        combined_stats = {
            "谱图文件统计": spectrum_stats,
            "碎裂树文件统计": merged_fragtree_stats,
            "最终有效对": final_valid_titles,
            "最终有效对数量": len(final_valid_titles)
        }

        # 保存统计信息到总JSON文件
        stats_path = Path(STATS_OUTPUT_PATH)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(combined_stats, f, ensure_ascii=False, indent=2)

        return combined_stats


# ======================== 主函数 ========================
def main(
    spectrum_file_path: str = EXAMPLE_SPECTRUM_FILE,
    fragtree_file_path: str = EXAMPLE_FRAGTREE_FILE,
    output_base_dir: str | Path | None = None,
):
    """
    主处理流程
    """
    if output_base_dir:
        set_output_base(output_base_dir)

    # 验证输入文件路径存在且合法
    spec_path = Path(spectrum_file_path)
    frag_path = Path(fragtree_file_path)
    
    if not spec_path.exists():
        raise FileNotFoundError(f"谱图文件不存在: {spectrum_file_path}")
    if not frag_path.exists():
        raise FileNotFoundError(f"碎裂树文件不存在: {fragtree_file_path}")
    
    # 1. 处理谱图文件（统一MGF/TXT解析逻辑，修复数量统计）
    spectrum_normalizer = SpectrumNormalizer()
    _, spectrum_stats = spectrum_normalizer.parse_spectrum_file(str(spec_path))
    print(f"谱图文件处理完成：")
    print(f"  - 解析的原始块数量: {spectrum_stats['解析的原始块数量']}")
    print(f"  - 合并后的真实谱图数量: {spectrum_stats['原始文件中的谱图数量']}")
    
    # 2. 处理碎裂树文件
    fragtree_parser = FragTreeParser()
    _, fragtree_stats = fragtree_parser.parse_fragtree_file(str(frag_path))
    print(f"碎裂树文件处理完成，共找到 {len(fragtree_stats['有效谱图title'])} 个有效碎裂树")
    
    # 3. 匹配并导出有效对总文件
    matcher = SpectrumFragTreeMatcher()
    combined_stats = matcher.match(spectrum_stats, fragtree_stats, spectrum_normalizer, fragtree_parser)
    print(f"最终有效谱图-碎裂树对数量: {combined_stats['最终有效对数量']}")
    print(f"有效对总MGF文件: {VALID_PAIRS_MGF}")
    print(f"有效对总JSON文件: {VALID_PAIRS_JSON}")
    print(f"统计信息已保存至: {STATS_OUTPUT_PATH}")


# ======================== 执行入口 ========================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"处理出错: {str(e)}")
        traceback.print_exc()