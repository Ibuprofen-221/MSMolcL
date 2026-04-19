import os
import sys
import json
from pathlib import Path
from threading import Lock
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 第三方库
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

BACKEND_ROOT = str(Path(__file__).resolve().parents[1])
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from core.config import (
    retrieve_batch_size,
    retrieve_ema_mode,
    retrieve_eval_mode,
    retrieve_missing_tree_policy,
    retrieve_model_weight_paths,
    retrieve_pubchem_parquet_path,
    retrieve_pubchem_ppm,
    retrieve_seed,
    retrieve_shared_smiles_txt_path,
    statas_path,
    valid_pairs_fragtrees_path,
    valid_pairs_spectra_path,
)
import core.config as app_cfg

PROJ_ROOT = str(Path(__file__).resolve().parent / "model")
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)  # 插入到sys.path最前面，优先查找
# 项目内导入
from modules import FragSimiModel
from dataset import (
    parse_mgf_file, parse_ms2_from_mgf, preprocess_spectrum,
    get_adduct_map
)
from utils import mol_fp_encoder
from torch_geometric.data import Data


# ======================== 全局配置（统一从core/config.py读取）========================
MGF_FILE_PATH = ""
FRAG_TREE_JSON_PATH = ""
STATS_JSON_PATH = str(statas_path)

CANDIDATE_POOL_MODE = "pubchem"
PUBCHEM_PPM = retrieve_pubchem_ppm
PUBCHEM_PARQUET_PATH = str(retrieve_pubchem_parquet_path)
DATABASE_ROOT = str(app_cfg.retrieve_database_root)
SHARED_SMILES_TXT_PATH = str(retrieve_shared_smiles_txt_path)

MODEL_WEIGHT_PATHS = {k: str(v) for k, v in retrieve_model_weight_paths.items()}
DEFAULT_ION_MODE = "pos"
VALID_ION_MODES = tuple(MODEL_WEIGHT_PATHS.keys())
BATCH_SIZE = retrieve_batch_size
SEED = retrieve_seed
EVAL_MODE = retrieve_eval_mode
MISSING_TREE_POLICY = retrieve_missing_tree_policy
EMA_MODE = retrieve_ema_mode
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================== 固定配置（无需修改）========================
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision('highest')
except:
    pass

# ======================== 常量定义（加合物质量修正）========================
ELECTRON = 5.485_799_090_441e-4
PROTON   = 1.0072764665789
H2O      = 18.01056468403
ACN      = 41.02654910112
FA       = 46.00547930360
NA       = 22.98922070209
K        = 38.96315790649
NH4      = 18.03382555344

# 原始加合物字典（带空格格式）
ADDUCT_DELTA_RAW = {
    "M+":         +ELECTRON,
    "M+H":        -PROTON - ELECTRON,
    "[M + H]+":   -PROTON - ELECTRON,
    "M-H2O+H":    -PROTON + H2O + ELECTRON,
    "[M - H2O + H]+": -PROTON + H2O + ELECTRON,
    "M+ACN+H":    -(ACN + PROTON) + ELECTRON,
    "[M + ACN + H]+": -(ACN + PROTON) + ELECTRON,
    "M+FA+H":     -(FA + PROTON) + ELECTRON,
    "[M + FA + H]+": -(FA + PROTON) + ELECTRON,
    "M+Na":       -NA + ELECTRON,
    "[M + Na]+":  -NA + ELECTRON,
    "M+K":        -K + ELECTRON,
    "[M + K]+":   -K + ELECTRON,
    "M+NH4":      -NH4 + ELECTRON,
    "[M + NH4]+": -NH4 + ELECTRON,
    "Unknown":     None,
}

# ======================== 加合物处理工具函数 ========================
def normalize_adduct(adduct_str: str) -> str:
    """
    统一标准化加合物字符串：
    1. 去除所有空格
    2. 保留原始格式的其他字符（[]+等）
    3. 返回标准化后的字符串
    """
    if not isinstance(adduct_str, str):
        return "Unknown"
    # 仅去除空格，保留其他字符
    normalized = adduct_str.strip().replace(" ", "")
    return normalized if normalized else "Unknown"

# 构建标准化后的加合物字典（自动去除空格）
ADDUCT_TO_NEUTRAL_DELTA = {}
for adduct_raw, delta in ADDUCT_DELTA_RAW.items():
    adduct_norm = normalize_adduct(adduct_raw)
    ADDUCT_TO_NEUTRAL_DELTA[adduct_norm] = delta
    # 同时保留原始键（防止兼容问题）
    ADDUCT_TO_NEUTRAL_DELTA[adduct_raw] = delta

# ======================== 多数据库候选池工具函数 ========================
def discover_database_paths(database_root: str, fallback_pubchem_path: str) -> dict[str, str]:
    db_paths: dict[str, str] = {}
    root = Path(database_root)
    if root.exists() and root.is_dir():
        for parquet_file in root.rglob("*.parquet"):
            db_name = parquet_file.parent.name.strip()
            if db_name and db_name not in db_paths:
                db_paths[db_name] = str(parquet_file)

    if "pubchem" not in db_paths and fallback_pubchem_path:
        fallback = Path(fallback_pubchem_path)
        if fallback.exists() and fallback.is_file():
            db_paths["pubchem"] = str(fallback)

    return db_paths


def resolve_parquet_columns(parquet_path: str) -> dict[str, str | None]:
    schema_names = pq.ParquetFile(parquet_path).schema_arrow.names
    lower_map = {name.lower(): name for name in schema_names}

    def _pick(*candidates: str) -> str | None:
        for c in candidates:
            found = lower_map.get(c.lower())
            if found:
                return found
        return None

    return {
        "mass": _pick("exactmass", "exact_mass"),
        "smiles": _pick("rdkit_smiles", "smiles"),
        "formula": _pick("formula"),
        "generic_name": _pick("generic_name"),
        "database_name": _pick("database_name"),
        "database_id": _pick("database_id"),
        "inchi_key": _pick("inchi_key"),
    }


def load_global_exactmass(parquet_path: str, mass_column: str) -> tuple[np.ndarray, pq.ParquetFile]:
    """仅加载全局 exactmass 数组和 ParquetFile 对象"""
    arr = ds.dataset(parquet_path, format="parquet").to_table(columns=[mass_column])[mass_column]
    exactmasses = arr.to_numpy(zero_copy_only=False).astype(np.float64)

    if exactmasses.size > 1 and not np.all(exactmasses[:-1] <= exactmasses[1:]):
        print(f"[警告] {parquet_path} 并非严格非降序，请确认已按质量列排序！")

    return exactmasses, pq.ParquetFile(parquet_path)

def mz_to_neutral_mass(mz: float, adduct_name: str) -> float | None:
    """计算分子中性质量（兼容各种格式的加合物字符串）"""
    # 标准化加合物字符串（仅去除空格）
    clean_adduct = normalize_adduct(adduct_name)
    
    # 查找delta值
    delta = ADDUCT_TO_NEUTRAL_DELTA.get(clean_adduct, None)
    if delta is None:
        print(f"[警告] 不支持的加合物类型：{adduct_name}（标准化后：{clean_adduct}），无法计算中性质量")
        print(f"[提示] 支持的加合物类型：{list(ADDUCT_TO_NEUTRAL_DELTA.keys())}")
        return None
    
    # 计算并返回中性质量
    neutral_mass = float(mz + delta)
    print(f"[信息] 加合物 {adduct_name} → 标准化 {clean_adduct}，m/z {mz} → 中性质量 {neutral_mass:.6f}")
    return neutral_mass

def load_smiles_in_ppm_range(
    neutral_mass: float,
    ppm: float,
    exactmasses: np.ndarray,
    parquet_file: pq.ParquetFile,
    smiles_column: str,
) -> list[str]:
    """加载指定 ppm 范围内的 SMILES 列表（单个谱图专属候选池）"""
    lo = neutral_mass - neutral_mass * ppm * 1e-6
    hi = neutral_mass + neutral_mass * ppm * 1e-6
    global_start = int(np.searchsorted(exactmasses, lo, side='left'))
    global_end = int(np.searchsorted(exactmasses, hi, side='right'))

    if global_start >= global_end:
        return []

    row_groups_to_scan = []
    current_rg_start = 0
    for rg_idx in range(parquet_file.num_row_groups):
        rg_meta = parquet_file.metadata.row_group(rg_idx)
        rg_row_count = rg_meta.num_rows
        current_rg_end = current_rg_start + rg_row_count

        if not (current_rg_end <= global_start or current_rg_start >= global_end):
            slice_start = max(global_start - current_rg_start, 0)
            slice_end = min(global_end - current_rg_start, rg_row_count)
            row_groups_to_scan.append((rg_idx, slice_start, slice_end))

        current_rg_start = current_rg_end
        if current_rg_start >= global_end:
            break

    smiles_list: list[str] = []
    for rg_idx, slice_start, slice_end in row_groups_to_scan:
        rg_table = parquet_file.read_row_group(rg_idx, columns=[smiles_column])
        sliced_table = rg_table.slice(offset=slice_start, length=slice_end - slice_start)
        smiles = sliced_table[smiles_column].to_pylist()
        smiles_list.extend([str(s).strip() for s in smiles if isinstance(s, str) and s.strip()])

    return list(dict.fromkeys(smiles_list))

def load_shared_smiles_pool(txt_path: str) -> list:
    """加载自定义/自选模式的全局共用SMILES候选池"""
    if not os.path.exists(txt_path):
        print(f"[错误] 共用SMILES文件不存在: {txt_path}")
        sys.exit(1)

    with open(txt_path, 'r', encoding='utf-8') as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    unique_smiles = list(set(smiles_list))
    print(f"[信息] 加载共用候选池：{len(unique_smiles)} 个分子")
    return unique_smiles


def _meta_aggregate_entry() -> dict:
    return {
        "formula": None,
        "inchi_key": None,
        "generic_name": set(),
        "database_name": set(),
        "database_id": set(),
    }


def _merge_meta(target: dict, row: dict, db_name: str) -> None:
    formula = row.get("formula")
    inchi_key = row.get("inchi_key")
    generic_name = row.get("generic_name")
    database_name = row.get("database_name")
    database_id = row.get("database_id")

    if not target.get("formula") and isinstance(formula, str) and formula.strip():
        target["formula"] = formula.strip()
    if not target.get("inchi_key") and isinstance(inchi_key, str) and inchi_key.strip():
        target["inchi_key"] = inchi_key.strip()

    if isinstance(generic_name, str) and generic_name.strip():
        target["generic_name"].add(generic_name.strip())
    if isinstance(database_name, str) and database_name.strip():
        target["database_name"].add(database_name.strip())
    else:
        target["database_name"].add(db_name)
    if isinstance(database_id, str) and database_id.strip():
        target["database_id"].add(database_id.strip())


def load_top_smiles_metadata(
    top_smiles: list[str],
    selected_db_infos: list[dict],
    neutral_mass: float,
    ppm: float,
) -> dict[str, dict]:
    """仅针对 TopK SMILES 回查多库元数据，避免在候选阶段缓存海量元信息。"""
    result: dict[str, dict] = {smi: _meta_aggregate_entry() for smi in top_smiles}
    top_set = set(top_smiles)
    if not top_set:
        return result

    lo = neutral_mass - neutral_mass * ppm * 1e-6
    hi = neutral_mass + neutral_mass * ppm * 1e-6

    for db in selected_db_infos:
        db_name = db["name"]
        parquet_file: pq.ParquetFile = db["parquet_file"]
        exactmasses: np.ndarray = db["exactmasses"]

        mass_col = db["columns"]["mass"]
        smiles_col = db["columns"]["smiles"]
        optional_pairs = [
            ("formula", db["columns"].get("formula")),
            ("generic_name", db["columns"].get("generic_name")),
            ("database_name", db["columns"].get("database_name")),
            ("database_id", db["columns"].get("database_id")),
            ("inchi_key", db["columns"].get("inchi_key")),
        ]
        read_columns = [smiles_col] + [col for _, col in optional_pairs if col]

        global_start = int(np.searchsorted(exactmasses, lo, side='left'))
        global_end = int(np.searchsorted(exactmasses, hi, side='right'))
        if global_start >= global_end:
            continue

        current_rg_start = 0
        for rg_idx in range(parquet_file.num_row_groups):
            rg_meta = parquet_file.metadata.row_group(rg_idx)
            rg_count = rg_meta.num_rows
            current_rg_end = current_rg_start + rg_count

            if current_rg_end <= global_start or current_rg_start >= global_end:
                current_rg_start = current_rg_end
                if current_rg_start >= global_end:
                    break
                continue

            slice_start = max(global_start - current_rg_start, 0)
            slice_end = min(global_end - current_rg_start, rg_count)
            table = parquet_file.read_row_group(rg_idx, columns=read_columns)
            sliced = table.slice(offset=slice_start, length=slice_end - slice_start)
            py_rows = sliced.to_pylist()

            for row in py_rows:
                smi = row.get(smiles_col)
                if not isinstance(smi, str):
                    continue
                smi = smi.strip()
                if smi not in top_set:
                    continue
                normalized_row = {
                    alias: row.get(col_name) if col_name else None
                    for alias, col_name in optional_pairs
                }
                _merge_meta(result[smi], normalized_row, db_name)

            current_rg_start = current_rg_end
            if current_rg_start >= global_end:
                break

    return result


# ======================== 数据集类 ========================
class MolDataset(Dataset):
    """用于加载SMILES列表的数据集"""
    def __init__(self, smis, cfg):
        self.smis = smis
        self.cfg = cfg

    def __len__(self):
        return len(self.smis)

    def __getitem__(self, idx):
        smi = self.smis[idx]
        features = {}

        try:
            from config import CFG
            if len(CFG) == 0:
                CFG.update(self.cfg)
        except Exception:
            pass
        mol_cfg = self.cfg.model.mol_encoder
        if 'fp' in mol_cfg.type:
            try:
                fp = mol_fp_encoder(smi, tp=mol_cfg.fp.type, nbits=mol_cfg.fp.nbits)
                if fp is None:
                    return None
                features['mol_fps'] = fp
            except ImportError:
                print("错误：未找到utils模块中的mol_fp_encoder函数！")
                sys.exit(1)
        if 'gnn' in mol_cfg.type:
            try:
                from utils import mol_graph_featurizer
                graph_features = mol_graph_featurizer(smi)
                if not graph_features:
                    return None
                features.update(graph_features)
            except ImportError:
                print("错误：未找到utils模块中的mol_graph_featurizer函数！")
                sys.exit(1)
        features['_row_idx'] = idx
        features['_smiles'] = smi  # 保存原始SMILES
        return features

class MSDataset(Dataset):
    """适配单个MGF+碎裂树JSON的谱图数据集（模型输入）"""
    def __init__(self, mgf_file, frag_tree_json, cfg, missing_tree_policy='discard'):
        self.mgf_file = mgf_file
        self.frag_tree_json = frag_tree_json
        self.cfg = cfg
        self.missing_tree_policy = missing_tree_policy
        
        try:
            from FragmentationTreeEncoder import FragmentTreeProcessor
            self.processor = FragmentTreeProcessor()
        except ImportError:
            print("错误：未找到FragmentationTreeEncoder模块中的FragmentTreeProcessor类！")
            sys.exit(1)
        
        self.queries = self._load_single_query()
        
        # 判断是否需要碎裂树特征
        fusion = getattr(getattr(cfg.model, 'ms_encoder', {}), 'fusion', None)
        if fusion is None:
            legacy_ft_mode = getattr(getattr(cfg.model.ms_encoder, 'frag_tree', {}), 'mode', None)
            fusion = 'clerms-only' if legacy_ft_mode == 'disable' else 'concat'
        self.need_tree = fusion in ('xattn', 'concat', 'tree-only')
        self.enhanced_tree = bool(getattr(getattr(cfg.model, 'ms_encoder', {}), 'tree_encoder', {}).get('enhanced_features', False))
        
        try:
            self.adduct_map = get_adduct_map(cfg)
        except ImportError:
            print("错误：未找到dataset模块中的get_adduct_map函数！")
            sys.exit(1)

    def _load_single_query(self):
        """加载单个MGF+JSON文件的谱图数据"""
        all_queries = []
        # 加载碎裂树JSON
        frag_tree_db = {}
        try:
            with open(self.frag_tree_json, 'r') as f:
                frag_tree_db = json.load(f)
        except Exception as e:
            print(f"警告：加载碎裂树文件失败 {self.frag_tree_json}: {e}")

        def _extract_root_info(entry: dict) -> dict:
            frag_tree = entry.get('frag_tree') if isinstance(entry, dict) else None
            if not frag_tree:
                return {}
            for fragment in frag_tree.get('fragments', []):
                if fragment.get('fragmentId') == 0:
                    return {
                        'mz': fragment.get('mz', 0.0),
                        'adduct': fragment.get('adduct', '')
                    }
            return {}
        
        # 解析MGF文件
        try:
            ms1_data = parse_mgf_file(self.mgf_file, self.cfg)
            ms2_data = parse_ms2_from_mgf(self.mgf_file)
        except ImportError:
            print("错误：未找到dataset模块中的parse_mgf_file/parse_ms2_from_mgf函数！")
            sys.exit(1)
        except Exception as e:
            print(f"错误：解析MGF文件失败 {self.mgf_file}: {e}")
            sys.exit(1)
        
        # 清洗ms2数据
        cleaned_ms2_data = []
        for ms2_entry in ms2_data:
            if not ms2_entry.get('title'):
                continue
            peaks = ms2_entry.get('ms2_peaks', [])
            cleaned_peaks = []
            for peak in peaks:
                if isinstance(peak, (list, tuple)) and len(peak) == 2:
                    mz, intensity = peak
                    if isinstance(mz, (int, float)) and isinstance(intensity, (int, float)):
                        cleaned_peaks.append(peak)
            if len(cleaned_peaks) == 0:
                print(f"警告：谱图 {ms2_entry.get('title', 'unknown')} 无有效峰数据，跳过")
                continue
            ms2_entry['ms2_peaks'] = cleaned_peaks
            cleaned_ms2_data.append(ms2_entry)
        
        ms2_dict = {item['title']: item for item in cleaned_ms2_data}
        
        for ms1_entry in ms1_data:
            title = ms1_entry.get('title')
            if not title:
                print("警告：MGF文件中未找到谱图TITLE字段，跳过该条目")
                continue
            
            ms2_entry = ms2_dict.get(title)
            if not ms2_entry:
                print(f"警告：谱图 {title} 无对应的MS2峰数据，跳过")
                continue
            
            # 处理true_smiles
            true_smiles = ms1_entry.get('smiles', 'unknown')
            if true_smiles is None or true_smiles.strip() == "":
                true_smiles = 'unknown'
            
            frag_tree_entry = frag_tree_db.get(title, {})
            root_info = _extract_root_info(frag_tree_entry)

            merged_adduct = ms1_entry.get('adduct') or root_info.get('adduct') or 'Unknown'
            merged_precursor_mz = (
                ms1_entry.get('precursor_mz')
                or root_info.get('mz')
                or ms2_entry.get('precursor_mz')
                or 0.0
            )
            ms1_entry['adduct'] = merged_adduct
            ms1_entry['precursor_mz'] = merged_precursor_mz
            ms2_entry['precursor_mz'] = merged_precursor_mz

            query_item = {
                **ms1_entry, 
                **ms2_entry, 
                'frag_tree_entry': frag_tree_entry, 
                'source_mgf': self.mgf_file,
                'true_smiles': true_smiles
            }
            all_queries.append(query_item)
        
        if not all_queries:
            print("错误：MGF文件中未解析到有效谱图数据")
            sys.exit(1)
        return all_queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        title = query.get('title')
        frag_tree_entry = query.get('frag_tree_entry', {})
        pyg_data = None

        if frag_tree_entry and 'frag_tree' in frag_tree_entry:
            try:
                processed_data = self.processor.json_to_pyg({'frag_tree': frag_tree_entry['frag_tree']},
                                                            edge_features=True, enhanced=self.enhanced_tree)
                if processed_data.num_nodes > 0:
                    pyg_data = processed_data
                else:
                    print(f"警告：谱图 {title} 的碎裂树处理后为空图，使用占位符")
            except Exception as e:
                print(f"警告：处理谱图 {title} 的碎裂树失败: {e}，使用占位符")
        
        # 即使无碎裂树，也创建占位符
        if pyg_data is None and self.need_tree:
            print(f"信息：为谱图 {title} 创建空图占位符")
            node_dim = 18 if self.enhanced_tree else 15
            edge_dim = 14 if self.enhanced_tree else 14
            pyg_data = Data(
                x=torch.zeros((1, node_dim), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, edge_dim), dtype=torch.float)
            )
        
        try:
            sdim = getattr(getattr(self.cfg.model, 'ms_encoder', {}), 'spectrum_dim', 100)
            sorder = getattr(getattr(self.cfg.model, 'ms_encoder', {}), 'spectrum_order', 'intensity_desc')
            spec_tensor = preprocess_spectrum(query['ms2_peaks'], dim=sdim, order=sorder)
        except Exception as e:
            print(f"警告：谱图 {title} 峰数据预处理失败: {e}，跳过")
            return None
        
        return {
            'title': title,
            'spec_tensor': spec_tensor,
            'adduct_type_idx': self.adduct_map.get(query['adduct'], self.adduct_map["Unknown"]),
            'precursor_mz': query['precursor_mz'],
            'true_smiles': query['true_smiles'],
            'adduct': query.get('adduct', None),
            'pyg_data': pyg_data
        }

# ======================== 数据拼接函数 ========================
def collate_mol(batch):
    batch = list(filter(None, batch))
    if not batch:
        return {}
    bat = {}
    if 'mol_fps' in batch[0]:
        bat['mol_fps'] = torch.stack([b['mol_fps'] for b in batch])
    if 'V' in batch[0]:
        try:
            from utils import pad_V, pad_A
            max_n = max(b['V'].shape[0] for b in batch)
            bat['V'] = torch.stack([pad_V(b['V'], max_n) for b in batch])
            bat['A'] = torch.stack([pad_A(b['A'], max_n) for b in batch])
            bat['mol_size'] = torch.cat([b['mol_size'] for b in batch])
        except ImportError:
            print("错误：未找到utils模块中的pad_V/pad_A函数！")
            sys.exit(1)
    bat['row_idx'] = torch.tensor([b['_row_idx'] for b in batch], dtype=torch.long)
    bat['smiles_list'] = [b['_smiles'] for b in batch]  # 保存SMILES
    return bat

def collate_ms(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    try:
        from torch_geometric.data import Batch
    except ImportError:
        print("错误：未安装torch_geometric库，请先安装！")
        sys.exit(1)
    return {
        'title': [b['title'] for b in batch],
        'spec_tensor': torch.stack([b['spec_tensor'] for b in batch]).unsqueeze(1),
        'adduct_type_idx': torch.tensor([b['adduct_type_idx'] for b in batch]),
        'precursor_mz': torch.tensor([b['precursor_mz'] for b in batch]),
        'true_smiles': [b['true_smiles'] for b in batch],
        'adduct': [b['adduct'] for b in batch],
        'pyg_data': Batch.from_data_list([b['pyg_data'] for b in batch])
    }

# ======================== 进程级模型常驻 ========================
_RUNTIME_MODELS: dict[str, torch.nn.Module] = {}
_RUNTIME_CFGS: dict[str, object] = {}
_RUNTIME_CHECKPOINTS: dict[str, dict] = {}
_RUNTIME_INIT_LOCK = Lock()


def _normalize_ion_mode(ion_mode: str | None) -> str:
    normalized = (ion_mode or DEFAULT_ION_MODE).strip().lower()
    if normalized not in VALID_ION_MODES:
        raise RuntimeError(f"不支持的离子模式: {ion_mode}")
    return normalized


def init_retrieve_process(ion_mode: str = DEFAULT_ION_MODE) -> None:
    """在子进程启动时按离子模式加载并缓存模型。"""
    mode = _normalize_ion_mode(ion_mode)
    if mode in _RUNTIME_MODELS and mode in _RUNTIME_CFGS:
        return

    with _RUNTIME_INIT_LOCK:
        if mode in _RUNTIME_MODELS and mode in _RUNTIME_CFGS:
            return

        model_weight_path = MODEL_WEIGHT_PATHS.get(mode)
        if not model_weight_path:
            raise RuntimeError(f"离子模式 {mode} 未配置模型权重路径")

        try:
            checkpoint = torch.load(model_weight_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"加载模型失败({mode}): {e}") from e

        if "config" not in checkpoint or not isinstance(checkpoint["config"], dict):
            raise RuntimeError(f"模型文件中未找到内置配置({mode})")

        from config import CFG, ConfigDict

        def _to_cfgdict(d):
            if isinstance(d, dict):
                return ConfigDict({k: _to_cfgdict(v) for k, v in d.items()})
            return d

        CFG.clear()
        cfg_runtime = ConfigDict(_to_cfgdict(checkpoint["config"]))
        CFG.update(cfg_runtime)

        model = FragSimiModel(CFG, enable_compile=False).to(DEVICE).eval()

        from compatibility import load_checkpoint_with_compat

        _ = load_checkpoint_with_compat(
            model,
            checkpoint,
            cfg=CFG,
            ema=EMA_MODE,
            verbose=True,
        )

        _RUNTIME_MODELS[mode] = model
        _RUNTIME_CFGS[mode] = CFG

        _RUNTIME_CHECKPOINTS[mode] = checkpoint


def warmup_retrieve_process(ion_mode: str = DEFAULT_ION_MODE) -> int:
    """触发一次进程初始化并返回pid，用于启动预热。"""
    init_retrieve_process(ion_mode=ion_mode)
    return os.getpid()


def get_retrieve_runtime(ion_mode: str = DEFAULT_ION_MODE):
    """获取当前进程对应离子模式的常驻模型与配置。"""
    mode = _normalize_ion_mode(ion_mode)
    if mode not in _RUNTIME_MODELS or mode not in _RUNTIME_CFGS:
        init_retrieve_process(ion_mode=mode)
    return _RUNTIME_MODELS[mode], _RUNTIME_CFGS[mode]


# ======================== 核心检索函数 ========================
def encode_molecules(smiles_list, model, cfg, device, batch_size=256):
    """编码分子列表为向量"""
    dataset = MolDataset(smiles_list, cfg)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, collate_fn=collate_mol, pin_memory=True)
    
    embeddings = []
    smiles_mapping = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="编码分子"):
            if not batch:
                continue
            smiles_mapping.extend(batch['smiles_list'])
            row_idx = batch.pop('row_idx')
            
            # 模型编码
            for k, v in batch.items():
                if k != 'smiles_list':
                    batch[k] = v.to(device)
            mol_emb = model.encode_mol(batch).cpu()
            embeddings.append(mol_emb)
    
    if not embeddings:
        return None, None
    return torch.cat(embeddings), smiles_mapping

def encode_spectra(ms_loader, model, device):
    """编码谱图为向量"""
    spec_embeddings = []
    spec_titles = []
    spec_true_smiles = []
    
    with torch.no_grad():
        for batch in tqdm(ms_loader, desc="编码谱图"):
            if not batch:
                continue
            spec_titles.extend(batch['title'])
            spec_true_smiles.extend(batch['true_smiles'])
            
            # 准备输入
            batch_input = {}
            for k, v in batch.items():
                if k not in ['title', 'true_smiles', 'adduct']:
                    if k != 'pyg_data':
                        batch_input[k] = v.to(device)
                    else:
                        batch_input[k] = v
            
            # 编码谱图
            spec_emb = model.encode_ms(batch_input).cpu()
            spec_embeddings.append(spec_emb)
    
    if not spec_embeddings:
        return None, None, None
    return torch.cat(spec_embeddings), spec_titles, spec_true_smiles

def retrieve_spectrum(spec_emb, mol_embs, smiles_list, topk=100):
    """谱图检索（计算相似度并排序）"""
    # 归一化
    spec_emb = F.normalize(spec_emb, dim=-1)
    mol_embs = F.normalize(mol_embs, dim=-1)
    
    # 计算相似度
    similarity = torch.mm(spec_emb, mol_embs.T)
    
    # 排序
    topk = min(topk, len(smiles_list))
    scores, indices = torch.topk(similarity, k=topk, dim=1, largest=True)
    
    # 转换为结果
    results = {}
    for i in range(spec_emb.shape[0]):
        top_smiles = [smiles_list[idx] for idx in indices[i].numpy()]
        top_scores = [float(score) for score in scores[i].numpy()]
        results[spec_titles[i]] = {
            "top1": top_smiles[0] if top_smiles else "",
            "top10": top_smiles[:10] if len(top_smiles)>=10 else top_smiles,
            "top100": top_smiles,
            "top100_score": top_scores,
            "true_smiles": spec_true_smiles[i],
            "status": "success" if top_smiles else "failed"
        }
    return results

# ======================== 主函数 ========================
def main(
    search_type: str = CANDIDATE_POOL_MODE,
    ppm_range: float = PUBCHEM_PPM,
    custom_smiles_list: list[str] | None = None,
    ion_mode: str = DEFAULT_ION_MODE,
    statas_json_path: str = str(statas_path),
    fragtrees_json_path: str = str(valid_pairs_fragtrees_path),
    spectra_mgf_path: str = str(valid_pairs_spectra_path),
    pubchem_parquet_path: str = PUBCHEM_PARQUET_PATH,
    shared_smiles_txt_path: str = SHARED_SMILES_TXT_PATH,
    databases: list[str] | None = None,
    database_root: str = DATABASE_ROOT,
):
    # 1. 加载统计信息JSON
    print("=== 加载统计信息JSON ===")
    if not os.path.exists(statas_json_path):
        print(f"[错误] 统计信息JSON不存在: {statas_json_path}")
        sys.exit(1)
    
    with open(statas_json_path, 'r', encoding='utf-8') as f:
        stats_data = json.load(f)
    
    if "碎裂树文件统计" not in stats_data or "有效碎裂树根节点信息" not in stats_data["碎裂树文件统计"]:
        print(f"[错误] 统计信息JSON缺少'碎裂树文件统计/有效碎裂树根节点信息'字段")
        sys.exit(1)

    # 先读取原有统计信息映射，后续将按“MGF 优先，缺失回落碎裂树”重新合并
    raw_spec_stats_list = stats_data["碎裂树文件统计"].get("有效碎裂树根节点信息", [])
    stats_title_map = {item.get("title"): item for item in raw_spec_stats_list if item.get("title")}
    print(f"[信息] 统计信息JSON中加载 {len(raw_spec_stats_list)} 个谱图信息")

    # 2. 获取进程常驻模型
    print("\n=== 获取常驻模型 ===")
    try:
        mode = _normalize_ion_mode(ion_mode)
        model, cfg = get_retrieve_runtime(ion_mode=mode)
    except Exception as e:
        print(f"[错误] 初始化常驻模型失败: {e}")
        sys.exit(1)

    # 2.5 构建 title -> (mz, adduct) 映射（MGF 优先，缺失回落碎裂树/统计文件）
    def _build_mgf_info_map(mgf_path: str, cfg_runtime) -> dict:
        info_map = {}
        try:
            for item in parse_mgf_file(mgf_path, cfg_runtime):
                title = item.get("title")
                if not title:
                    continue
                info_map[title] = {
                    "mz": item.get("precursor_mz", 0.0),
                    "adduct": item.get("adduct", "")
                }
        except Exception as e:
            print(f"[警告] 解析 MGF 获取前体质量/加合物失败：{e}")
        return info_map

    def _build_frag_root_map(frag_path: str) -> dict:
        info_map = {}
        try:
            with open(frag_path, 'r', encoding='utf-8') as f:
                frag_db = json.load(f)
            for title, entry in frag_db.items():
                frag_tree = entry.get("frag_tree") if isinstance(entry, dict) else None
                if not frag_tree:
                    continue
                for fragment in frag_tree.get("fragments", []):
                    if fragment.get("fragmentId") == 0:
                        info_map[title] = {
                            "mz": fragment.get("mz", 0.0),
                            "adduct": fragment.get("adduct", "")
                        }
                        break
        except Exception as e:
            print(f"[警告] 解析碎裂树根节点失败：{e}")
        return info_map

    mgf_root_map = _build_mgf_info_map(spectra_mgf_path, cfg)
    frag_root_map = _build_frag_root_map(fragtrees_json_path)

    def _merge_title_info(title: str) -> dict:
        mgf_info = mgf_root_map.get(title, {})
        frag_info = frag_root_map.get(title, {})
        stats_info = stats_title_map.get(title, {})
        return {
            "mz": mgf_info.get("mz") or frag_info.get("mz") or stats_info.get("mz", 0.0),
            "adduct": mgf_info.get("adduct") or frag_info.get("adduct") or stats_info.get("adduct", "")
        }

    spec_title_map = {title: _merge_title_info(title) for title in stats_title_map.keys()}

    # 3. 加载谱图数据集（模型输入）
    print("\n=== 加载谱图数据 ===")
    try:
        ms_dataset = MSDataset(spectra_mgf_path, fragtrees_json_path, cfg, missing_tree_policy=MISSING_TREE_POLICY)
        ms_loader = DataLoader(ms_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=collate_ms, pin_memory=True)
    except Exception as e:
        print(f"[错误] 加载谱图数据失败: {e}")
        sys.exit(1)
    
    # 4. 编码谱图
    print("\n=== 编码谱图 ===")
    spec_embs, spec_titles, spec_true_smiles = encode_spectra(ms_loader, model, DEVICE)
    if spec_embs is None:
        print("[错误] 谱图编码失败")
        sys.exit(1)

    # 计算谱图两两余弦相似度（仅普通检索使用）
    pair_similarity_map = {}
    if spec_embs is not None and len(spec_titles) >= 2:
        norm_spec = F.normalize(spec_embs, dim=-1)
        sim_matrix = torch.mm(norm_spec, norm_spec.T)
        n_specs = len(spec_titles)
        for i in range(n_specs):
            for j in range(i + 1, n_specs):
                key = "-".join(sorted([spec_titles[i], spec_titles[j]]))
                pair_similarity_map[key] = float(sim_matrix[i, j])
    stats_data["谱图相似度"] = pair_similarity_map

    # 确保所有待检索谱图的 m/z 与加合物已按优先级合并
    for title in spec_titles:
        if title not in spec_title_map:
            spec_title_map[title] = _merge_title_info(title)

    # 5. 处理候选池并执行检索
    print("\n=== 处理候选池 ===")
    final_results = {}
    
    if search_type == "pubchem":
        available_db_paths = discover_database_paths(database_root, pubchem_parquet_path)
        selected_databases = list(dict.fromkeys([d.strip() for d in (databases or []) if isinstance(d, str) and d.strip()]))
        if not selected_databases:
            selected_databases = ["pubchem"]

        missing = [db for db in selected_databases if db not in available_db_paths]
        if missing:
            print(f"[错误] 选中的数据库不存在: {', '.join(missing)}")
            sys.exit(1)

        selected_db_infos: list[dict] = []
        for db_name in selected_databases:
            db_path = available_db_paths[db_name]
            cols = resolve_parquet_columns(db_path)
            if not cols.get("mass") or not cols.get("smiles"):
                print(f"[错误] 数据库 {db_name} 缺少质量列或SMILES列")
                sys.exit(1)
            exactmasses, parquet_file = load_global_exactmass(db_path, cols["mass"])
            selected_db_infos.append({
                "name": db_name,
                "path": db_path,
                "columns": cols,
                "exactmasses": exactmasses,
                "parquet_file": parquet_file,
            })

        print(f"[信息] 使用多数据库+{ppm_range}ppm模式：{', '.join(selected_databases)}")

        spec_candidates: dict[str, list[str]] = {}
        spec_neutral_mass_map: dict[str, float] = {}
        failed_results = {}

        def _load_for_title(title: str):
            spec_info = spec_title_map.get(title) or _merge_title_info(title)
            if not spec_info:
                return title, None, None, "统计信息缺失"
            adduct = spec_info.get("adduct", "")
            mz = spec_info.get("mz", 0.0)
            if not mz:
                return title, None, None, "缺少前体质量"

            neutral_mass = mz_to_neutral_mass(mz, adduct)
            if neutral_mass is None:
                return title, None, None, "无法计算中性质量"

            merged_smiles: list[str] = []
            for db in selected_db_infos:
                one_db = load_smiles_in_ppm_range(
                    neutral_mass,
                    ppm_range,
                    db["exactmasses"],
                    db["parquet_file"],
                    db["columns"]["smiles"],
                )
                merged_smiles.extend(one_db)

            unique_smiles = list(dict.fromkeys(merged_smiles))
            if not unique_smiles:
                return title, None, neutral_mass, "候选池为空"

            spec_title_map[title] = {"mz": mz, "adduct": adduct}
            return title, unique_smiles, neutral_mass, None

        max_workers = min(8, max(2, len(spec_titles)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_load_for_title, title): title for title in spec_titles}
            for fut in as_completed(futures):
                title, smiles_list, neutral_mass, err = fut.result()
                if err:
                    failed_results[title] = {
                        "status": "failed",
                        "reason": err,
                        "top1": "",
                        "top10": [],
                        "top100": [],
                        "top100_score": [],
                        "result_top100": [],
                    }
                    continue
                spec_candidates[title] = smiles_list
                spec_neutral_mass_map[title] = float(neutral_mass)

        if not spec_candidates:
            final_results.update(failed_results)
            print("[错误] 所有谱图候选池为空或计算失败")
            sys.exit(1)

        all_unique_smiles = list({smi for lst in spec_candidates.values() for smi in lst})
        print(f"[信息] 当前任务共 {len(all_unique_smiles)} 个唯一 SMILES，开始统一编码")

        mol_embs, mol_smiles = encode_molecules(all_unique_smiles, model, cfg, DEVICE, BATCH_SIZE)
        if mol_embs is None:
            final_results.update(failed_results)
            print("[错误] 分子编码失败")
            sys.exit(1)

        smi_to_idx = {smi: i for i, smi in enumerate(mol_smiles)}
        norm_mol = F.normalize(mol_embs)
        title_to_index = {title: idx for idx, title in enumerate(spec_titles)}

        for title in spec_titles:
            if title in failed_results:
                final_results[title] = failed_results[title]
                continue

            smiles_list = spec_candidates.get(title, [])
            idx = [smi_to_idx[s] for s in smiles_list if s in smi_to_idx]
            if not idx:
                final_results[title] = {
                    "status": "failed",
                    "reason": "候选池为空",
                    "top1": "",
                    "top10": [],
                    "top100": [],
                    "top100_score": [],
                    "result_top100": [],
                }
                continue

            spec_idx = title_to_index[title]
            spec_vec = F.normalize(spec_embs[spec_idx:spec_idx+1])
            cand_emb = norm_mol[torch.tensor(idx, dtype=torch.long)]

            similarity = torch.mm(spec_vec, cand_emb.T)
            topk = min(100, cand_emb.shape[0])
            scores, indices = torch.topk(similarity, k=topk, dim=1, largest=True)
            top_smiles = [smiles_list[i] for i in indices[0].numpy()]
            top_scores = [float(score) for score in scores[0].numpy()]

            neutral_mass = spec_neutral_mass_map.get(title, 0.0)
            top_meta_map = load_top_smiles_metadata(top_smiles, selected_db_infos, neutral_mass, ppm_range)
            result_top100 = []
            for rank, (smi, score) in enumerate(zip(top_smiles, top_scores), start=1):
                meta = top_meta_map.get(smi) or _meta_aggregate_entry()
                generic_names = sorted(meta.get("generic_name") or [])
                database_names = sorted(meta.get("database_name") or [])
                database_ids = sorted(meta.get("database_id") or [])
                result_top100.append({
                    "rank": rank,
                    "score": score,
                    "smiles": smi,
                    "formula": meta.get("formula") or "",
                    "inchi_key": meta.get("inchi_key") or "",
                    "generic_name": ", ".join(generic_names),
                    "database_name": ", ".join(database_names),
                    "database_id": ", ".join(database_ids),
                    "generic_name_list": generic_names,
                    "database_name_list": database_names,
                    "database_id_list": database_ids,
                })

            spec_info = spec_title_map.get(title) or _merge_title_info(title)
            if spec_info:
                spec_title_map[title] = spec_info

            final_results[title] = {
                "true_smiles": spec_true_smiles[spec_idx],
                "top1": top_smiles[0] if top_smiles else "",
                "top10": top_smiles[:10] if len(top_smiles) >= 10 else top_smiles,
                "top100": top_smiles,
                "top100_score": top_scores,
                "result_top100": result_top100,
                "status": "success",
                "ppm": ppm_range,
                "candidate_count": len(smiles_list),
                "adduct_original": spec_info.get("adduct", "Unknown") if spec_info else "Unknown",
                "adduct_normalized": normalize_adduct(spec_info.get("adduct", "Unknown") if spec_info else "Unknown"),
                "databases": selected_databases,
            }
    
    elif search_type == "custom":
        # 自定义模式：全局共用候选池
        print(f"[信息] 使用{search_type}模式，全局共用候选池")
        
        # 加载共用候选池（仅加载一次）
        if custom_smiles_list is not None:
            smiles_list = [s.strip() for s in custom_smiles_list if isinstance(s, str) and s.strip()]
            smiles_list = list(dict.fromkeys(smiles_list))
            print(f"[信息] 从内存候选池加载 {len(smiles_list)} 个分子")
        else:
            smiles_list = load_shared_smiles_pool(shared_smiles_txt_path)

        if not smiles_list:
            print("[错误] 共用候选池为空")
            sys.exit(1)
        
        # 编码候选分子（仅编码一次）
        mol_embs, mol_smiles = encode_molecules(smiles_list, model, cfg, DEVICE, BATCH_SIZE)
        if mol_embs is None:
            print("[错误] 分子编码失败")
            sys.exit(1)
        
        # 对所有谱图执行检索
        similarity = torch.mm(F.normalize(spec_embs), F.normalize(mol_embs).T)
        topk = min(100, len(mol_smiles))
        scores, indices = torch.topk(similarity, k=topk, dim=1, largest=True)
        
        # 整理结果
        for i, title in enumerate(spec_titles):
            top_smiles = [mol_smiles[idx] for idx in indices[i].numpy()]
            top_scores = [float(score) for score in scores[i].numpy()]
            
            result_top100 = [
                {
                    "rank": rank,
                    "score": score,
                    "smiles": smi,
                    "formula": "",
                    "inchi_key": "",
                    "generic_name": "",
                    "database_name": "",
                    "database_id": "",
                    "generic_name_list": [],
                    "database_name_list": [],
                    "database_id_list": [],
                }
                for rank, (smi, score) in enumerate(zip(top_smiles, top_scores), start=1)
            ]

            final_results[title] = {
                "true_smiles": spec_true_smiles[i],
                "top1": top_smiles[0] if top_smiles else "",
                "top10": top_smiles[:10] if len(top_smiles)>=10 else top_smiles,
                "top100": top_smiles,
                "top100_score": top_scores,
                "result_top100": result_top100,
                "status": "success" if top_smiles else "failed",
                "candidate_pool_mode": search_type,
                "candidate_count": len(smiles_list)
            }
    
    else:
        print(f"[错误] 不支持的候选池模式：{search_type}")
        sys.exit(1)

    # 6. 将检索结果写入统计信息JSON
    print("\n=== 写入检索结果 ===")
    stats_root = stats_data["碎裂树文件统计"]["有效碎裂树根节点信息"]
    for i, spec_item in enumerate(stats_root):
        title = spec_item.get("title")
        merged_info = spec_title_map.get(title) or _merge_title_info(title)
        if merged_info:
            stats_root[i]["mz"] = merged_info.get("mz", spec_item.get("mz", 0.0))
            stats_root[i]["adduct"] = merged_info.get("adduct", spec_item.get("adduct", ""))
            spec_title_map[title] = merged_info
        if title in final_results:
            stats_root[i]["检索结果"] = final_results[title]
        else:
            stats_root[i]["检索结果"] = {
                "status": "not_processed",
                "reason": "谱图未参与检索"
            }
    
    # 7. 将结果回写到传入的 statas.json
    write_path = os.path.abspath(statas_json_path)
    os.makedirs(os.path.dirname(write_path), exist_ok=True)
    temp_write_path = write_path + ".tmp"
    with open(temp_write_path, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=4)
    os.replace(temp_write_path, write_path)

    print(f"[信息] 检索结果已回写至：{write_path}")
    
    # 8. 打印结果摘要
    print("\n=== 检索结果摘要 ===")
    success_count = sum([1 for v in final_results.values() if v["status"] == "success"])
    total_count = len(final_results)
    print(f"总谱图数：{total_count}")
    print(f"成功检索：{success_count}")
    print(f"失败数：{total_count - success_count}")
    
    if success_count > 0:
        top1_correct = sum([
            1 for v in final_results.values() 
            if v["status"] == "success" and v["top1"] == v["true_smiles"] and v["true_smiles"] != "unknown"
        ])
        valid_count = sum([
            1 for v in final_results.values() 
            if v["status"] == "success" and v["true_smiles"] != "unknown"
        ])
        if valid_count > 0:
            accuracy = (top1_correct / valid_count) * 100
            print(f"Top1准确率（有效谱图）：{top1_correct}/{valid_count} ({accuracy:.2f}%)")

    return {
        "statas_path": write_path,
        "total_count": total_count,
        "success_count": success_count,
        "failed_count": total_count - success_count,
    }

if __name__ == "__main__":
    main()