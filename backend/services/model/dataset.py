
import os
import logging
import json
import torch
import re
import io
import lmdb
from torch_geometric.data import Data
from typing import List, Tuple, Dict

import numpy as np
import glob

from FragmentationTreeEncoder import FragmentTreeProcessor
ADDUCT_MAP= {
    "M+": 0, "M+H": 1, "M-H2O+H": 2, "M+ACN+H": 3,
    "M+FA+H": 4, "M+Na": 5, "M+K": 6, "M+NH4": 7,
    "Unknown": 8
}
# --- NEW: two canonical maps ---
_POS_ADDUCT_MAP = {
    "M+": 0, "M+H": 1, "M-H2O+H": 2, "M+ACN+H": 3,
    "M+FA+H": 4, "M+Na": 5, "M+K": 6, "M+NH4": 7,
    "Unknown": 8
}

_NEG_ADDUCT_MAP = {
    "M+Br": 0, "M+Cl": 1, "M+FA-H": 2, "M+Hac-H": 3,
    "M+K-2H": 4, "M+Na-2H": 5, "M+CH3OH-H": 6, "M-C6H10O4-H": 7,
    "M-C6H10O5-H": 8, "M-H2O-H": 9, "M-H": 10, "M-NH3-H": 11,
    "M-C6H8O6-H": 12,
    "Unknown": 13
}

def _get_ion_mode(cfg) -> str:
    if cfg is None:
        return 'positive'
    try:
        ms_enc = getattr(getattr(cfg, 'model', {}), 'ms_encoder', {})
        mode = getattr(ms_enc, 'ion_mode', 'positive')
        mode = str(mode).lower()
        return 'negative' if mode == 'negative' else 'positive'
    except Exception:
        return 'positive'

def get_adduct_map(cfg):
    return _NEG_ADDUCT_MAP if _get_ion_mode(cfg) == 'negative' else _POS_ADDUCT_MAP

def parse_adduct_string(raw_adduct_str: str, cfg) -> str:
    """
    Normalize adduct text from MGF into one of the keys of the active ADDUCT_MAP.
    Handles [ ... ]+ (positive) vs [ ... ]- (negative). Defaults to 'Unknown'.
    """
    if raw_adduct_str is None:
        return "Unknown"
    content = raw_adduct_str.strip()
    ion_mode = _get_ion_mode(cfg)

    if ion_mode == 'negative':
        if content.startswith('[') and content.endswith(']-'):
            content = content[1:-2]
        standardized = 'M-' if content == 'M' else content
        lst = list(_NEG_ADDUCT_MAP.keys())
        return standardized if standardized in lst else "Unknown"
    else:
        if content.startswith('[') and content.endswith(']+'):
            content = content[1:-2]
        standardized = 'M+' if content == 'M' else content
        lst = list(_POS_ADDUCT_MAP.keys())
        return standardized if standardized in lst else "Unknown"


def parse_mgf_file(mgf_path: str, cfg=None) -> List[Dict]:
    """解析 MGF 文件以提取 MS1 (前体) 信息"""
    with open(mgf_path, 'r') as f:
        mgf_content = f.read()
    ion_blocks = re.findall(r'BEGIN IONS(.*?)END IONS', mgf_content, re.DOTALL)
    data = []
    for block in ion_blocks:
        lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
        title, smiles, precursor_mz, ms_level, adduct = None, None, None, None, None
        for line in lines:
            if line.startswith('TITLE='):
                title = line.split('=', 1)[1].strip()
            elif line.startswith('SMILES:'):
                smiles = line.split(':', 1)[-1].strip() # 必须用 -1
            elif line.startswith('SMILES='):
                smiles = line.split('=', 1)[-1].strip()
            elif line.startswith('PEPMASS='):
                precursor_mz = float(line.split('=', 1)[1].split()[0])
            elif line.startswith('MSLEVEL='):
                ms_level = int(line.split('=', 1)[1])
            elif line.startswith('ADDUCTIONNAME='):
                adduct = parse_adduct_string(line.split('=', 1)[1].strip(), cfg)

        
        if ms_level == 1 and precursor_mz is not None and title is not None:
            data.append({
                'title': title, 'smiles': smiles, 'precursor_mz': precursor_mz,
                'ms_level': ms_level, 'adduct': adduct if adduct else "Unknown"
            })
    return data

def parse_ms2_from_mgf(mgf_path: str) -> List[Dict]:
    """
    解析 MGF 文件以提取 MS2 (碎片) 谱图信息
    """
    with open(mgf_path, 'r') as f:
        mgf_content = f.read()
    ion_blocks = re.findall(r'BEGIN IONS(.*?)END IONS', mgf_content, re.DOTALL)
    ms2_results = []
    for block in ion_blocks:
        lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
        title, precursor_mz, ms_level = None, None, None
        peaks = []
        for line in lines:
            if line.startswith('TITLE='):
                title = line.split('=', 1)[1].strip()
            elif line.startswith('PEPMASS='):
                precursor_mz = float(line.split('=', 1)[1].split()[0])
            elif line.startswith('MSLEVEL='):
                ms_level = int(line.split('=', 1)[1])
            elif re.match(r'^[\d.]+[\s\t]+[\d.]+$', line):
                mz, intensity = map(float, line.split())
                peaks.append((mz, intensity))
        
        if ms_level == 2 and precursor_mz is not None and title is not None:
            ms2_results.append({
                'title': title, 'precursor_mz': precursor_mz, 'ms2_peaks': peaks
            })
    return ms2_results


def preprocess_spectrum(
    peaks: List[Tuple[float, float]],
    dim: int = 100,
    order: str = "intensity_desc",  # "intensity_desc" (old) or "topk_then_mz_asc" (new)
) -> torch.Tensor:
    """
    预处理质谱峰数据，生成固定长度的 [2, dim] 张量。
    "intensity_desc": 选前 dim 个后按强度从大到小（原行为）
    "topk_then_mz_asc": 先按强度取 Top-K，再按 m/z 升序重排（实验 3.1）
    """
    if not peaks:
        return torch.zeros((2, dim), dtype=torch.float32)

    peaks_np = np.array(peaks, dtype=np.float32)  # [n, 2]
    mz_all = peaks_np[:, 0]
    inten_all = peaks_np[:, 1]

    inten = np.sqrt(inten_all)
    max_i = np.max(inten)
    inten = inten if max_i == 0 else inten / max_i

    # 先按强度取 Top-K
    idx_desc = inten.argsort()[::-1]
    top_idx = idx_desc[:dim]
    mz_top = mz_all[top_idx]
    inten_top = inten[top_idx]

    # 然后根据 order 决定最终顺序
    if order == "topk_then_mz_asc":
        order_idx = np.argsort(mz_top)  # m/z 升序
        mz_top = mz_top[order_idx]
        inten_top = inten_top[order_idx]
    else:
        pass

    n = len(mz_top)
    if n < dim:
        mz_top = np.pad(mz_top, (0, dim - n), constant_values=0.0)
        inten_top = np.pad(inten_top, (0, dim - n), constant_values=0.0)

    return torch.tensor(np.stack([mz_top, inten_top]), dtype=torch.float32)


def calc_feats(smi: str, ms2_entry: dict, frag_tree_entry: dict, cfg):
    """
    根据给定的SMILES、MS2谱和碎裂树数据，计算所有特征。
    """
    import utils
    features = {}

    sdim   = getattr(getattr(cfg.model, 'ms_encoder', {}), 'spectrum_dim', 100)
    sorder = getattr(getattr(cfg.model, 'ms_encoder', {}), 'spectrum_order', 'intensity_desc')
    features['spec_tensor'] = preprocess_spectrum(ms2_entry['ms2_peaks'], dim=sdim, order=sorder)

    features['precursor_mz'] = ms2_entry['precursor_mz']
    if frag_tree_entry and 'frag_tree' in frag_tree_entry:
        enhanced = bool(getattr(getattr(cfg.model, 'ms_encoder', {}), 'tree_encoder', {}).get('enhanced_features', False))
        pyg_data = FragmentTreeProcessor.json_to_pyg(
            {'frag_tree': frag_tree_entry['frag_tree']},
            edge_features=True,
            enhanced=enhanced
        )
        features['pyg_data'] = pyg_data

    mol_cfg = cfg.model.mol_encoder
    if 'fp' in mol_cfg.type:
        fp = utils.mol_fp_encoder(smi, tp=mol_cfg.fp.type, nbits=mol_cfg.fp.nbits)
        if fp is None: return None
        features['mol_fps'] = fp
    if 'gnn' in mol_cfg.type:
        graph_features = utils.mol_graph_featurizer(smi)
        if not graph_features: return None
        features.update(graph_features)

    if 'mol_fps' not in features and 'V' not in features: return None

    features['true_smiles'] = smi
    return features




class PreprocessedDataset(torch.utils.data.Dataset):
    """
      * 主进程不持久打开 LMDB（只用临时 env 读取 __keys__，立即关闭）
      * 每个进程/worker 内部懒打开 env（self._env），每次 __getitem__ 用短事务 + buffers=False
    """
    def __init__(self, processed_dir: str):
        self.processed_dir = processed_dir
        self.backend = None  # "lmdb" or "pt"
        self.file_paths = []
        self.lmdb_path = None
        self._env = None     # 不在主进程持久打开
        self._keys = []

        # 1) If 'processed_dir' itself is an LMDB folder (contains data.mdb), use it
        if os.path.isdir(processed_dir) and os.path.isfile(os.path.join(processed_dir, "data.mdb")):
            self.backend = "lmdb"
            self.lmdb_path = processed_dir
        else:
            # 2) Try '<parent>/<split>.lmdb' sibling
            parent = os.path.dirname(processed_dir.rstrip("/"))
            split = os.path.basename(processed_dir.rstrip("/"))
            candidate = os.path.join(parent, f"{split}.lmdb")
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "data.mdb")):
                self.backend = "lmdb"
                self.lmdb_path = candidate

        if self.backend == "lmdb":
            # ★ 只用临时 env 读取 __keys__，读完立刻关闭；主进程不保留 env/txn
            tmp_env = lmdb.open(
                self.lmdb_path,
                readonly=True, lock=False, readahead=False,
                max_readers=4096, subdir=True
            )
            with tmp_env.begin(write=False, buffers=False) as txn:  # buffers=False -> bytes
                raw = txn.get(b"__keys__")
                if raw is None:
                    raise FileNotFoundError(f"LMDB at {self.lmdb_path} missing '__keys__' index.")
                self._keys = json.loads(raw.decode("utf-8"))
            tmp_env.close()

            logging.info(f"从 {self.lmdb_path} 以 LMDB 加载 {len(self._keys)} 个样本。")
            return

        # 3) Fall back to old '*.pt' files
        if not os.path.isdir(processed_dir):
            raise FileNotFoundError(f"预处理数据目录不存在: {processed_dir} "
                                    f"(也未发现同名 .lmdb 文件夹)")
        self.file_paths = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
        logging.info(f"从 {processed_dir} 找到 {len(self.file_paths)} 个预处理样本（.pt 模式）。")
        self.backend = "pt"

    # ---------- LMDB helpers ----------
    def _open_env_if_needed(self):
        """在当前进程/worker 内懒打开只读 env；不创建长期 txn。"""
        if self._env is None and self.lmdb_path is not None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,       # 只读不加锁
                readahead=False,  # NFS/并行更稳
                max_readers=4096,
                subdir=True
            )

    def __getstate__(self):
        """
        使得 DataLoader workers 在各自进程里重新打开 env。
        注意：fork 模式下不一定调用到；所以 __getitem__ 仍然要做懒打开。
        """
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 不在这里自动打开 env，保持懒加载（兼容 fork 与 spawn）
        self._env = None

    def __len__(self):
        if self.backend == "lmdb":
            return len(self._keys)
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.backend == "lmdb":
            self._open_env_if_needed()  # 每个进程/worker 首次使用时打开 env
            key = self._keys[idx].encode("utf-8")
            # ★ 短事务 + buffers=False，返回 bytes，避免 memoryview 跨进程生命周期问题
            with self._env.begin(write=False, buffers=False) as txn:
                buf = txn.get(key)
                if buf is None:
                    raise KeyError(f"Key not found in LMDB: {key!r}")
            return torch.load(io.BytesIO(buf), map_location="cpu", weights_only=False)

        # old .pt path
        file_path = self.file_paths[idx]
        try:
            return torch.load(file_path, weights_only=False, map_location="cpu")
        except Exception as e:
            logging.error(f"加载文件 {file_path} 失败: {e}")
            return None

    def __del__(self):
        try:
            if self._env is not None:
                self._env.close()
                self._env = None
        except Exception:
            pass