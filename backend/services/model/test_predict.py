import os
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import time
from collections import defaultdict

# 第三方库
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

PROJ_ROOT = "/root/web/backend/services/model"
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)  # 插入到sys.path最前面，优先查找
# 项目内导入
from modules_advanced import FragSimiModel
from dataset import (
    parse_mgf_file, parse_ms2_from_mgf, preprocess_spectrum,
    get_adduct_map
)
from torch_geometric.data import Data

# ======================== 全局显式参数配置（直接修改此处！）========================
# ======================== 1. 模型输入文件（MGF+碎裂树JSON）========================
# 单个谱图MGF文件路径（模型需要的谱图峰数据）
MGF_FILE_PATH = "/home/nfs05/wuzt/AI+/ZZZ_grn/web/backend/temp/task_2b9c524b7f2b4f4897ef8ff59e55790d/valid_pairs_spectra.mgf"  

# 单个谱图碎裂树JSON文件路径（模型需要的碎裂树数据）
FRAG_TREE_JSON_PATH = "/home/nfs05/wuzt/AI+/ZZZ_grn/web/backend/temp/task_2b9c524b7f2b4f4897ef8ff59e55790d/valid_pairs_fragtrees.json"  

# ======================== 2. 统计信息JSON（含有效碎裂树根节点信息）========================
# 统计信息JSON文件（含"有效碎裂树根节点信息"，用于读取title/adduct/mz，写入检索结果）
STATS_JSON_PATH = "/home/nfs05/wuzt/AI+/ZZZ_grn/web/backend/temp/task_2b9c524b7f2b4f4897ef8ff59e55790d/statas.json"  

# ======================== 3. 模型与输出配置 ========================
# 模型权重文件
MODEL_WEIGHT_PATH = "/home/nfs05/wuzt/AI+/4C/4C_v1/model/pos/model_epoch-36_vloss-0.1429.pth"  

# 最终输出JSON文件（整合统计信息+检索结果）
OUTPUT_JSON_PATH = "/home/nfs05/wuzt/AI+/ZZZ_grn/web/backend/temp/final_statas.json"  

# ======================== 4. 候选池模式配置 ========================
# 候选池模式（三选一：pubchem / custom / self）
CANDIDATE_POOL_MODE = "pubchem"  # pubchem: PubChem+ppm | custom: 自定义txt | self: 自选txt

# PubChem模式专属配置（仅CANDIDATE_POOL_MODE=pubchem时生效）
PUBCHEM_PPM = 1.0  # ppm偏差值
PUBCHEM_PARQUET_PATH = "/home/nfs06/wuzt/wmr/pubchem_final_with_formula.parquet"  # PubChem的parquet文件路径

# 自定义/自选模式专属配置（仅CANDIDATE_POOL_MODE=custom/self时生效）
SHARED_SMILES_TXT_PATH = "/home/nfs05/wuzt/AI+/ZZZ_grn/src/files/shared_smiles.txt"  # 全局共用SMILES列表

# ======================== 5. 运行配置 ========================
BATCH_SIZE = 256  # 批次大小
SEED = 42  # 随机种子
EVAL_MODE = "full"  # 评测模式：full/one-vs-rand
MISSING_TREE_POLICY = "discard"  # 无碎裂树处理策略：discard/placeholder
EMA_MODE = "auto"  # EMA权重使用：auto/on/off
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 计算设备

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
    "M+H":        -PROTON,
    "[M + H]+":   -PROTON,
    "M-H2O+H":    -PROTON + H2O,
    "[M - H2O + H]+": -PROTON + H2O,
    "M+ACN+H":    -(ACN + PROTON),
    "[M + ACN + H]+": -(ACN + PROTON),
    "M+FA+H":     -(FA + PROTON),
    "[M + FA + H]+": -(FA + PROTON),
    "M+Na":       -NA,
    "[M + Na]+":  -NA,
    "M+K":        -K,
    "[M + K]+":   -K,
    "M+NH4":      -NH4,
    "[M + NH4]+": -NH4,
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

# ======================== PubChem候选池工具函数 ========================
def load_global_exactmass(parquet_path: str) -> tuple[np.ndarray, pq.ParquetFile]:
    """仅加载全局exactmass数组和ParquetFile对象"""
    arr = ds.dataset(parquet_path, format="parquet").to_table(columns=['exactmass'])['exactmass']
    exactmasses = arr.to_numpy(zero_copy_only=False).astype(np.float64)
    
    if exactmasses.size > 1 and not np.all(exactmasses[:-1] <= exactmasses[1:]):
        print("[警告] parquet 并非严格非降序，请确认已按 exactmass 排序！")
    
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

def load_smiles_in_ppm_range(neutral_mass: float, ppm: float, 
                            exactmasses: np.ndarray, parquet_file: pq.ParquetFile) -> list:
    """加载指定ppm范围内的SMILES列表（单个谱图专属候选池）"""
    # 1. 计算ppm质量范围对应的全局索引
    lo = neutral_mass - neutral_mass * ppm * 1e-6
    hi = neutral_mass + neutral_mass * ppm * 1e-6
    global_start = int(np.searchsorted(exactmasses, lo, side='left'))
    global_end = int(np.searchsorted(exactmasses, hi, side='right'))
    
    if global_start >= global_end:
        print(f"[警告] {ppm}ppm范围内无匹配分子（中性质量：{neutral_mass:.6f}，范围：{lo:.6f} ~ {hi:.6f}）")
        return []
    
    # 2. 筛选需扫描的行组
    row_groups_to_scan = []
    current_rg_start = 0  # 当前行组的起始全局索引
    for rg_idx in range(parquet_file.num_row_groups):
        rg_meta = parquet_file.metadata.row_group(rg_idx)
        rg_row_count = rg_meta.num_rows
        current_rg_end = current_rg_start + rg_row_count
        
        # 判断行组是否与目标范围重叠
        if not (current_rg_end <= global_start or current_rg_start >= global_end):
            slice_start = max(global_start - current_rg_start, 0)
            slice_end = min(global_end - current_rg_start, rg_row_count)
            row_groups_to_scan.append((rg_idx, slice_start, slice_end))
        
        current_rg_start = current_rg_end
        if current_rg_start >= global_end:
            break
    
    # 3. 读取行组并提取SMILES
    smiles_list = []
    for rg_idx, slice_start, slice_end in row_groups_to_scan:
        rg_table = parquet_file.read_row_group(
            rg_idx, 
            columns=['rdkit_smiles']
        )
        sliced_table = rg_table.slice(offset=slice_start, length=slice_end - slice_start)
        smiles = sliced_table['rdkit_smiles'].to_pylist()
        smiles_list.extend([s for s in smiles if s])
    
    # 去重并返回
    unique_smiles = list(set(smiles_list))
    print(f"[信息] {ppm}ppm范围内找到 {len(unique_smiles)} 个候选分子（中性质量：{neutral_mass:.6f}）")
    return unique_smiles

def load_shared_smiles_pool(txt_path: str) -> list:
    """加载自定义/自选模式的全局共用SMILES候选池"""
    if not os.path.exists(txt_path):
        print(f"[错误] 共用SMILES文件不存在: {txt_path}")
        sys.exit(1)
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    # 去重
    unique_smiles = list(set(smiles_list))
    print(f"[信息] 加载共用候选池：{len(unique_smiles)} 个分子")
    return unique_smiles

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
        mol_cfg = self.cfg.model.mol_encoder
        if 'fp' in mol_cfg.type:
            try:
                from utils import mol_fp_encoder
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
            edge_dim = 14 if self.enhanced_tree else 13
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

# ======================== 核心检索函数 ========================
def encode_molecules(smiles_list, model, cfg, device, batch_size=256):
    """编码分子列表为向量"""
    dataset = MolDataset(smiles_list, cfg)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_mol, pin_memory=True)
    
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
def main():
    # 1. 加载统计信息JSON
    print("=== 加载统计信息JSON ===")
    if not os.path.exists(STATS_JSON_PATH):
        print(f"[错误] 统计信息JSON不存在: {STATS_JSON_PATH}")
        sys.exit(1)
    
    with open(STATS_JSON_PATH, 'r', encoding='utf-8') as f:
        stats_data = json.load(f)
    
    if "碎裂树文件统计" not in stats_data or "有效碎裂树根节点信息" not in stats_data["碎裂树文件统计"]:
        print(f"[错误] 统计信息JSON缺少'碎裂树文件统计/有效碎裂树根节点信息'字段")
        sys.exit(1)

    # 从碎裂树文件统计下读取有效碎裂树根节点信息
    spec_stats_list = stats_data["碎裂树文件统计"]["有效碎裂树根节点信息"]
    spec_title_map = {item["title"]: item for item in spec_stats_list}
    print(f"[信息] 统计信息JSON中加载 {len(spec_stats_list)} 个谱图信息")

    # 2. 加载模型
    print("\n=== 加载模型 ===")
    try:
        checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"[错误] 加载模型失败: {e}")
        sys.exit(1)
    
    # 加载配置
    if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
        from config import ConfigDict
        def _to_cfgdict(d):
            if isinstance(d, dict):
                return ConfigDict({k: _to_cfgdict(v) for k, v in d.items()})
            return d
        try:
            from config import CFG
            CFG.clear()
            CFG.update(_to_cfgdict(checkpoint['config']))
        except ImportError:
            print("错误：未找到config模块中的CFG/ConfigDict！")
            sys.exit(1)
    else:
        print("错误：模型文件中未找到内置配置！")
        sys.exit(1)
    
    # 初始化模型
    model = FragSimiModel(CFG, enable_compile=False).to(DEVICE).eval()
    
    # 加载权重
    try:
        from compatibility import load_checkpoint_with_compat
        _ = load_checkpoint_with_compat(
            model,
            checkpoint,
            cfg=CFG,
            ema=EMA_MODE,
            verbose=True
        )
    except ImportError:
        print("错误：未找到compatibility模块中的load_checkpoint_with_compat函数！")
        sys.exit(1)

    # 3. 加载谱图数据集（模型输入）
    print("\n=== 加载谱图数据 ===")
    try:
        ms_dataset = MSDataset(MGF_FILE_PATH, FRAG_TREE_JSON_PATH, CFG, missing_tree_policy=MISSING_TREE_POLICY)
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
    
    # 5. 处理候选池并执行检索
    print("\n=== 处理候选池 ===")
    final_results = {}
    
    if CANDIDATE_POOL_MODE == "pubchem":
        # PubChem模式：每个谱图专属候选池
        print(f"[信息] 使用PubChem+{PUBCHEM_PPM}ppm模式，每个谱图生成专属候选池")
        
        # 加载PubChem全局数据
        exactmasses, parquet_file = load_global_exactmass(PUBCHEM_PARQUET_PATH)
        
        for title in spec_titles:
            # 获取谱图的adduct和mz
            if title not in spec_title_map:
                print(f"[警告] 统计信息中未找到谱图 {title}，跳过")
                continue
            
            spec_info = spec_title_map[title]
            adduct = spec_info["adduct"]
            mz = spec_info["mz"]
            
            # 计算中性质量（自动标准化加合物字符串）
            neutral_mass = mz_to_neutral_mass(mz, adduct)
            if neutral_mass is None:
                final_results[title] = {
                    "status": "failed",
                    "reason": "无法计算中性质量",
                    "top1": "",
                    "top10": [],
                    "top100": [],
                    "top100_score": []
                }
                continue
            
            # 加载该谱图的专属候选池
            smiles_list = load_smiles_in_ppm_range(neutral_mass, PUBCHEM_PPM, exactmasses, parquet_file)
            if not smiles_list:
                final_results[title] = {
                    "status": "failed",
                    "reason": "候选池为空",
                    "top1": "",
                    "top10": [],
                    "top100": [],
                    "top100_score": []
                }
                continue
            
            # 编码候选分子
            mol_embs, mol_smiles = encode_molecules(smiles_list, model, CFG, DEVICE, BATCH_SIZE)
            if mol_embs is None:
                final_results[title] = {
                    "status": "failed",
                    "reason": "分子编码失败",
                    "top1": "",
                    "top10": [],
                    "top100": [],
                    "top100_score": []
                }
                continue
            
            # 找到该谱图的编码向量
            spec_idx = spec_titles.index(title)
            single_spec_emb = spec_embs[spec_idx:spec_idx+1]
            
            # 执行检索
            similarity = torch.mm(F.normalize(single_spec_emb), F.normalize(mol_embs).T)
            topk = min(100, len(mol_smiles))
            scores, indices = torch.topk(similarity, k=topk, dim=1, largest=True)
            
            # 整理结果
            top_smiles = [mol_smiles[idx] for idx in indices[0].numpy()]
            top_scores = [float(score) for score in scores[0].numpy()]
            
            final_results[title] = {
                "true_smiles": spec_true_smiles[spec_idx],
                "top1": top_smiles[0] if top_smiles else "",
                "top10": top_smiles[:10] if len(top_smiles)>=10 else top_smiles,
                "top100": top_smiles,
                "top100_score": top_scores,
                "status": "success",
                "neutral_mass": neutral_mass,
                "ppm": PUBCHEM_PPM,
                "candidate_count": len(smiles_list),
                "adduct_original": adduct,
                "adduct_normalized": normalize_adduct(adduct)
            }
    
    elif CANDIDATE_POOL_MODE in ["custom", "self"]:
        # 自定义/自选模式：全局共用候选池
        print(f"[信息] 使用{CANDIDATE_POOL_MODE}模式，全局共用候选池")
        
        # 加载共用候选池（仅加载一次）
        smiles_list = load_shared_smiles_pool(SHARED_SMILES_TXT_PATH)
        if not smiles_list:
            print("[错误] 共用候选池为空")
            sys.exit(1)
        
        # 编码候选分子（仅编码一次）
        mol_embs, mol_smiles = encode_molecules(smiles_list, model, CFG, DEVICE, BATCH_SIZE)
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
            
            final_results[title] = {
                "true_smiles": spec_true_smiles[i],
                "top1": top_smiles[0] if top_smiles else "",
                "top10": top_smiles[:10] if len(top_smiles)>=10 else top_smiles,
                "top100": top_smiles,
                "top100_score": top_scores,
                "status": "success" if top_smiles else "failed",
                "candidate_pool_mode": CANDIDATE_POOL_MODE,
                "candidate_count": len(smiles_list)
            }
    
    else:
        print(f"[错误] 不支持的候选池模式：{CANDIDATE_POOL_MODE}")
        sys.exit(1)

    # 6. 将检索结果写入统计信息JSON
    print("\n=== 写入检索结果 ===")
    stats_root = stats_data["碎裂树文件统计"]["有效碎裂树根节点信息"]
    for i, spec_item in enumerate(stats_root):
        title = spec_item["title"]
        if title in final_results:
            stats_root[i]["检索结果"] = final_results[title]
        else:
            stats_root[i]["检索结果"] = {
                "status": "not_processed",
                "reason": "谱图未参与检索"
            }
    
    # 7. 保存最终结果
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=4)
    
    print(f"[信息] 最终结果已保存至：{OUTPUT_JSON_PATH}")
    
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

if __name__ == "__main__":
    main()