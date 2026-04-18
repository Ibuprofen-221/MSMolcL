import os
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import time

from config import CFG
from modules import FragSimiModel
from dataset import (
    parse_mgf_file, parse_ms2_from_mgf, preprocess_spectrum,
    get_adduct_map
)
from utils_advanced import mol_fp_encoder, mol_graph_featurizer, pad_V, pad_A
from torch_geometric.data import Data, Batch
from FragmentationTreeEncoder import FragmentTreeProcessor

from compatibility import load_checkpoint_with_compat

# === 10ppm: RDKit 计算精确质量（单同位素）
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try: torch.set_float32_matmul_precision('highest')
except: pass

# === 10ppm: 正离子常量（与 collect_pubchem_cand.py 一致）
ELECTRON = 5.485_799_090_441e-4
PROTON   = 1.0072764665789
H2O      = 18.01056468403
ACN      = 41.02654910112
FA       = 46.00547930360
NA       = 22.98922070209
K        = 38.96315790649
NH4      = 18.03382555344
ADDUCT_TO_NEUTRAL_DELTA = {
    "M+":         +ELECTRON,
    "M+H":        -PROTON,
    "M-H2O+H":    -PROTON + H2O,
    "M+ACN+H":    -(ACN + PROTON),
    "M+FA+H":     -(FA + PROTON),
    "M+Na":       -NA,
    "M+K":        -K,
    "M+NH4":      -NH4,
    "Unknown":     None,
}

def ppm_window(mass: float, ppm: float) -> tuple[float, float]:
    tol = mass * (ppm * 1e-6)
    return (mass - tol, mass + tol)

def smiles_exact_mass(smi: str) -> float:
    """RDKit 单同位素精确质量；失败返回 NaN。"""
    m = Chem.MolFromSmiles(smi)
    if m is None: return float('nan')
    return float(rdMolDescriptors.CalcExactMolWt(m))

def mz_to_neutral_mass_pos(mz: float, adduct: str | None, fallback_smiles_mass: float) -> float:
    """仅正离子：根据加合物把 m/z 映射为中性质量；若加合物未知/不在映射表，按要求用该查询的 SMILES 质量作为中性质量。"""
    if adduct is None: 
        return float(fallback_smiles_mass)
    delta = ADDUCT_TO_NEUTRAL_DELTA.get(str(adduct), None)
    return float(mz + delta) if delta is not None else float(fallback_smiles_mass)

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
            fp = mol_fp_encoder(smi, tp=mol_cfg.fp.type, nbits=mol_cfg.fp.nbits)
            if fp is None: return None
            features['mol_fps'] = fp
        if 'gnn' in mol_cfg.type:
            graph_features = mol_graph_featurizer(smi)
            if not graph_features: return None
            features.update(graph_features)
        
        features['_row_idx'] = idx
        return features

class MSDataset(Dataset):
    """用于加载MGF文件作为查询的数据集"""
    def __init__(self, mgf_files, data_source_dir, frag_tree_dir, cfg, missing_tree_policy='discard'):
        self.mgf_files = mgf_files
        self.data_source_dir = data_source_dir
        self.frag_tree_dir = frag_tree_dir
        self.cfg = cfg
        self.missing_tree_policy = missing_tree_policy
        self.processor = FragmentTreeProcessor()
        self.queries = self._load_queries()
        fusion = getattr(getattr(cfg.model, 'ms_encoder', {}), 'fusion', None)
        if fusion is None:
            legacy_ft_mode = getattr(getattr(cfg.model.ms_encoder, 'frag_tree', {}), 'mode', None)
            fusion = 'clerms-only' if legacy_ft_mode == 'disable' else 'concat'
        self.need_tree = fusion in ('xattn', 'concat', 'tree-only')
        self.enhanced_tree = bool(getattr(getattr(cfg.model, 'ms_encoder', {}), 'tree_encoder', {}).get('enhanced_features', False))
        self.adduct_map = get_adduct_map(cfg)

    def _load_queries(self):
        all_queries = []
        for mgf_file in tqdm(self.mgf_files, desc="Loading Query Spectra"):
            path = os.path.join(self.data_source_dir, mgf_file)
            
            base_name = os.path.splitext(os.path.basename(mgf_file))[0]
            frag_tree_path = os.path.join(self.frag_tree_dir, f"{base_name}.json")
            frag_tree_db = {}
            if os.path.exists(frag_tree_path):
                try:
                    with open(frag_tree_path, 'r') as f:
                        frag_tree_db = json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load frag tree {frag_tree_path}: {e}")
            
            ms1_data = parse_mgf_file(path, self.cfg)
            ms2_data = parse_ms2_from_mgf(path)
            ms2_dict = {item['title']: item for item in ms2_data}

            for ms1_entry in ms1_data:
                title = ms1_entry.get('title')
                ms2_entry = ms2_dict.get(title)
                if ms1_entry.get('smiles') and ms2_entry:
                    frag_tree_entry = frag_tree_db.get(title, {})
                    query_item = {**ms1_entry, **ms2_entry, 'frag_tree_entry': frag_tree_entry, 'source_mgf': path}
                    all_queries.append(query_item)
        return all_queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        
        frag_tree_entry = query.get('frag_tree_entry', {})
        pyg_data = None

        if frag_tree_entry and 'frag_tree' in frag_tree_entry:
            try:
                processed_data = self.processor.json_to_pyg({'frag_tree': frag_tree_entry['frag_tree']}, 
                                                            edge_features=True, enhanced=self.enhanced_tree)
                if processed_data.num_nodes > 0:
                    pyg_data = processed_data
                else:
                    print(f"Warning: Processed frag_tree for {query['title']} resulted in an empty graph. Using a placeholder.")
            except Exception as e:
                print(f"Warning: Failed to process frag_tree for {query['title']}: {e}. Using a placeholder.")
        
        if pyg_data is None:
            if self.missing_tree_policy == 'discard' and self.need_tree:
                print(f"Info: Discarding sample '{query['title']}' due to missing tree and policy='discard'.")
                return None
            if self.need_tree:
                if self.missing_tree_policy == 'placeholder':
                    print(f"Info: Using empty graph placeholder for '{query['title']}'.")

            node_dim = 19 if self.enhanced_tree else 16
            edge_dim = 15 if self.enhanced_tree else 14
            pyg_data = Data(
                x=torch.zeros((1, node_dim), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, edge_dim), dtype=torch.float)
            )
        sdim   = getattr(getattr(self.cfg.model, 'ms_encoder', {}), 'spectrum_dim', 100)
        sorder = getattr(getattr(self.cfg.model, 'ms_encoder', {}), 'spectrum_order', 'intensity_desc')
        return {
            'spec_tensor': preprocess_spectrum(query['ms2_peaks'], dim=sdim, order=sorder),
            'adduct_type_idx': self.adduct_map.get(query['adduct'], self.adduct_map["Unknown"]),
            'precursor_mz': query['precursor_mz'],
            'true_smiles': query['smiles'],
            'adduct': query.get('adduct', None),  # === 10ppm: 加合物字符串
            'pyg_data': pyg_data
        }

def collate_mol(batch):
    batch = list(filter(None, batch))
    if not batch: return {}
    bat = {}
    if 'mol_fps' in batch[0]:
        bat['mol_fps'] = torch.stack([b['mol_fps'] for b in batch])
    if 'V' in batch[0]:
        max_n = max(b['V'].shape[0] for b in batch)
        bat['V'] = torch.stack([pad_V(b['V'], max_n) for b in batch])
        bat['A'] = torch.stack([pad_A(b['A'], max_n) for b in batch])
        bat['mol_size'] = torch.cat([b['mol_size'] for b in batch])
        
    bat['row_idx'] = torch.tensor([b['_row_idx'] for b in batch], dtype=torch.long)
    return bat

def collate_ms(batch):
    batch = list(filter(None, batch))
    if not batch: return {}
    return {
        'spec_tensor': torch.stack([b['spec_tensor'] for b in batch]).unsqueeze(1),
        'adduct_type_idx': torch.tensor([b['adduct_type_idx'] for b in batch]),
        'precursor_mz': torch.tensor([b['precursor_mz'] for b in batch]),
        'true_smiles': [b['true_smiles'] for b in batch],
        'adduct': [b['adduct'] for b in batch],  # === 10ppm: 加合物字符串
        'pyg_data': Batch.from_data_list([b['pyg_data'] for b in batch])
    }


def main(args):
    # 模型加载和数据准备
    print(f"正在从 {args.model_path} 加载checkpoint...")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)

    # 先使用模型内置配置（若存在）
    if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
        from config import ConfigDict
        def _to_cfgdict(d):
            if isinstance(d, dict):
                return ConfigDict({k: _to_cfgdict(v) for k, v in d.items()})
            return d
        CFG.clear()
        CFG.update(_to_cfgdict(checkpoint['config']))
        print("已使用checkpoint内保存的配置作为默认CFG。")
    else:
        print("注意：checkpoint中未找到内置配置。")

    # 若用户显式提供 --cfg，则再覆盖
    if args.cfg is not None:
        CFG.load(args.cfg)
        print(f"已读取并使用用户提供的配置覆盖默认CFG: {args.cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = FragSimiModel(CFG, enable_compile=False).to(device).eval()

    _ = load_checkpoint_with_compat(
        model,
        checkpoint,
        cfg=CFG,
        ema=args.ema,     # "auto" | "on" | "off"
        verbose=True
    )

    with open(args.query_list, 'r') as f:
        query_files = [line.strip() for line in f]
    with open(args.candidate_list, 'r') as f:
        candidate_pool_smiles = {line.strip() for line in f}
    
    ms_dataset = MSDataset(query_files, CFG.data.dataset_path, CFG.data.frag_tree_dir, CFG, missing_tree_policy=args.missing_tree_policy)
    ms_loader = DataLoader(ms_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=collate_ms, pin_memory=True)

    all_true_smiles_list = [q['smiles'] for q in ms_dataset.queries]
    corpus_smiles_set = set(all_true_smiles_list) | candidate_pool_smiles
    corpus_smiles_list = sorted(list(corpus_smiles_set))
    smi_to_corpus_idx = {smi: i for i, smi in enumerate(corpus_smiles_list)}
    print(f"已构建分子语料库，包含 {len(corpus_smiles_list)} 个独立分子。")

    # 预编码部分
    with torch.no_grad():
        print("正在预编码分子语料库...")
        corpus_dataset = MolDataset(corpus_smiles_list, CFG)
        corpus_loader = DataLoader(corpus_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=collate_mol, pin_memory=True)
        
        corpus_embeddings = None
        D = None
        
        # 寻找第一个有效批次来确定嵌入维度
        first_valid_batch_found = False
        for batch in corpus_loader:
            if batch:
                row_indices_first = batch.pop('row_idx').cpu().numpy()
                for k, v in batch.items(): batch[k] = v.to(device)
                first_embeds = model.encode_mol(batch).cpu()
                D = first_embeds.shape[1]
                corpus_embeddings = torch.full((len(corpus_smiles_list), D), float('nan'), dtype=torch.float32, device='cpu')
                corpus_embeddings[row_indices_first] = first_embeds
                first_valid_batch_found = True
                break
        
        if not first_valid_batch_found:
             raise ValueError("语料库编码失败：语料库中未找到任何有效分子。")

        print(f"已预分配语料库嵌入张量，形状为: {corpus_embeddings.shape}")

        print("正在编码并填充语料库...")
        for batch in tqdm(corpus_loader, desc="编码语料库"):
            if not batch: continue
            row_indices = batch.pop('row_idx').cpu().numpy()
            for k, v in batch.items(): batch[k] = v.to(device)
            embeds = model.encode_mol(batch).cpu()
            corpus_embeddings[row_indices] = embeds

        failed_mask = torch.isnan(corpus_embeddings).any(dim=1)
        if failed_mask.any():
            failed_indices = failed_mask.nonzero().squeeze(1).tolist()
            num_failed = len(failed_indices)
            print(f"警告：{len(corpus_smiles_list)} 个语料库分子中有 {num_failed} 个编码失败。")
            print(f"失败分子的索引（前10个）: {failed_indices[:10]}...")
            valid_corpus_mask = ~failed_mask
            corpus_embeddings = corpus_embeddings[valid_corpus_mask]
            original_smiles_list = corpus_smiles_list
            corpus_smiles_list = [smi for i, smi in enumerate(original_smiles_list) if valid_corpus_mask[i]]
            smi_to_corpus_idx = {smi: i for i, smi in enumerate(corpus_smiles_list)}
            print(f"为进行评测，语料库大小已缩减至 {len(corpus_smiles_list)} 个有效分子。")
        
        # === 10ppm: 计算语料库每个 SMILES 的精确质量（一次性，内存中缓存）
        print("正在计算候选分子的精确质量（RDKit 单同位素）...")
        corpus_exact_masses = np.array([smiles_exact_mass(smi) for smi in corpus_smiles_list], dtype=np.float64)

        kept_true_smiles = []
        kept_adducts = []          # === 10ppm
        kept_precursor_mz = []     # === 10ppm
        all_query_embeds = []
        print("正在预编码查询谱图...")
        for batch in tqdm(ms_loader, desc="编码查询"):
            if not batch: continue
            true_smiles_batch = batch.pop('true_smiles')  # 记录被实际编码的查询
            adducts_batch = batch.pop('adduct')           # === 10ppm: 保存字符串列表
            precursor_mz_list = batch['precursor_mz'].tolist()  # === 10ppm: 在 to(device) 前拷贝数值
            for k, v in batch.items():
                if k != 'pyg_data':
                    batch[k] = v.to(device)
            embeds = model.encode_ms(batch).cpu()
            all_query_embeds.append(embeds)
            kept_true_smiles.extend(true_smiles_batch)
            kept_adducts.extend(adducts_batch)            # === 10ppm
            kept_precursor_mz.extend(precursor_mz_list)   # === 10ppm
        
        if not all_query_embeds:
            raise ValueError("没有任何可编码的查询。")
        
        query_embeddings_raw = torch.cat(all_query_embeds)

    corpus_embeddings = F.normalize(corpus_embeddings).contiguous()

    # === EVAL MODE 互斥：full / one-vs-rand / 10ppm
    if args.eval_mode == "full":
        print(f"\n开始评测（策略：全候选分子集；由模型 predict(mode='rank') 计算）...")
        method_str = "Full-corpus ranking"
    elif args.eval_mode == "one-vs-rand":
        print(f"\n开始评测（策略：'1个真值 + 99个随机负例'；由模型 predict(mode='rank') 计算）...")
        method_str = "1 True + 99 Random Negatives"
    else:  # "10ppm"
        print(f"\n开始评测（策略：按查询的 {args.ppm} ppm 窗口筛候选；由模型 predict(mode='rank') 计算）...")
        method_str = f"{args.ppm} ppm window per query"

    eval_start_time = time.time()

    kept_true_smiles_np = np.array(kept_true_smiles, dtype=object)
    valid_mask = np.array([smi in smi_to_corpus_idx for smi in kept_true_smiles_np], dtype=bool)
    true_indices = np.array([smi_to_corpus_idx[smi] for smi in kept_true_smiles_np[valid_mask]], dtype=np.int64)
    query_embeddings = F.normalize(query_embeddings_raw[valid_mask]).contiguous()
    num_evaluable = len(true_indices)
    num_queries = len(kept_true_smiles)  # 以成功编码的查询数量作为总数口径
    num_corpus = len(corpus_smiles_list)
    assert len(query_embeddings) == num_evaluable, "编码后的谱图数量与可评测的谱图数量不匹配！"
    print(f"在 {num_queries} 个成功编码的谱图中，找到 {num_evaluable} 个可评测的。")

    D = corpus_embeddings.shape[1]

    # ================== 三种模式 ==================
    if args.eval_mode == "full":
        ranks_tensor = model.predict(
            query_batch=None,
            preencoded_queries=query_embeddings.to(device) if device.type == "cuda" else query_embeddings,
            candidate_embeddings=corpus_embeddings,  # predict 会自行管理存放位置
            mode="rank",
            labels=torch.from_numpy(true_indices),
            device=device,
            cand_chunk_size=args.eval_chunk_size or 0
        )
        ranks = ranks_tensor.cpu().numpy()

    elif args.eval_mode == "one-vs-rand":
        rng = np.random.default_rng(args.seed)
        num_negatives = 99
        if num_corpus <= num_negatives:
            raise ValueError(f"语料库大小 ({num_corpus}) 不足以采样 {num_negatives} 个独立的负例。")
        negative_pool_size = num_corpus - 1

        neg_indices = np.empty((num_evaluable, num_negatives), dtype=np.int64)
        print("正在为每个query生成负例（此过程很快）...")
        for i, t in enumerate(tqdm(true_indices, desc="采样负例")):
            row_samples = rng.choice(negative_pool_size, size=num_negatives, replace=False)
            row_samples[row_samples >= t] += 1
            neg_indices[i] = row_samples

        per_query_candidates = np.concatenate([true_indices[:, None], neg_indices], axis=1)
        per_query_candidates_t = torch.from_numpy(per_query_candidates)  # [B, 100]

        ranks_tensor = model.predict(
            query_batch=None,
            preencoded_queries=query_embeddings.to(device) if device.type == "cuda" else query_embeddings,
            candidate_embeddings=corpus_embeddings,
            mode="rank",
            per_query_candidates=per_query_candidates_t,
            device=device,
            cand_chunk_size=args.eval_chunk_size or 0
        )
        ranks = ranks_tensor.cpu().numpy()

    else:
        # === 10ppm 模式（加速版）：
        # 1) 预处理：质量数组排序 + 向量化窗口 → O(Q log N + ΣK)
        ppm = float(args.ppm)

        # 序列化必要的 query 信息
        adducts_arr = np.array(kept_adducts, dtype=object)[valid_mask]
        precursor_mz_arr = np.array(kept_precursor_mz, dtype=np.float64)[valid_mask]
        true_masses_arr = corpus_exact_masses[true_indices]

        # 若有 NaN 的真值质量，回退计算一次（极少见）
        nan_tm = ~np.isfinite(true_masses_arr)
        if np.any(nan_tm):
            for i in np.where(nan_tm)[0]:
                true_masses_arr[i] = smiles_exact_mass(corpus_smiles_list[int(true_indices[i])])

        # 计算中性质量 & ppm 窗口（向量化）
        neutral_mass_arr = np.empty_like(precursor_mz_arr, dtype=np.float64)
        for i in range(num_evaluable):
            neutral_mass_arr[i] = mz_to_neutral_mass_pos(
                float(precursor_mz_arr[i]),
                adducts_arr[i],
                float(true_masses_arr[i])
            )
        tol_arr = neutral_mass_arr * (ppm * 1e-6)
        lo_arr = neutral_mass_arr - tol_arr
        hi_arr = neutral_mass_arr + tol_arr

        # 语料质量排序索引（一次性）
        print("预处理：按质量对语料排序以加速窗口检索 ...")
        valid_mass_mask = np.isfinite(corpus_exact_masses)
        # 仅对有效质量排序，避免 NaN 干扰；同时保留映射回全量索引
        valid_indices_all = np.nonzero(valid_mass_mask)[0]
        masses_valid = corpus_exact_masses[valid_mass_mask]
        sort_order = np.argsort(masses_valid, kind="mergesort")
        masses_sorted = masses_valid[sort_order]
        idx_sorted = valid_indices_all[sort_order]

        # 2) 为每个 query 构造“候选索引切片” + 真值是否在窗
        #    同时为 K=0 或真值不在窗者直接给 rank，无需调用 predict
        print("构造每个查询的候选集合（searchsorted） ...")
        ranks = np.empty((num_evaluable,), dtype=np.int64)
        missed_cnt = 0
        missed_ppm_list = []

        # 分桶：K -> [(q_idx, per_query_candidates)]
        buckets = {}  # K : list of (q_idx, np.ndarray[K])
        for i in range(num_evaluable):
            lo = lo_arr[i]; hi = hi_arr[i]
            if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
                # 无效窗口，当作 K=0 处理
                ranks[i] = num_corpus + 1
                missed_cnt += 1
                if np.isfinite(true_masses_arr[i]) and neutral_mass_arr[i] > 0:
                    missed_ppm_list.append(abs(true_masses_arr[i] - neutral_mass_arr[i]) / neutral_mass_arr[i] * 1e6)
                continue

            l = np.searchsorted(masses_sorted, lo, side='left')
            r = np.searchsorted(masses_sorted, hi, side='right')
            if r <= l:
                # K=0
                ranks[i] = num_corpus + 1
                missed_cnt += 1
                if np.isfinite(true_masses_arr[i]) and neutral_mass_arr[i] > 0:
                    missed_ppm_list.append(abs(true_masses_arr[i] - neutral_mass_arr[i]) / neutral_mass_arr[i] * 1e6)
                continue

            cand_idx = idx_sorted[l:r]
            t = int(true_indices[i])

            # 真值是否在窗口内
            if not np.any(cand_idx == t):
                K = cand_idx.size
                ranks[i] = K + 1  # 真值不在窗：视为“排在所有候选之后”
                missed_cnt += 1
                if np.isfinite(true_masses_arr[i]) and neutral_mass_arr[i] > 0:
                    missed_ppm_list.append(abs(true_masses_arr[i] - neutral_mass_arr[i]) / neutral_mass_arr[i] * 1e6)
                continue

            # 真值在窗：构造 per_query_candidates（第0列为真值）
            if cand_idx.size > 1:
                others = cand_idx[cand_idx != t]
                per_query = np.concatenate([[t], others])
            else:
                per_query = np.array([t], dtype=np.int64)

            K = per_query.size
            buckets.setdefault(K, []).append((i, per_query))

        # 3) 按 K 分桶批量调用 predict(mode='rank')，减少高频小调用开销
        if buckets:
            print("按候选数(K)分桶批量评测 ...")
        for K, items in sorted(buckets.items()):
            # 将该桶中的查询再按批次切分，避免一次过大
            # 这里复用 args.batch_size 作为“每个 predict 的 query 数量”
            B = args.batch_size if args.batch_size > 0 else 256
            total = len(items)
            for start in range(0, total, B):
                chunk = items[start:start+B]
                q_indices = [q for (q, _) in chunk]
                per_query_mat = np.stack([pq for (_, pq) in chunk], axis=0)  # [b, K]
                # 收集对应的 query embeddings（并放到 GPU 上以减少搬运）
                q_embeds = query_embeddings[q_indices]
                q_embeds_dev = q_embeds.to(device) if device.type == "cuda" else q_embeds
                per_query_candidates_t = torch.from_numpy(per_query_mat)

                ranks_tensor = model.predict(
                    query_batch=None,
                    preencoded_queries=q_embeds_dev,
                    candidate_embeddings=corpus_embeddings,  # predict 内部以 CPU->GPU 分块方式处理
                    mode="rank",
                    per_query_candidates=per_query_candidates_t,
                    device=device,
                    cand_chunk_size=args.eval_chunk_size or 0
                ).cpu().numpy()

                # 写回该批的结果
                for local_i, q_idx in enumerate(q_indices):
                    ranks[q_idx] = int(ranks_tensor[local_i])

    # 指标计算
    top_k_hits = {k: np.sum(ranks <= k) for k in [1, 3, 5, 10, 20, 50, 100]}
    mrr_scores = 1.0 / ranks

    eval_end_time = time.time()
    print(f"评测完成，耗时 {eval_end_time - eval_start_time:.2f} 秒。")

    # 报告和保存结果
    summary_data = {
        "run_info": {
            "model_path": os.path.basename(args.model_path),
            "query_list": args.query_list,
            "candidate_list": args.candidate_list,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries": num_queries,
            "evaluable_queries": int(num_evaluable),
            "evaluation_method": method_str,
            "random_seed": args.seed,
            "corpus_size_initial": len(set([q['smiles'] for q in ms_dataset.queries]) | candidate_pool_smiles),
            "corpus_size_final": num_corpus,
        },
        "metrics": {
            "top_k_accuracy": {k: f"{(v/num_evaluable):.4f}" if num_evaluable > 0 else "0.0" for k, v in top_k_hits.items()},
            "mrr": f"{np.mean(mrr_scores):.4f}" if num_evaluable > 0 else "0.0"
        }
    }

    print("\n--- 评测结果 ---")
    if num_evaluable > 0:
        for k, hits in top_k_hits.items():
            accuracy = (hits / num_evaluable) * 100
            print(f"Top-{k} 准确率: {int(hits)}/{num_evaluable} ({accuracy:.2f}%)")
        mean_mrr = np.mean(mrr_scores)
        print(f"平均倒数排名 (MRR): {mean_mrr:.4f}")
    else:
        print("没有可用于评测的查询。")
    print("------------------------\n")

    # === 10ppm: 补充统计（真值不在窗口内的占比与偏差）
    if args.eval_mode == "10ppm":
        try:
            miss_n = int(missed_cnt)
            miss_pct = (miss_n / num_evaluable * 100.0) if num_evaluable > 0 else 0.0
            if len(missed_ppm_list) > 0:
                avg_ppm = float(np.mean(missed_ppm_list))
                med_ppm = float(np.median(missed_ppm_list))
            else:
                avg_ppm = float('nan')
                med_ppm = float('nan')

            print(f"[10ppm统计] 真值不在 {args.ppm} ppm 窗口内：{miss_n}/{num_evaluable} ({miss_pct:.2f}%)")
            print(f"[10ppm统计] 这些样本的 ppm 偏差：平均 {avg_ppm:.2f} ppm，中位数 {med_ppm:.2f} ppm")

            summary_data["10ppm_stats"] = {
                "ppm": float(args.ppm),
                "missed_count": miss_n,
                "missed_ratio": f"{(miss_n/num_evaluable):.4f}" if num_evaluable > 0 else "0.0",
                "missed_ppm_mean": None if len(missed_ppm_list)==0 else round(avg_ppm, 6),
                "missed_ppm_median": None if len(missed_ppm_list)==0 else round(med_ppm, 6),
            }
        except Exception:
            pass

    # 保存排名到文件（如果指定了路径）
    if args.save_ranks_to:
        try:
            output_dir = os.path.dirname(args.save_ranks_to)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"已自动创建目录: {output_dir}")
            np.savetxt(args.save_ranks_to, ranks, fmt='%d')
            print(f"所有查询的排名已成功保存至: {args.save_ranks_to}")
        except Exception as e:
            print(f"错误：无法保存排名文件: {e}")

    model_name_tag = os.path.basename(args.model_path).replace('.pth', '')
    output_basename = f"results_{model_name_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    summary_filepath = os.path.join(args.output_dir, f"{output_basename}_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    print(f"评测摘要已保存至: {summary_filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run model prediction with strict, vectorized evaluation.")
    parser.add_argument("model_path", type=str, help="Path to the trained model .pth file.")
    parser.add_argument("query_list", type=str, help="Path to the .txt file listing query MGF files.")
    parser.add_argument("candidate_list", type=str, help="Path to the .txt file of all possible negatives.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for negative sampling to ensure reproducibility.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for encoding and per-bucket predict.")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save prediction results.")
    parser.add_argument("--cfg", type=str, default=None, help="可选：配置文件路径。若不提供，则默认使用模型checkpoint内置配置。")
    parser.add_argument("--eval_chunk_size", type=int, default=0,
                        help="评测时的分块大小。0 表示自动根据可用显存估算。")
    # === EVAL MODE（互斥）
    parser.add_argument("--eval_mode", choices=["full", "one-vs-rand", "10ppm"], default="full",
                        help="评测模式：full=全候选分子集；one-vs-rand=1真+99负；10ppm=每query在其10ppm窗口内评测（仅正离子）。")
    parser.add_argument("--ppm", type=float, default=10.0,
                        help="10ppm模式：质量偏差ppm（默认 10.0）。")
    parser.add_argument("--save_ranks_to", type=str, default=None,
                        help="如果指定，则将所有查询的原始排名保存到此文件（每行一个排名）。")
    parser.add_argument("--ema", choices=["auto", "on", "off"], default="auto",
                        help="评测时是否使用 EMA 权重：auto=若配置或ckpt提供则用(默认)，on=强制使用，off=不使用")

    parser.add_argument("--missing-tree-policy", choices=["discard", "placeholder"], default="discard",
                    help="处理无碎裂树样本的策略: 'discard' (默认) 则丢弃样本, 'placeholder' 则使用空图占位符。")


    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
