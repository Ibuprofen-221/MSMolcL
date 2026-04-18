import os
import json
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
import sys
# 第三方库
try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ========= 项目路径 =========
PROJ_ROOT = "/root/web/backend/services/model"
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# ========= 你的模型配置 =========
from config import CFG, ConfigDict
from modules_advanced import FragSimiModel
from predict import MolDataset, collate_mol
from compatibility import load_checkpoint_with_compat

# ======================== 全局显式参数配置（直接修改此处！）========================
MGF_FILE_PATH = "/home/nfs05/wuzt/AI+/ZZZ_grn/web/backend/temp/task_8b723751f9df4c62b8dd789d357011d8/valid_pairs_spectra.mgf"
FRAG_TREE_JSON_PATH = "/home/nfs05/wuzt/AI+/ZZZ_grn/web/backend/temp/task_8b723751f9df4c62b8dd789d357011d8/valid_pairs_fragtrees.json"
STATS_JSON_PATH = "/home/nfs05/wuzt/AI+/ZZZ_grn/web/backend/temp/task_8b723751f9df4c62b8dd789d357011d8/statas.json"
CANDIDATE_JSON_PATH = "/home/nfs05/wuzt/AI+/ZZZ_grn/web/backend/temp/task_8b723751f9df4c62b8dd789d357011d8/statas.json"
MODEL_WEIGHT_PATH = "/home/nfs06/wuzt/wzt/outputs_finetune/512_triple_finetune_mist_xatten_morgan3_torsion_true/ft_e8_loss1.0037.pth"
OUTPUT_JSON_PATH = "/home/nfs05/wuzt/AI+/ZZZ_grn/web/backend/temp/task_8b723751f9df4c62b8dd789d357011d8/final_statas.json"

BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MISSING_TREE_POLICY = "discard"

# ======================== 二阶段投影模型（完全使用你提供的版本）========================
class ContrastiveFineTuneModel(nn.Module):
    def __init__(self, base_model, projection_dim=512, dropout=0.1):
        super().__init__()
        self.main_model = base_model
        
        if hasattr(self.main_model, 'fp_prediction_head'):
            self.ms_dim = self.main_model.fp_prediction_head.in_features
        else:
            self.ms_dim = 1024 
            
        if hasattr(self.main_model, 'mol_gnn_encoder'):
             self.mol_dim = int(self.main_model.cfg.model.mol_encoder.embedding_dim)*2
        else:
             self.mol_dim = 512

        self.spec_projector = nn.Sequential(
            nn.Linear(self.ms_dim, self.ms_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ms_dim, projection_dim)
        )

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_dim, self.mol_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.mol_dim, projection_dim)
        )

    def encode_query(self, batch):
        ms_feats = self.main_model.encode_ms(batch)
        proj = self.spec_projector(ms_feats)
        return F.normalize(proj, dim=1)

    def encode_candidate(self, batch):
        mol_feats = self.main_model.encode_mol(batch)
        proj = self.mol_projector(mol_feats)
        return F.normalize(proj, dim=1)

# ======================== SMILES 规范化 ========================
def canonicalize_smiles(smi: str) -> str:
    if not HAS_RDKIT:
        return smi
    if not smi or smi == "None":
        return ""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        return smi

# ======================== 模型加载（你提供的原版）========================
def load_full_finetuned_model(model_path: str, device):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    
    if 'config' in ckpt and isinstance(ckpt['config'], dict):
        def _to_cfgdict(d):
            if isinstance(d, dict):
                return ConfigDict({k: _to_cfgdict(v) for k, v in d.items()})
            return d
        CFG.clear()
        CFG.update(_to_cfgdict(ckpt['config']))

    base_model = FragSimiModel(CFG, enable_compile=False)
    proj_dim = int(getattr(CFG.finetune, "projection_dim", 512))
    model = ContrastiveFineTuneModel(base_model, projection_dim=proj_dim)
    
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()

# ======================== 分子编码（投影版）========================
def encode_unique_smiles_projected(smiles_list, model, device):
    if not smiles_list:
        return torch.empty((0, CFG.finetune.projection_dim), dtype=torch.float32)

    ds_mol = MolDataset(smiles_list, CFG)
    loader = DataLoader(ds_mol, batch_size=BATCH_SIZE, collate_fn=collate_mol, pin_memory=True)

    proj_dim = CFG.finetune.projection_dim
    embeds = torch.full((len(smiles_list), proj_dim), float('nan'), dtype=torch.float32, device='cpu')

    for batch in tqdm(loader, desc="编码候选分子", leave=False):
        if not batch: continue
        row_idx = batch.pop('row_idx').cpu().numpy()
        for k, v in batch.items():
            if hasattr(v, "to"):
                batch[k] = v.to(device)
        with torch.no_grad():
            emb = model.encode_candidate(batch).detach().cpu()
        embeds[row_idx] = emb

    failed_mask = torch.isnan(embeds).any(dim=1)
    embeds[failed_mask] = 0.0
    return embeds.contiguous()

# ======================== 谱图编码（投影版）========================
def encode_spectra(ms_loader, model, device):
    spec_embeddings = []
    spec_titles = []
    spec_true_smiles = []

    with torch.no_grad():
        for batch in tqdm(ms_loader, desc="编码谱图", leave=False):
            if not batch: continue
            spec_titles.extend(batch['title'])
            spec_true_smiles.extend(batch['true_smiles'])

            batch_input = {}
            for k, v in batch.items():
                if k not in ['title', 'true_smiles', 'adduct']:
                    if k != 'pyg_data':
                        batch_input[k] = v.to(device)
                    else:
                        batch_input[k] = v

            spec_emb = model.encode_query(batch_input).cpu()
            spec_embeddings.append(spec_emb)

    if not spec_embeddings:
        return None, None, None
    return torch.cat(spec_embeddings), spec_titles, spec_true_smiles

# ======================== 数据集 & 预处理（保持不变）========================
from dataset import (
    parse_mgf_file, parse_ms2_from_mgf, preprocess_spectrum, get_adduct_map
)
from torch_geometric.data import Data

class MSDataset(Dataset):
    def __init__(self, mgf_file, frag_tree_json, cfg, missing_tree_policy='discard'):
        self.mgf_file = mgf_file
        self.frag_tree_json = frag_tree_json
        self.cfg = cfg
        self.missing_tree_policy = missing_tree_policy
        
        try:
            from FragmentationTreeEncoder import FragmentTreeProcessor
            self.processor = FragmentTreeProcessor()
        except ImportError:
            raise ImportError("Missing FragmentTreeEncoder")
        
        self.queries = self._load_single_query()
        fusion = getattr(getattr(cfg.model, 'ms_encoder', {}), 'fusion', None)
        if fusion is None:
            fusion = 'concat'
        self.need_tree = fusion in ('xattn', 'concat', 'tree-only')
        self.enhanced_tree = bool(getattr(getattr(cfg.model, 'ms_encoder', {}), 'tree_encoder', {}).get('enhanced_features', False))
        self.adduct_map = get_adduct_map(cfg)

    def _load_single_query(self):
        all_queries = []
        try:
            with open(self.frag_tree_json, 'r') as f:
                frag_tree_db = json.load(f)
        except:
            frag_tree_db = {}

        ms1_data = parse_mgf_file(self.mgf_file, self.cfg)
        ms2_data = parse_ms2_from_mgf(self.mgf_file)
        cleaned_ms2 = []
        for e in ms2_data:
            if e.get('title') and e.get('ms2_peaks'):
                cleaned_ms2.append(e)
        ms2_dict = {x['title']: x for x in cleaned_ms2}

        for entry in ms1_data:
            title = entry.get('title')
            if not title or title not in ms2_dict:
                continue
            ms2_entry = ms2_dict[title]
            true_smiles = entry.get('smiles', 'unknown')
            ft_entry = frag_tree_db.get(title, {})
            all_queries.append({**entry, **ms2_entry, 'frag_tree_entry': ft_entry, 'true_smiles': true_smiles})
        return all_queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q = self.queries[idx]
        title = q['title']
        ft = q.get('frag_tree_entry', {})
        pyg_data = None
        if ft and 'frag_tree' in ft:
            try:
                pyg_data = self.processor.json_to_pyg({'frag_tree': ft['frag_tree']}, edge_features=True, enhanced=self.enhanced_tree)
            except:
                pass
        if pyg_data is None and self.need_tree:
            node_dim = 18 if self.enhanced_tree else 15
            edge_dim = 14 if self.enhanced_tree else 13
            pyg_data = Data(x=torch.zeros(1, node_dim), edge_index=torch.zeros(2,0).long(), edge_attr=torch.zeros(0, edge_dim))
        spec = preprocess_spectrum(q['ms2_peaks'])
        return {
            'title': title,
            'spec_tensor': spec,
            'adduct_type_idx': self.adduct_map.get(q.get('adduct'), self.adduct_map["Unknown"]),
            'precursor_mz': q['precursor_mz'],
            'true_smiles': q['true_smiles'],
            'adduct': q.get('adduct'),
            'pyg_data': pyg_data
        }

def collate_ms(batch):
    batch = [b for b in batch if b]
    from torch_geometric.data import Batch
    return {
        'title': [b['title'] for b in batch],
        'spec_tensor': torch.stack([b['spec_tensor'] for b in batch]).unsqueeze(1),
        'adduct_type_idx': torch.tensor([b['adduct_type_idx'] for b in batch]),
        'precursor_mz': torch.tensor([b['precursor_mz'] for b in batch]),
        'true_smiles': [b['true_smiles'] for b in batch],
        'adduct': [b['adduct'] for b in batch],
        'pyg_data': Batch.from_data_list([b['pyg_data'] for b in batch])
    }

# ======================== 主函数（候选池来自JSON）========================
def main():
    print("=== 加载统计JSON ===")
    with open(STATS_JSON_PATH, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    spec_list = stats["碎裂树文件统计"]["有效碎裂树根节点信息"]
    title2info = {x['title']: x for x in spec_list}

    print("=== 加载模型 ===")
    model = load_full_finetuned_model(MODEL_WEIGHT_PATH, DEVICE)

    print("=== 加载谱图 ===")
    ms_ds = MSDataset(MGF_FILE_PATH, FRAG_TREE_JSON_PATH, CFG)
    ms_loader = DataLoader(ms_ds, batch_size=BATCH_SIZE, collate_fn=collate_ms, pin_memory=True)
    spec_embs, titles, true_smiles = encode_spectra(ms_loader, model, DEVICE)

    print("=== 加载候选池JSON ===")
    with open(CANDIDATE_JSON_PATH, 'r', encoding='utf-8') as f:
        cand_data = json.load(f)
    cand_map = {}
    for item in cand_data["碎裂树文件统计"]["有效碎裂树根节点信息"]:
        t = item.get("title")
        pool = item.get("检索结果", {}).get("top100", [])
        if t and pool:
            cand_map[t] = pool

    final = {}
    for idx, title in enumerate(tqdm(titles, desc="检索")):
        if title not in cand_map:
            final[title] = {"status": "failed", "reason": "无候选池"}
            continue
        smiles = cand_map[title]
        if not smiles:
            final[title] = {"status": "failed", "reason": "候选池空"}
            continue

        emb = encode_unique_smiles_projected(smiles, model, DEVICE).to(DEVICE)
        q_emb = spec_embs[idx:idx+1].to(DEVICE)
        scores = torch.mm(q_emb, emb.T).squeeze(0)
        ranks = torch.argsort(scores, descending=True)
        ordered = [smiles[i] for i in ranks]
        scores_np = scores[ranks].cpu().numpy().tolist()

        final[title] = {
            "true_smiles": true_smiles[idx],
            "top1": ordered[0] if ordered else "",
            "top10": ordered[:10],
            "top100": ordered,
            "top100_score": scores_np,
            "status": "success",
            "candidate_count": len(smiles)
        }

    for item in spec_list:
        t = item["title"]
        item["检索结果"] = final.get(t, {"status": "not_processed"})

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    ok = sum(1 for v in final.values() if v["status"] == "success")
    print(f"完成！成功 {ok}/{len(final)} → {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()