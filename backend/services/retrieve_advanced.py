import json
import os
import sys
from threading import Lock

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import core.config as app_cfg
from services.retrieve import MSDataset, collate_ms

PROJ_ROOT = "/root/web/backend/services/model"
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from config import CFG, ConfigDict
from modules_advanced import FragSimiModel
from predict import MolDataset, collate_mol

DEFAULT_ION_MODE = "pos"
VALID_ION_MODES = ("pos", "neg")

ADV_MODEL_WEIGHT_PATHS = {
    "pos": str(
        getattr(
            app_cfg,
            "retrieve_advanced_model_weight_path_pos",
            "/home/nfs06/wuzt/wzt/outputs_finetune/512_triple_finetune_mist_xatten_morgan3_torsion_true/ft_e8_loss1.0037.pth",
        )
    ),
    "neg": str(
        getattr(
            app_cfg,
            "retrieve_advanced_model_weight_path_neg",
            "/home/nfs06/wuzt/wzt/outputs_neg_finetune/neg_mist_finetune/ft_e5_loss1.0547.pth",
        )
    ),
}

BATCH_SIZE = int(getattr(app_cfg, "retrieve_batch_size", 256))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MISSING_TREE_POLICY = str(getattr(app_cfg, "retrieve_missing_tree_policy", "discard"))


class ContrastiveFineTuneModel(nn.Module):
    def __init__(self, base_model, projection_dim=512, dropout=0.1):
        super().__init__()
        self.main_model = base_model

        if hasattr(self.main_model, "fp_prediction_head"):
            self.ms_dim = self.main_model.fp_prediction_head.in_features
        else:
            self.ms_dim = 1024

        if hasattr(self.main_model, "mol_gnn_encoder"):
            self.mol_dim = int(self.main_model.cfg.model.mol_encoder.embedding_dim) * 2
        else:
            self.mol_dim = 512

        self.spec_projector = nn.Sequential(
            nn.Linear(self.ms_dim, self.ms_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ms_dim, projection_dim),
        )
        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_dim, self.mol_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.mol_dim, projection_dim),
        )

    def encode_query(self, batch):
        ms_feats = self.main_model.encode_ms(batch)
        proj = self.spec_projector(ms_feats)
        return F.normalize(proj, dim=1)

    def encode_candidate(self, batch):
        mol_feats = self.main_model.encode_mol(batch)
        proj = self.mol_projector(mol_feats)
        return F.normalize(proj, dim=1)


_RUNTIME_MODELS: dict[str, torch.nn.Module] = {}
_RUNTIME_INIT_LOCK = Lock()


def _normalize_ion_mode(ion_mode: str | None) -> str:
    mode = (ion_mode or DEFAULT_ION_MODE).strip().lower()
    if mode not in VALID_ION_MODES:
        raise RuntimeError(f"不支持的离子模式: {ion_mode}")
    return mode


def _to_cfgdict(d):
    if isinstance(d, dict):
        return ConfigDict({k: _to_cfgdict(v) for k, v in d.items()})
    return d


def init_retrieve_advanced_process(ion_mode: str = DEFAULT_ION_MODE) -> None:
    mode = _normalize_ion_mode(ion_mode)
    if mode in _RUNTIME_MODELS:
        return

    with _RUNTIME_INIT_LOCK:
        if mode in _RUNTIME_MODELS:
            return

        model_weight_path = ADV_MODEL_WEIGHT_PATHS.get(mode)
        if not model_weight_path:
            raise RuntimeError(f"离子模式 {mode} 未配置高级检索模型权重路径")

        checkpoint = torch.load(model_weight_path, map_location="cpu", weights_only=False)
        if "config" not in checkpoint or not isinstance(checkpoint["config"], dict):
            raise RuntimeError(f"高级检索模型文件中未找到内置配置({mode})")

        CFG.clear()
        CFG.update(_to_cfgdict(checkpoint["config"]))

        base_model = FragSimiModel(CFG, enable_compile=False)
        projection_dim = int(getattr(CFG.finetune, "projection_dim", 512))
        model = ContrastiveFineTuneModel(base_model, projection_dim=projection_dim)

        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        _RUNTIME_MODELS[mode] = model.to(DEVICE).eval()


def get_retrieve_advanced_runtime(ion_mode: str = DEFAULT_ION_MODE):
    mode = _normalize_ion_mode(ion_mode)
    if mode not in _RUNTIME_MODELS:
        init_retrieve_advanced_process(ion_mode=mode)
    return _RUNTIME_MODELS[mode]


def encode_unique_smiles_projected(smiles_list: list[str], model: ContrastiveFineTuneModel, device: str):
    if not smiles_list:
        projection_dim = int(getattr(CFG.finetune, "projection_dim", 512))
        return torch.empty((0, projection_dim), dtype=torch.float32)

    ds_mol = MolDataset(smiles_list, CFG)
    loader = DataLoader(ds_mol, batch_size=BATCH_SIZE, collate_fn=collate_mol, pin_memory=True)

    projection_dim = int(getattr(CFG.finetune, "projection_dim", 512))
    embeds = torch.full((len(smiles_list), projection_dim), float("nan"), dtype=torch.float32, device="cpu")

    with torch.no_grad():
        for batch in tqdm(loader, desc="编码候选分子", leave=False):
            if not batch:
                continue
            row_idx = batch.pop("row_idx").cpu().numpy()
            for k, v in batch.items():
                if hasattr(v, "to"):
                    batch[k] = v.to(device)
            emb = model.encode_candidate(batch).detach().cpu()
            embeds[row_idx] = emb

    failed_mask = torch.isnan(embeds).any(dim=1)
    embeds[failed_mask] = 0.0
    return embeds.contiguous()


def encode_spectra_projected(ms_loader, model: ContrastiveFineTuneModel, device: str):
    spec_embeddings = []
    spec_titles = []
    spec_true_smiles = []

    with torch.no_grad():
        for batch in tqdm(ms_loader, desc="编码谱图", leave=False):
            if not batch:
                continue
            spec_titles.extend(batch["title"])
            spec_true_smiles.extend(batch["true_smiles"])

            batch_input = {}
            for k, v in batch.items():
                if k in {"title", "true_smiles", "adduct"}:
                    continue
                if k != "pyg_data":
                    batch_input[k] = v.to(device)
                else:
                    batch_input[k] = v

            spec_emb = model.encode_query(batch_input).cpu()
            spec_embeddings.append(spec_emb)

    if not spec_embeddings:
        return None, None, None
    return torch.cat(spec_embeddings), spec_titles, spec_true_smiles


def _default_result_meta() -> dict:
    return {
        "formula": "",
        "exact_mass": None,
        "inchikey": "",
        "adduct": "",
        "source": "",
    }


def _build_result_row(rank: int, score: float, smiles: str, meta: dict | None = None) -> dict:
    safe_meta = _default_result_meta()
    if isinstance(meta, dict):
        for key, value in meta.items():
            if key in {"rank", "smiles", "score"}:
                continue
            safe_meta[key] = value

    return {
        "rank": rank,
        "smiles": smiles,
        "score": float(score),
        **safe_meta,
    }


def _load_candidate_map(statas_data: dict) -> tuple[dict[str, list[str]], dict[str, dict[str, dict]]]:
    try:
        entries = statas_data["碎裂树文件统计"]["有效碎裂树根节点信息"]
    except Exception as exc:
        raise RuntimeError("statas.json结构不合法，缺少碎裂树文件统计/有效碎裂树根节点信息") from exc

    candidate_map: dict[str, list[str]] = {}
    metadata_map: dict[str, dict[str, dict]] = {}

    for item in entries:
        title = item.get("title")
        result = item.get("检索结果", {})
        if not title:
            continue

        pool: list[str] = []
        title_meta: dict[str, dict] = {}

        result_top100 = result.get("result_top100", [])
        if isinstance(result_top100, list) and result_top100:
            for row in result_top100:
                if not isinstance(row, dict):
                    continue
                smi = row.get("smiles")
                if isinstance(smi, str) and smi.strip():
                    safe_smi = smi.strip()
                    pool.append(safe_smi)
                    if safe_smi not in title_meta:
                        title_meta[safe_smi] = {
                            k: v for k, v in row.items() if k not in {"rank", "smiles", "score"}
                        }

        if not pool:
            legacy_pool = result.get("top100", [])
            if isinstance(legacy_pool, list):
                pool = [s.strip() for s in legacy_pool if isinstance(s, str) and s.strip()]

        candidate_map[title] = list(dict.fromkeys(pool))
        metadata_map[title] = title_meta

    return candidate_map, metadata_map


def main(
    ion_mode: str = DEFAULT_ION_MODE,
    statas_json_path: str = "/home/nfs05/wuzt/AI+/ZZZ_grn/web/backend/temp/task_xxx/statas_advanced.json",
    fragtrees_json_path: str = "/home/nfs05/wuzt/AI+/ZZZ_grn/web/backend/temp/task_xxx/valid_pairs_fragtrees.json",
    spectra_mgf_path: str = "/home/nfs05/wuzt/AI+/ZZZ_grn/web/backend/temp/task_xxx/valid_pairs_spectra.mgf",
):
    if not os.path.exists(statas_json_path):
        raise RuntimeError("statas.json不存在")

    with open(statas_json_path, "r", encoding="utf-8") as f:
        stats_data = json.load(f)

    model = get_retrieve_advanced_runtime(ion_mode=ion_mode)

    ms_dataset = MSDataset(spectra_mgf_path, fragtrees_json_path, CFG, missing_tree_policy=MISSING_TREE_POLICY)
    ms_loader = DataLoader(ms_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=collate_ms, pin_memory=True)

    spec_embs, titles, true_smiles = encode_spectra_projected(ms_loader, model, DEVICE)
    if spec_embs is None:
        raise RuntimeError("高级检索谱图编码失败")

    candidate_map, metadata_map = _load_candidate_map(stats_data)

    final = {}
    for idx, title in enumerate(tqdm(titles, desc="高级检索")):
        smiles = candidate_map.get(title, [])
        title_meta = metadata_map.get(title, {})
        if not smiles:
            final[title] = {
                "status": "failed",
                "reason": "候选池为空",
                "top1": "",
                "top10": [],
                "top100": [],
                "top100_score": [],
                "result_top100": [],
            }
            continue

        emb = encode_unique_smiles_projected(smiles, model, DEVICE).to(DEVICE)
        q_emb = spec_embs[idx : idx + 1].to(DEVICE)
        scores = torch.mm(q_emb, emb.T).squeeze(0)
        ranks = torch.argsort(scores, descending=True)
        ordered = [smiles[i] for i in ranks]
        scores_np = scores[ranks].cpu().numpy().tolist()

        result_top100 = [
            _build_result_row(
                rank=rank,
                score=score,
                smiles=smi,
                meta=title_meta.get(smi),
            )
            for rank, (smi, score) in enumerate(zip(ordered, scores_np), start=1)
        ]

        final[title] = {
            "true_smiles": true_smiles[idx],
            "top1": ordered[0] if ordered else "",
            "top10": ordered[:10],
            "top100": ordered,
            "top100_score": scores_np,
            "result_top100": result_top100,
            "status": "success",
            "candidate_count": len(smiles),
            "ion_mode": _normalize_ion_mode(ion_mode),
            "stage": "advanced",
        }

    stats_root = stats_data.get("碎裂树文件统计", {}).get("有效碎裂树根节点信息", [])
    for i, item in enumerate(stats_root):
        title = item.get("title")
        if not title:
            continue
        stats_root[i]["检索结果"] = final.get(
            title,
            {
                "status": "not_processed",
                "reason": "谱图未参与高级检索",
            },
        )

    write_path = os.path.abspath(statas_json_path)
    tmp_path = write_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, write_path)

    success_count = sum(1 for v in final.values() if v.get("status") == "success")
    total_count = len(final)
    return {
        "statas_path": write_path,
        "total_count": total_count,
        "success_count": success_count,
        "failed_count": total_count - success_count,
    }
