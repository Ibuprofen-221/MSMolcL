# compatibility.py
import torch
from typing import Dict, Tuple, List

def _unwrap_orig_mod_keys(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], bool]:
    """保留：把 came-from-compile 的 '._orig_mod.' / '_orig_mod.' 映射回普通名字（主要用于旧 ckpt）。"""
    need_map = any(("._orig_mod." in k) or k.startswith("_orig_mod.") for k in sd.keys())
    if not need_map:
        return sd, False
    mapped = {}
    for k, v in sd.items():
        nk = k.replace("._orig_mod.", ".")
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        mapped[nk] = v
    return mapped, True

# === 新增：规范化与映射工具 ===
def _normalize_key(k: str) -> str:
    """统一键名空间：去掉 DataParallel 的 'module.'，去掉 compile 包装的 '._orig_mod.' / 前缀 '_orig_mod.'。"""
    if k.startswith("module."):
        k = k[len("module."):]
    k = k.replace("._orig_mod.", ".")
    if k.startswith("_orig_mod."):
        k = k[len("_orig_mod."):]
    return k

def _strip_dataparallel_prefix(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], bool]:
    if not any(k.startswith("module.") for k in sd):
        return sd, False
    out = {}
    for k, v in sd.items():
        out[k[len("module."):] if k.startswith("module.") else k] = v
    return out, True

def _build_model_keymap(model: torch.nn.Module) -> Dict[str, str]:
    """
    返回 {规范名 -> 模型真实键名}。模型内部即便有被 compile 的子模块，
    规范名也会把 '._orig_mod.' 等去掉，便于和 ckpt 对齐。
    """
    cur = model.state_dict()
    keymap = {}
    for real_k in cur.keys():
        norm_k = _normalize_key(real_k)
        keymap[norm_k] = real_k
    return keymap

def load_checkpoint_with_compat(
    model: torch.nn.Module,
    checkpoint: Dict,
    *,
    cfg=None,
    ema: str = "auto",          # "auto" | "on" | "off"
    verbose: bool = True
) -> Dict:
    report = {
        "removed_ms_projection_clerms": 0,
        "did_map_orig_mod_ckpt": False,   # 仅 ckpt 侧的 unwrap
        "stripped_module_prefix": False,
        "mapped_fn3_to_fn2": False,
        "mapped_fn2_to_fn3": False,
        "missing": [],
        "unexpected": [],
        "shape_mismatch": [],
        "used_ema": False,
    }

    # ------- 取出并做早期清洗 -------
    sd = dict(checkpoint["state_dict"])  # shallow copy

    # (0) 去 'module.'（DDP/DP 保存）
    sd, stripped = _strip_dataparallel_prefix(sd)
    report["stripped_module_prefix"] = stripped
    if verbose and stripped:
        print("提示：已去除 'module.' 前缀（可能来自多卡保存的权重）。")

    # (1) 移除旧垃圾键
    keys_to_remove = [k for k in sd if k.startswith("ms_projection_clerms.")]
    for k in keys_to_remove:
        del sd[k]
    report["removed_ms_projection_clerms"] = len(keys_to_remove)
    if verbose and keys_to_remove:
        print(f"提示：移除了 {len(keys_to_remove)} 个 'ms_projection_clerms.*' 旧权重。")

    # (2) 若 ckpt 自身带有 compile 包装，先解掉（老 ckpt 场景）
    sd, did_map_ckpt = _unwrap_orig_mod_keys(sd)
    report["did_map_orig_mod_ckpt"] = did_map_ckpt
    if verbose and did_map_ckpt:
        print("[提示] 检测到 ckpt 带 `_orig_mod` 包装，已对 ckpt 键名做解包。")

    # ------- 规范名视角的 fn.2 <-> fn.3 判断/映射 -------
    # 先构造模型侧与 ckpt 侧的“规范名集合”
    model_keys_norm = set(_normalize_key(k) for k in model.state_dict().keys())
    ckpt_keys_norm  = set(_normalize_key(k) for k in sd.keys())

    def has_pattern_norm(keyset, pattern):
        return any(("trees_encoder." in k) and (pattern in k) for k in keyset)

    model_has_fn2 = has_pattern_norm(model_keys_norm, ".fn.2.")
    model_has_fn3 = has_pattern_norm(model_keys_norm, ".fn.3.")
    ckpt_has_fn2  = has_pattern_norm(ckpt_keys_norm,  ".fn.2.")
    ckpt_has_fn3  = has_pattern_norm(ckpt_keys_norm,  ".fn.3.")

    # 把 ckpt 映射到“规范名字典”，便于改名
    sd_norm = {}
    for k, v in sd.items():
        sd_norm[_normalize_key(k)] = v

    # 规则与原逻辑一致，但在“规范名”上执行
    if model_has_fn2 and not model_has_fn3 and ckpt_has_fn3:
        new_sd_norm = {}
        for nk, v in sd_norm.items():
            if "trees_encoder." in nk and ".fn.3." in nk:
                new_sd_norm[nk.replace(".fn.3.", ".fn.2.", 1)] = v
            else:
                new_sd_norm[nk] = v
        sd_norm = new_sd_norm
        report["mapped_fn3_to_fn2"] = True
        if verbose:
            print("提示：模型需要 'fn.2' 而 ckpt 含有 'fn.3'，已在规范名空间做 fn.3 -> fn.2。")
    elif model_has_fn3 and not model_has_fn2 and ckpt_has_fn2 and not ckpt_has_fn3:
        new_sd_norm = {}
        for nk, v in sd_norm.items():
            if "trees_encoder." in nk and ".fn.2." in nk:
                new_sd_norm[nk.replace(".fn.2.", ".fn.3.", 1)] = v
            else:
                new_sd_norm[nk] = v
        sd_norm = new_sd_norm
        report["mapped_fn2_to_fn3"] = True
        if verbose:
            print("提示：模型需要 'fn.3' 而 ckpt 含有 'fn.2'，已在规范名空间做 fn.2 -> fn.3。")
    else:
        if verbose:
            print("提示：'fn.2'/'fn.3' 键名兼容性检查完成，无需映射。")

    # ------- 把“规范名 ckpt”映射到“模型真实键名” -------
    keymap = _build_model_keymap(model)          # {规范名 -> 真实键名}
    remapped = {}
    for nk, v in sd_norm.items():
        real = keymap.get(nk, None)
        if real is not None:
            remapped[real] = v
    # 之后都用 remapped 来加载

    # ------- 预报告（在规范名维度上更有意义） -------
    cur = model.state_dict()
    model_norm_set = set(_normalize_key(k) for k in cur.keys())
    ckpt_norm_set  = set(sd_norm.keys())
    report["missing"] = sorted(list(model_norm_set - ckpt_norm_set))   # 模型需要但 ckpt 没有（规范名）
    report["unexpected"] = sorted(list(ckpt_norm_set - model_norm_set))# ckpt 有但模型没有（规范名）

    shape_diff: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    for nk in (model_norm_set & ckpt_norm_set):
        rk = keymap[nk]                  # 模型真实键
        if cur[rk].shape != sd_norm[nk].shape:
            shape_diff.append((nk, tuple(sd_norm[nk].shape), tuple(cur[rk].shape)))
    report["shape_mismatch"] = shape_diff

    if verbose:
        print(f"\n[Missing keys] {len(report['missing'])}")
        for k in report["missing"][:50]:
            print("  ", k)
        print(f"\n[Unexpected keys] {len(report['unexpected'])}")
        for k in report["unexpected"][:50]:
            print("  ", k)
        print(f"\n[Shape mismatch] {len(shape_diff)}")
        for t in shape_diff[:50]:
            print("  ", t)

    # ------- 真正加载（strict=False） -------
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if verbose:
        print("Model state_dict loaded (strict=False).")

    # ------- EMA 逻辑（不做 fn.2/3 互换；但做规范名→真实名映射） -------
    def _cfg_ema_enabled(_cfg) -> bool:
        try:
            ema_cfg = getattr(_cfg.training, "ema", {})
            return bool(getattr(ema_cfg, "enabled", False))
        except Exception:
            return False

    use_ema = (ema == "on") or (
        ema == "auto" and (
            (_cfg_ema_enabled(cfg) if cfg is not None else False) or ("ema" in checkpoint and checkpoint["ema"] is not None)
        )
    )

    if use_ema:
        ema_state = checkpoint.get("ema", None)
        if isinstance(ema_state, dict) and ("shadow" in ema_state):
            shadow = ema_state["shadow"]

            # 只做 _orig_mod 解包与 module. 去前缀（与 ckpt 侧一致），不做 fn.2/3 互换
            shadow, _ = _unwrap_orig_mod_keys(shadow)
            shadow, _ = _strip_dataparallel_prefix(shadow)
            shadow_norm = { _normalize_key(k): v for k, v in shadow.items() }

            current = model.state_dict()
            # 反向映射：规范名 -> 模型真实键
            replaced = 0
            for nk, v in shadow_norm.items():
                rk = keymap.get(nk, None)
                if rk is not None:
                    current[rk] = v.to(dtype=current[rk].dtype)
                    replaced += 1
            model.load_state_dict(current, strict=False)
            if verbose:
                print(f"已应用 EMA 参数（替换 {replaced} 个权重，strict=False）。")
            report["used_ema"] = True
        else:
            if verbose:
                print("未在 checkpoint 中找到 EMA 权重，继续使用原始权重。")
    else:
        if verbose:
            print("按参数设置，不使用 EMA 权重。")

    return report
