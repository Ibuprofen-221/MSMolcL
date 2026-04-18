from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import inchi
import re
from typing import Optional

AROMATIC_ALLOWED = {5, 6, 7, 8, 15, 16, 33, 34}  # Atomic numbers for B, C, N, O, P, S, As, Se


def smiles_to_mol_robust(smiles: str) -> Optional[Chem.Mol]:
    """
    1) First try RDKit's default strict parsing + full sanitize
       (includes valence/chemical reasonableness checks).
    2) If that fails, do lenient parsing (strictParsing=False, sanitize=False),
       then perform a "partial sanitize": skip PROPERTIES/ADJUSTHS (to avoid
       valence errors) but keep ring perception, Kekulization, aromaticity,
       conjugation, hybridization, etc., to preserve usability and comparability.
    Returns a Mol that is *as acceptable as possible* for downstream functions;
    it may still be None for extremely bad inputs.
    """
    if not smiles or not smiles.strip():
        return None

    # Path A: standard strict mode
    try:
        m = Chem.MolFromSmiles(smiles)  # sanitize=True, strictParsing=True (defaults)
        m.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(m)
        m.ClearComputedProps()
        return m
    except Exception:
        pass

    # Path B: lenient parsing + partial sanitize
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        return None

    # Skip only the "chemical validity/valence-related" steps; keep the rest for functionality/comparability
    # - SANITIZE_PROPERTIES performs valence checks (most common explicit/implicit valence errors)
    # - SANITIZE_ADJUSTHS can also trigger related issues, so skip it as well
    keep_flags = (
        Chem.SanitizeFlags.SANITIZE_ALL
        ^ (Chem.SanitizeFlags.SANITIZE_PROPERTIES | Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    )
    try:
        Chem.SanitizeMol(m, sanitizeOps=keep_flags)
    except Exception:
        # Fall back to a "minimal usable set" so MolToSmiles/fingerprints/substructure are likely usable
        minimal_flags = (
            Chem.SanitizeFlags.SANITIZE_CLEANUP
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
            | Chem.SanitizeFlags.SANITIZE_KEKULIZE
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        )
        Chem.SanitizeMol(m, sanitizeOps=minimal_flags)

    m.UpdatePropertyCache(strict=False)
    m.ClearComputedProps()
    # Provide a visible flag to the caller: this molecule was "partially sanitized"
    m.SetBoolProp("_PartialSanitized", True)
    return m


def canon_smiles_no_stereo(m: Chem.Mol) -> str:
    """
    Produce a stable, stereo-agnostic canonical SMILES.
    - In normal cases: use aromatic SMILES (kekuleSmiles=False), which is more stable across resonance;
    - If “nonstandard elements are marked as aromatic” (e.g., Si, Ge…): force Kekulize to clear
      aromaticity and then use Kekulé canonical SMILES, to avoid lowercase nonstandard elements
      (e.g., [si]).
    """
    has_nonstd_arom = any(
        a.GetIsAromatic() and a.GetAtomicNum() not in AROMATIC_ALLOWED
        for a in m.GetAtoms()
    )

    if has_nonstd_arom:
        mc = Chem.Mol(m)
        try:
            # Clear aromaticity and choose a deterministic Kekulé form
            Chem.Kekulize(mc, clearAromaticFlags=True)
            return Chem.MolToSmiles(
                mc, canonical=True, isomericSmiles=False, kekuleSmiles=True
            )
        except Exception:
            # If Kekulize still fails in extreme cases, fall back to aromatic SMILES (very rare)
            pass

    # Regular stable path: aromatic canonical SMILES
    return Chem.MolToSmiles(
        m, canonical=True, isomericSmiles=False, kekuleSmiles=False
    )


def same_molecule_no_stereo(smi1: str, smi2: str) -> tuple[bool, str, str]:
    """
    Convenience comparison: returns (are_equal, canonical_smi1, canonical_smi2).
    """
    m1 = smiles_to_mol_robust(smi1)
    m2 = smiles_to_mol_robust(smi2)
    if m1 is None or m2 is None:
        return (False, "", "")
    s1 = canon_smiles_no_stereo(m1)
    s2 = canon_smiles_no_stereo(m2)
    return (s1 == s2, s1, s2)



def smiles_to_formula(smiles: str) -> Optional[str]:
    """
    将 SMILES 规范化后，返回分子式（Hill System）
    设计目标：
      1) 对同一 SMILES 输出确定且稳定；
      2) 对化学上共享同一分子式的不同 SMILES，输出完全相同；
      3) 尽可能覆盖更多“不严格”SMILES（与 smiles_to_mol_robust 的回退逻辑一致）；
      4) 最终无法生成时返回 None。

    细节：
      - 统一将同位素信息清零（不区分 D/T、13C 等），增强跨表示的一致性；
      - 多片段（"."）整体合并计数；
      - 跳过原子号为 0 的占位/虚原子（如 "*"）；
      - 氢计数策略：以重原子 GetTotalNumHs() 汇总 + 孤立氢原子（度数为 0 的 H），避免双计；
      - 输出遵循 Hill System：含 C 时 C、H 优先，其余元素字母序；不含 C 时全部字母序。
    """
    # 1) 通过项目的鲁棒规范化路径获取 Mol
    m = smiles_to_mol_robust(smiles)
    if m is None:
        return None

    # 2) 工作副本：清除同位素，避免同位素标签导致的式子差异
    mc = Chem.Mol(m)
    for a in mc.GetAtoms():
        a.SetIsotope(0)
    mc.UpdatePropertyCache(strict=False)

    # ---- 辅助：把“元素计数”按 Hill System 组装为分子式 ----
    def counts_to_hill(counts: dict[str, int]) -> str:
        if not counts:
            return ""
        counts = {k: int(v) for k, v in counts.items() if int(v) > 0}

        parts = []
        if "C" in counts:
            c = counts.pop("C")
            parts.append(("C", c))
            h = counts.pop("H", 0)
            if h:
                parts.append(("H", h))
            for el in sorted(counts.keys()):
                parts.append((el, counts[el]))
        else:
            for el in sorted(counts.keys()):
                parts.append((el, counts[el]))

        out = []
        for el, n in parts:
            out.append(el if n == 1 else f"{el}{n}")
        return "".join(out)

    # ---- 辅助：解析形如 "C2H6O.NaCl" 的式子为计数字典，并合并碎片 ----
    def parse_formula_to_counts(formula: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        if not formula:
            return counts
        for frag in formula.split("."):
            for el, num in re.findall(r"([A-Z][a-z]?)(\d*)", frag):
                n = int(num) if num else 1
                counts[el] = counts.get(el, 0) + n
        return counts

    # 3) 首选：RDKit 自带的分子式（通常已是 Hill System）
    try:
        f = rdMolDescriptors.CalcMolFormula(mc)  # 不区分同位素，因已清零 Isotope
        if f:
            # 保险起见再标准化一次（合并碎片并按 Hill System重排），确保完全确定性
            return counts_to_hill(parse_formula_to_counts(f)) or None
    except Exception:
        pass

    # 4) 回退：用 InChI 提取分子式层（对一些边缘结构更稳），再标准化
    try:
        ich = inchi.MolToInchi(mc)
        if ich and ich.startswith("InChI="):
            # 形如 "InChI=1S/<FORMULA>/<layers...>" 或 "InChI=1/<FORMULA>/..."
            rest = ich.split("/", 1)[-1]
            # 如果 rest 还以 "1S/" 或 "1/" 开头，再剥一层
            if rest.startswith("1S/") or rest.startswith("1/"):
                rest = rest.split("/", 1)[-1]
            formula_layer = rest.split("/", 1)[0]  # 只要第一层（分子式）
            if formula_layer:
                counts = parse_formula_to_counts(formula_layer)
                if counts:
                    return counts_to_hill(counts) or None
    except Exception:
        pass

    # 5) 最后回退：自行统计元素计数（跳过原子号 0；氢按“重原子总氢+孤立氢”）
    try:
        counts: dict[str, int] = {}
        total_H = 0
        for a in mc.GetAtoms():
            Z = a.GetAtomicNum()
            if Z == 0:
                continue  # 跳过占位/虚原子
            if Z == 1:
                # 孤立氢（如 [H-]）需要单独统计；与重原子相连的显式氢通过重原子总氢统计，避免双计
                if a.GetDegree() == 0:
                    total_H += 1
                continue
            # 重原子
            sym = a.GetSymbol()
            counts[sym] = counts.get(sym, 0) + 1
            h_here = int(a.GetTotalNumHs() or 0)  # 总氢（隐式+显式）计入
            if h_here:
                total_H += h_here

        if total_H:
            counts["H"] = counts.get("H", 0) + total_H

        if counts:
            return counts_to_hill(counts) or None
    except Exception:
        pass

    # 6) 实在不行
    return None
