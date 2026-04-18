from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.rdmolops import FastFindRings
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import torch
import numpy as np
import scipy
import scipy.sparse as ss
import scipy.sparse.linalg
import math
import json
import itertools as it
import re
import logging
import sys
from config import CFG
import random

from rdkit_utils import smiles_to_mol_robust
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors

import rdkit.RDLogger as rkl
logger = rkl.logger()
logger.setLevel(rkl.ERROR)

import rdkit.rdBase as rkrb
rkrb.DisableLog('rdApp.error')

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from rdkit.Chem.rdmolops import FastFindRings
import pandas as pd
import numpy as np
import json
import os
import itertools
import time
import resource  # 新增
from datetime import datetime
from tqdm import tqdm
from scipy.spatial.distance import pdist
import multiprocessing
from joblib import Parallel, delayed
import pyarrow.parquet as pq

# RDKit imports
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors

# 检查 Avalon
try:
    from rdkit.Avalon import pyAvalonTools
    HAS_AVALON = True
except ImportError:
    HAS_AVALON = False
    print("⚠️  未检测到 Avalon 模块")

# 屏蔽日志
RDLogger.DisableLog('rdApp.*') 
def _rehydrate_computed_props(m: Chem.Mol) -> Chem.Mol:
    """
    在副本上恢复环信息与常用派生属性；不做 PROPERTIES/ADJUSTHS 之类的“化学有效性”检查。
    """
    mc = Chem.Mol(m)
    try:
        FastFindRings(mc)
    except Exception:
        try:
            Chem.GetSymmSSSR(mc)
        except Exception:
            pass
    try:
        Chem.SetAromaticity(mc)
        Chem.SetConjugation(mc)
        Chem.SetHybridization(mc)
    except Exception:
        pass
    mc.UpdatePropertyCache(strict=False)
    return mc


def standardize_parent_mol_from_smiles(smiles: str) -> Mol | None:
    """
    The original implementation is replaced by a call to smiles_to_mol_robust.
    """
    return smiles_to_mol_robust(smiles)

# 50w metabolites fpbit relative aboundance > 5%
FPBitIdx = [1, 5, 13, 41, 69, 80, 84, 94, 114, 117, 118, 119, 125, 133, 145,
            147, 191, 192, 197, 202, 222, 227, 231, 249, 283, 294, 310, 314,
            322, 333, 352, 361, 378, 387, 389, 392, 401, 406, 441, 478, 486,
            489, 519, 521, 524, 555, 561, 591, 598, 599, 610, 622, 650, 656,
            667, 675, 677, 679, 680, 694, 695, 715, 718, 722, 729, 736, 739,
            745, 750, 760, 775, 781, 787, 794, 798, 802, 807, 811, 823, 835,
            841, 849, 869, 872, 874, 875, 881, 890, 896, 926, 935, 980, 991,
            1004, 1009, 1017, 1019, 1027, 1028, 1035, 1037, 1039, 1057, 1060,
            1066, 1070, 1077, 1088, 1097, 1114, 1126, 1136, 1142, 1143, 1145,
            1152, 1154, 1160, 1162, 1171, 1181, 1195, 1199, 1202, 1218, 1234,
            1236, 1243, 1257, 1267, 1274, 1279, 1283, 1292, 1294, 1309, 1313,
            1323, 1325, 1349, 1356, 1357, 1366, 1380, 1381, 1385, 1386, 1391,
            1399, 1436, 1440, 1441, 1444, 1452, 1454, 1457, 1475, 1476, 1477,
            1480, 1487, 1516, 1536, 1544, 1558, 1564, 1573, 1599, 1602, 1604,
            1607, 1619, 1648, 1670, 1683, 1693, 1716, 1722, 1737, 1738, 1745,
            1747, 1750, 1754, 1755, 1764, 1781, 1803, 1808, 1810, 1816, 1838,
            1844, 1847, 1855, 1860, 1866, 1873, 1905, 1911, 1917, 1921, 1923,
            1928, 1933, 1950, 1951, 1970, 1977, 1980, 1984, 1991, 2002, 2033, 2034, 2038]

def conv_out_dim(length_in, kernel, stride, padding, dilation):
    length_out = (length_in + 2 * padding - dilation * (kernel - 1) - 1)// stride + 1
    return length_out

def filter_ms(ms, thr=0.05, max_mz=2000):
    mz = []
    intn = []
    maxi = 0
    for m, i in ms:
        if m < max_mz and i > maxi:
            maxi = i

    for m, i in ms:
        if m < max_mz and i/maxi > thr:
            mz.append(m)
            intn.append(round(i/maxi*100, 2))

    return mz, intn

def calc_nls(ms, thr=0.05, max_mz=2000):
    mz, intn = filter_ms(ms, thr=0.05, max_mz=2000)

    nlmass = []
    nlintn = []
    for a, b in it.combinations(mz[::-1], 2):
        nl = a - b
        if 0 < nl < 200:
            nlmass.append(round(nl, 5))
            idxa = mz.index(a)
            idxb = mz.index(b)
            nlintn.append(round((intn[idxa]+intn[idxb])/2., 5))

    nls = sorted(list(zip(nlmass, nlintn)))
    return nls

def ms_binner(ms, nls=[], min_mz=20, max_mz=2000, bin_size=0.05, add_nl=False, binary_intn=False):
    """
    Convert the given spectrum to a binned sparse SciPy vector.
    """
    if add_nl and not nls:
        nls = calc_nls(ms, max_mz=max_mz)

    nltensor = None
    mz, intn = filter_ms(ms)

    if add_nl:
        nlmass = []
        nlintn = []

        if not nls:
            nls = calc_nls(ms, max_mz=max_mz)

        for m, i in nls:
            if m < 200:
                if binary_intn:
                    i = 1
                nlmass.append(m)
                nlintn.append(i)

        nlmass = np.array(nlmass)
        nlintn = np.array(nlintn)
        if len(nlintn) > 0:
            nlintn = nlintn/nlintn.max()
        num_nlbins = math.ceil((200) / bin_size)
        nlbins = (nlmass / bin_size).astype(np.int32)

        if len(nlmass) > 0:
            vecnl = ss.csr_matrix(
                (nlintn,
                (np.repeat(0, len(nlintn)), nlbins)),
                shape=(1, num_nlbins),
                dtype=np.float32)

            vecnl = (vecnl / scipy.sparse.linalg.norm(vecnl)*100)
            nltensor = torch.FloatTensor(vecnl.todense()).view(-1)
        else:
            nltensor = torch.zeros(num_nlbins)

    mz = np.array(mz)
    keepidx = (mz <= max_mz)
    mz = mz[keepidx]
    intn = np.array(intn)
    intn = intn[keepidx]

    if binary_intn:
        intn[intn > 0] = 1.0
    elif len(intn) > 0:
        intn = intn/intn.max()

    num_bins = math.ceil((max_mz - min_mz) / bin_size)
    bins = ((mz - min_mz) / bin_size).astype(np.int32)

    if len(mz) > 0:
        vec = ss.csr_matrix(
            (intn,
            (np.repeat(0, len(intn)), bins)),
            shape=(1, num_bins),
            dtype=np.float32)

        if not binary_intn:
            vec = (vec / scipy.sparse.linalg.norm(vec)*100)

        mstensor = torch.FloatTensor(vec.todense()).view(-1)
    else:
        mstensor = torch.zeros(num_bins)

    if not nltensor is None:
        return torch.cat([nltensor, mstensor], dim=0)

    return mstensor

def formula2vec(formula, elements=['C', 'H', 'O', 'N', 'P', 'S', 'P', 'F', 'Cl', 'Br']):
    formula_p = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    vec = np.zeros(len(elements))
    for i in range(len(formula_p)):
        ele = formula_p[i][0]
        num = formula_p[i][1]
        if num == '':
            num = 1
        else:
            num = int(num)
        if ele in elements:
            vec[elements.index(ele)] += num
    return np.array(vec)


def mol_fp_encoder0(smiles, tp='rdkit', nbits=2048):
    mol = standardize_parent_mol_from_smiles(smiles)
    if mol is None:
        return None, None
    mol_fp = _rehydrate_computed_props(mol)
    def _calc_on(mol_ready, tp, nbits):
        if tp == 'morgan':
            fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol_ready, 2, nBits=nbits)
        elif tp == 'morgan1':
            fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol_ready, 1, nBits=2048)
        elif tp == 'macc':
            fp_vec = MACCSkeys.GenMACCSKeys(mol_ready)
        elif tp == 'rdkit':
            fp_vec = Chem.RDKFingerprint(mol_ready, nBitsPerHash=1)
        elif tp =='morgan3+torsion':
            fp_vec1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_ready, 3, nBits=nbits)
            fp_vec2 = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol_ready, nBits=nbits)
            fp_vec=fp_vec1+fp_vec2
        else:
            raise ValueError(f'Unknown fp type: {tp}')
        arr = np.frombuffer(fp_vec.ToBitString().encode(), 'u1') - ord('0')
        if tp == 'morgan1':
            arr = arr[FPBitIdx]
        return torch.FloatTensor(arr.tolist())
    try:
        fp = _calc_on(mol_fp, tp, nbits)
    except RuntimeError as e:
        if 'RingInfo not initialized' in str(e):
            mol_fp = _rehydrate_computed_props(mol_fp)
            fp = _calc_on(mol_fp, tp, nbits)
        else:
            raise
    return fp, mol_fp


def mol_fp_encoder(smiles, tp='rdkit', nbits=2048):
    fpenc, _ = mol_fp_encoder0(smiles, tp, nbits)
    return fpenc

def mol_fp_fm_encoder(smiles, tp='rdkit', nbits=2048):
    fmenc = None
    fpenc, mol = mol_fp_encoder0(smiles, tp, nbits)
    if not mol is None:
        fm = CalcMolFormula(mol)
        fmenc = torch.FloatTensor(formula2vec(fm))
    return fpenc, fmenc

def smi2fmvec(smiles):
    mol = standardize_parent_mol_from_smiles(smiles)
    if mol is None:
        return None
    fm = CalcMolFormula(mol)
    fmenc = torch.FloatTensor(formula2vec(fm))
    return fmenc

def mol_graph_featurizer(smiles):
    from GNN import featurizer as ft
    mol = standardize_parent_mol_from_smiles(smiles)
    if mol is None:
        return {}
    smiles_std = Chem.MolToSmiles(mol, isomericSmiles=False, kekuleSmiles=True)

    try:
        gnn_cfg = CFG.model.mol_encoder.gnn
    except Exception:
        class _D:
            use_subgraph_fp = False
            subgraph_fp_radius = 2
            subgraph_fp_nbits = 256
        gnn_cfg = _D()

    use_sub = bool(getattr(gnn_cfg, 'use_subgraph_fp', False))
    radius  = int(getattr(gnn_cfg, 'subgraph_fp_radius', 2))
    nbits   = int(getattr(gnn_cfg, 'subgraph_fp_nbits', 256))

    mol_graph = ft.calc_data_from_smile(
        smiles_std,
        addh=False,
        with_ring_conj=True,
        with_atom_feats=True,
        with_submol_fp=use_sub,          # ← 开关
        radius=radius,                   # ← 半径
        fp_nbits=nbits                   # ← 位数
    )
    return mol_graph

def pad_V(v, max_n):
    """分子节点特征V的padding（保留第二组的鲁棒性设计）"""
    pad_size = max_n - v.shape[0]
    if pad_size > 0:
        return torch.cat([v, torch.zeros(pad_size, v.shape[1], dtype=v.dtype, device=v.device)], dim=0)
    return v

def pad_A(a, max_n):
    """分子邻接矩阵A的padding（改为第一组的填充逻辑，保留鲁棒性）"""
    N, L, _ = a.shape  # 先获取原始节点数N、边类型数L
    pad_size = max_n - N
    if pad_size > 0:
        # 第一步：在最后一维（列维度）补0，形状从[N, L, N] → [N, L, max_n]
        pad_last_dim = torch.zeros(N, L, pad_size, dtype=a.dtype, device=a.device)
        a = torch.cat([a, pad_last_dim], dim=-1)
        
        # 第二步：在第0维（行维度）补0，形状从[N, L, max_n] → [max_n, L, max_n]
        pad_0_dim = torch.zeros(pad_size, L, max_n, dtype=a.dtype, device=a.device)
        a = torch.cat([a, pad_0_dim], dim=0)
    return a
    
import torch.nn.functional as F

def pad_A_to_channel_first(A, max_n):
    """
    输入 A: [N, N] 或 [N, N, L]
    输出: [L, max_n, max_n] (即 Channels First)
    """
    
    if A.dim() == 2:
        A = A.unsqueeze(1)  # 在第1维插入通道维度，得到 [N, 1, N]
    
    # 处理3维输入 [N, N, L] -> [N, L, N]
    elif A.dim() == 3:
        A = A.permute(0, 2, 1)  # 维度置换：[N,N,L] -> [N,L,N]
    
    # 获取当前的N维度大小（第一个维度）
    curr_n = A.shape[0]
    
    # 裁剪：如果当前大小超过max_n，裁剪到max_n
    if curr_n > max_n:
        A = A[:max_n, :, :max_n]
    
    # 填充：如果当前大小小于max_n，补0到max_n
    elif curr_n < max_n:
        pad_len = max_n - curr_n
        # F.pad填充顺序：(左, 右, 上, 下, 前, 后)
        # 这里需要填充：
        # - 最后一维（第三个维度）右侧补pad_len: (0, pad_len)
        # - 第一维度（第一个维度）前侧补0、后侧补pad_len: 对应最后两位 (0, pad_len)
        A = F.pad(A, (0, pad_len, 0, 0, 0, pad_len), value=0.0)
    
    # 大小正好等于max_n时直接返回
    return A
    
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    

def _segment_ids(size_list, device):
    sizes = torch.as_tensor(size_list, dtype=torch.long, device=device)
    ids = torch.repeat_interleave(torch.arange(sizes.numel(), device=device), sizes)
    return ids, sizes

def _flatten_feat(x):
    # x: [N, *F]  ->  x2: [N, C], feat_shape: tuple(*F)
    x = x if torch.is_tensor(x) else torch.as_tensor(x)
    feat_shape = x.shape[1:]
    x2 = x.reshape(x.shape[0], -1)
    return x2, feat_shape

def _unflatten_feat(y2, feat_shape):
    # y2: [S or N, C] -> [S or N, *F]
    return y2.reshape(y2.shape[0], *feat_shape)

def segment_sum(x, size_list):
    """
    对 x 的第0维按 size_list 分段求和，保持其余维度不变。
    输入:  [N, *F] / [N]
    输出:  [S, *F]，S=len(size_list)
    """
    x2, feat_shape = _flatten_feat(x)
    ids, sizes = _segment_ids(size_list, x2.device)
    S = int(sizes.numel())

    out2 = x2.new_zeros((S, x2.shape[1]))
    out2.index_add_(0, ids, x2)               # [S, C]

    return _unflatten_feat(out2, feat_shape)

def segment_max(x, size_list):
    """
    对 x 的第0维按 size_list 分段取最大，保持其余维度不变。
    输入:  [N, *F] / [N]
    输出:  [S, *F]
    """
    x2, feat_shape = _flatten_feat(x)
    ids, sizes = _segment_ids(size_list, x2.device)
    S = int(sizes.numel())

    out2 = x2.new_full((S, x2.shape[1]), -float("inf"))
    # 优先用稳定版 scatter_reduce_，老版本则回退到 index_reduce_
    if hasattr(out2, "scatter_reduce_"):
        index = ids[:, None].expand(-1, x2.size(1))    # [N, C]
        out2.scatter_reduce_(0, index, x2, reduce="amax", include_self=True)
    else:
        out2.index_reduce_(0, ids, x2, reduce='amax', include_self=True)

    return _unflatten_feat(out2, feat_shape)

def segment_softmax(gate, size_list):
    """
    在每个 segment 内沿第0维做 softmax（数值稳定），其余维度保持不变。
    输入:  [N, *F] / [N]
    输出:  [N, *F]
    """
    g2, feat_shape = _flatten_feat(gate)
    ids, _ = _segment_ids(size_list, g2.device)

    # 分段最大（[S, C]），再拉回 [N, C] 做稳定 softmax
    segmax2 = segment_max(g2, size_list).reshape(-1, g2.shape[1])   # [S, C]
    g_center = g2 - segmax2.index_select(0, ids)                    # [N, C]

    exp2 = torch.exp(g_center)
    segsum2 = segment_sum(exp2, size_list).reshape(-1, g2.shape[1]) # [S, C]
    denom2 = segsum2.index_select(0, ids) + 1e-16                   # [N, C]

    attn2 = exp2 / denom2                                           # [N, C]
    return _unflatten_feat(attn2, feat_shape)



def pad_ms_list(ms_list, thr=0.05, min_mz=20, max_mz=2000):
    thr = thr*100
    mslst = []
    for ms in ms_list:
        ms = np.array(ms)
        ms[:,1] = ms[:,1]/ms[:,1].max()*100

        if thr > 0:
            ms = ms[(ms[:,1] >= thr)]

        ms = ms[(ms[:,0] >= min_mz)]
        ms = ms[(ms[:,0] <= max_mz)]

        mslst.append(ms)

    size_list = [ms.shape[0] for ms in mslst]
    maxlen = max(size_list)

    l = []
    for ms in mslst:
        extn = maxlen-len(ms)
        if extn > 0:
            l.append(np.concatenate([ms, [[0,0]]*extn], axis=0))
        else:
            l.append(ms)

    return torch.FloatTensor(np.stack(l)), torch.IntTensor(size_list)

def setup_logging(log_path):
    """
    配置日志系统，使其能够同时输出到控制台和指定的文件。
    """
    log_format = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    logging.info(f"日志系统已启动，日志将记录到: {log_path}")


def capture_rng_state():
    """
    返回一个可序列化的 RNG 状态字典，用于在 checkpoint 中保存。
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state()
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["torch_cuda_all"] = None
    else:
        state["torch_cuda_all"] = None
    return state

def restore_rng_state(state: dict):
    """
    从字典中恢复 RNG 状态。键不存在时自动跳过。
    """
    try:
        if "python" in state and state["python"] is not None:
            random.setstate(state["python"])
    except Exception:
        pass

    try:
        if "numpy" in state and state["numpy"] is not None:
            np.random.set_state(state["numpy"])
    except Exception:
        pass

    try:
        if "torch_cpu" in state and state["torch_cpu"] is not None:
            torch.set_rng_state(state["torch_cpu"])
    except Exception:
        pass

    try:
        if torch.cuda.is_available() and state.get("torch_cuda_all") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    except Exception:
        pass
    

class ModelEMA:
    """
    维护一份指数滑动平均（跟踪需要梯度的浮点参数和所有缓冲区）；
    默认与模型在同一设备上以获得最佳性能。
    """
    def __init__(self, model, decay: float = 0.999, device=None):
        self.decay = float(decay)

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        self.shadow = {}
        self.backup = {}
        
        # Register both parameters and buffers
        # 1. Register trainable parameters
        for name, p in model.named_parameters():
            if p.requires_grad and p.data.dtype.is_floating_point:
                self.shadow[name] = p.detach().to(device=self.device, dtype=torch.float32, copy=True)
        
        # 2. Register all buffers (e.g., running_mean, running_var for BatchNorm)
        for name, b in model.named_buffers():
            self.shadow[name] = b.detach().to(device=self.device, copy=True)

        self._applied = False

    @torch.no_grad()
    def update(self, model):
        """在每次 optimizer.step() 后调用"""
        m = self.decay

        # 1. Update parameters with exponential decay
        for name, p in model.named_parameters():
            if name in self.shadow and p.requires_grad:
                new_param_data = p.detach().to(dtype=torch.float32)
                self.shadow[name].mul_(m).add_(new_param_data, alpha=(1.0 - m))

        # 2. Copy buffers directly (no decay)
        for name, b in model.named_buffers():
            if name in self.shadow:
                self.shadow[name].copy_(b.detach())

    @torch.no_grad()
    def apply_to(self, model):
        """把模型权重临时替换成 EMA 影子权重（用于验证/导出）"""
        if self._applied:
            return
        self.backup = {}

        # 1. Swap parameters
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].to(dtype=p.dtype))

        # 2. Swap buffers
        for name, b in model.named_buffers():
            if name in self.shadow:
                self.backup[name] = b.detach().clone()
                b.data.copy_(self.shadow[name])
                
        self._applied = True

    @torch.no_grad()
    def restore(self, model):
        """把模型权重从 backup 恢复回来。"""
        if not self._applied:
            return
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        for name, b in model.named_buffers():
            if name in self.backup:
                b.data.copy_(self.backup[name])
        self.backup.clear()
        self._applied = False

    def state_dict(self):
        # 保存时，为了可移植性，还是移到 CPU
        return {
            "decay": self.decay,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()}
        }

    def load_state_dict(self, state):
        self.decay = float(state.get("decay", self.decay))
        sd = state.get("shadow", {})
        # 加载时，将影子权重恢复到 self.device
        self.shadow = {k: v.clone().to(self.device) for k, v in sd.items()}
        self.backup = {}
        self._applied = False