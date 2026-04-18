import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, to_dense_adj
from graph_transformer_pytorch import GraphTransformer
from collections import defaultdict
import os
import time
import json
from typing import Dict, List, Tuple, Optional
import shutil 

class FragmentTreeProcessor:
    ELEMENT_ORDER = ['C','H','N','O','S','P','F','Cl','Br','I','Se','Si','B']

    @staticmethod
    def parse_formula(formula: str) -> Dict[str, int]:
        counts = defaultdict(int)
        current = ""
        for char in formula:
            if char.isupper():
                if current:
                    element = ''.join([c for c in current if not c.isdigit()])
                    num = ''.join([c for c in current if c.isdigit()])
                    counts[element] = int(num) if num else 1
                current = char
            else:
                current += char
        if current:
            element = ''.join([c for c in current if not c.isdigit()])
            num = ''.join([c for c in current if c.isdigit()])
            counts[element] = int(num) if num else 1
        return counts

    @staticmethod
    def _compute_dbe(elem_counts: Dict[str, int]) -> float:
        # 基于通用价电子数的 DBE ≈ 1 + 0.5*Σ n_i*(v_i - 2)
        valence = {'C':4,'H':1,'N':3,'O':2,'S':2,'P':3,'F':1,'Cl':1,'Br':1,'I':1,'Se':2,'Si':4,'B':3}
        d = 1.0
        for e, n in elem_counts.items():
            v = valence.get(e, 0)
            d += 0.5 * n * (v - 2)
        return float(max(0.0, d))

    @staticmethod
    def json_to_pyg(
        data_dict: Dict,
        edge_features: bool = True,
        enhanced: bool = False
    ) -> Data:
        frag_tree = data_dict["frag_tree"]

        # --- 建图、找根、BFS 深度、子树规模 ---
        num_nodes = len(frag_tree['fragments'])
        indeg = [0]*num_nodes
        children = [[] for _ in range(num_nodes)]
        node_mz_list = []
        node_int_list = []
        node_score_list  = []
        elem_counts_list = []

        # 先收集所有节点属性
        for i, frag in enumerate(frag_tree['fragments']):
            counts = FragmentTreeProcessor.parse_formula(frag['molecularFormula'])
            elem_counts_list.append(counts)
            node_mz_list.append(float(frag['mz']))
            node_int_list.append(float(frag['intensity']))
            node_score_list.append(float(frag['score']))

        # 再填充有向边与入度
        for loss in frag_tree['losses']:
            s = loss['sourceFragmentIdx']; t = loss['targetFragmentIdx']
            children[s].append(t); indeg[t] += 1

        # 现在才能按 m/z 对每个父节点的子节点进行稳定排序
        for u in range(num_nodes):
            children[u].sort(key=lambda v: (node_mz_list[v], v))

        # 只进行一次、确定性的根节点选择：
        # 1) 优先在入度为 0 的节点中选强度最大、m/z 最大、索引最大的
        # 2) 若不存在入度为 0（如有环），则在全体节点中按相同规则选择
        roots = [i for i in range(num_nodes) if indeg[i] == 0]
        if roots:
            root = max(roots, key=lambda i: (node_int_list[i], node_mz_list[i], i))
        else:
            root = max(range(num_nodes), key=lambda i: (node_int_list[i], node_mz_list[i], i)) if num_nodes > 0 else 0
        root_int = max(1e-8, node_int_list[root])

        # BFS 计算深度
        depth = [-1]*num_nodes
        from collections import deque
        q = deque([root]); depth[root] = 0
        order = []
        while q:
            u = q.popleft(); order.append(u)
            for v in children[u]:
                if depth[v] < 0:
                    depth[v] = depth[u] + 1; q.append(v)
        for i in range(num_nodes):
            if depth[i] < 0: depth[i] = 0  # 断开成分兜底

        # 逆序后序，计算子树规模（含自身）
        subtree_size = [1]*num_nodes
        for u in reversed(order):
            for v in children[u]:
                subtree_size[u] += subtree_size[v]

        # --- 组装节点特征 ---
        node_features = []
        for i in range(num_nodes):
            counts = elem_counts_list[i]
            if not enhanced:
                # feats = [node_mz_list[i], node_int_list[i], node_score_list[i]]   # 加入score
                feats = [node_mz_list[i], node_int_list[i] ]
            else:
                rel_int = node_int_list[i] / root_int
                dbe = FragmentTreeProcessor._compute_dbe(counts)
                # feats = [node_mz_list[i], rel_int, node_score_list[i], float(depth[i]), dbe, float(subtree_size[i])]   #加入score
                feats = [node_mz_list[i], rel_int, float(depth[i]), dbe, float(subtree_size[i])]
            feats += [counts.get(e, 0) for e in FragmentTreeProcessor.ELEMENT_ORDER]
            node_features.append(feats)


        # --- 组装边 ---
        edge_index = [[], []]
        edge_attrs  = []
        for loss in frag_tree['losses']:
            s = loss['sourceFragmentIdx']; t = loss['targetFragmentIdx']
            edge_index[0].append(s); edge_index[1].append(t)
            if edge_features:
                counts = FragmentTreeProcessor.parse_formula(loss['molecularFormula'])
                if enhanced:
                    delta_mz = max(0.0, node_mz_list[s] - node_mz_list[t])
                    # ea = [delta_mz] + [float(loss['score'])] + [counts.get(e, 0) for e in FragmentTreeProcessor.ELEMENT_ORDER]
                    ea = [delta_mz] + [counts.get(e, 0) for e in FragmentTreeProcessor.ELEMENT_ORDER]
                else:
                    # ea = [float(loss['score'])] + [counts.get(e, 0) for e in FragmentTreeProcessor.ELEMENT_ORDER] # 加入score
                    ea = [counts.get(e, 0) for e in FragmentTreeProcessor.ELEMENT_ORDER]
                edge_attrs.append(ea)

        edge_attr_tensor = None
        if edge_features:
            D = (1 + len(FragmentTreeProcessor.ELEMENT_ORDER)) if enhanced else (0 + len(FragmentTreeProcessor.ELEMENT_ORDER))
            if len(edge_attrs) > 0:
                edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float)
            else:
                edge_attr_tensor = torch.zeros((0, D), dtype=torch.float)

        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=edge_attr_tensor
        )

class GraphTransformerEncoder(nn.Module):
    def __init__(self,
                 input_dim: int = 15,
                 edge_dim: int = 13,
                 total_depth: int = 2,
                 hidden_dim: int = 256,
                 heads1: int = 8,
                 heads2: int = 1,
                 pool_out_dim: int = 256,
                 norm_type: str = 'layernorm'
                 ):
        super().__init__()
        self.norm_type = norm_type

        if norm_type == 'layernorm':
            norm_layer_fn = lambda dim: nn.LayerNorm(dim)
        elif norm_type == 'batchnorm':
            norm_layer_fn = lambda dim: nn.BatchNorm1d(dim)
        else:
            raise ValueError(f"GraphTransformerEncoder: Unknown norm_type: {norm_type}")

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            norm_layer_fn(hidden_dim)
        )

        self.gt1 = GraphTransformer(
            dim=hidden_dim,
            depth=(total_depth + 1)//2,
            heads=heads1,
            edge_dim=edge_dim,
            with_feedforwards=True,
            gated_residual=True,
            rel_pos_emb=False,
            accept_adjacency_matrix=True,
        )

        self.post_gt1 = nn.Sequential(norm_layer_fn(hidden_dim), nn.ELU())

        self.gt2 = GraphTransformer(
            dim=hidden_dim,
            depth=max(1, total_depth//2),
            heads=heads2,
            edge_dim=edge_dim,
            with_feedforwards=True,
            gated_residual=True,
            rel_pos_emb=False,
            accept_adjacency_matrix=True,
        )

        self.post_gt2 = nn.Sequential(norm_layer_fn(hidden_dim), nn.ELU())

        self.pool = nn.Sequential(
            nn.Linear(2 * hidden_dim, pool_out_dim),
            nn.ELU(),
            nn.LayerNorm(pool_out_dim)
        )



    def forward(self, data: Data, return_nodes: bool = False, return_mz: bool = False):
        x_raw, edge_index, batch = data.x, data.edge_index, data.batch

        # dense nodes + mask
        x, mask = to_dense_batch(x_raw, batch)          # x: [B, N, Cin]
        B, N, Cin = x.shape
        node_mz = x[:, :, 0]

        # project to 256
        x = x.reshape(-1, Cin)
        x = self.proj(x)                                # [B*N, 256]
        x = x.reshape(B, N, -1)                         # [B, N, 256]

        edge_feat = None
        if getattr(data, 'edge_attr', None) is not None and data.edge_attr.numel() > 0:
            ones = torch.ones((data.edge_attr.size(0), 1),
                            dtype=x.dtype, device=data.edge_attr.device)
            aug_attr = torch.cat([data.edge_attr.to(x.dtype), ones], dim=1)   # [E, Fe+1]

            ef = to_dense_adj(edge_index, batch, edge_attr=aug_attr)          # [B, N, N, Fe+1]
            adj = ef[..., -1]                                                 # [B, N, N]，边处为 1
            edge_feat = ef[..., :-1].to(x.dtype)                               # [B, N, N, Fe]
        else:
            adj = to_dense_adj(edge_index, batch).to(x.dtype)                  # [B, N, N]

        # GT blocks
        x, _ = self.gt1(x, adj_mat=adj, mask=mask, edges=edge_feat)
        if self.norm_type == 'batchnorm':
            x = self.post_gt1(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.post_gt1(x)

        x, _ = self.gt2(x, adj_mat=adj, mask=mask, edges=edge_feat)
        
        if self.norm_type == 'batchnorm':
            x = self.post_gt2(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.post_gt2(x)

        if return_nodes:
            pad_mask = ~mask
            if return_mz:
                return x, pad_mask, node_mz
            return x, pad_mask

        # pool
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1).values
        x_pooled = torch.cat([x_mean, x_max], dim=-1)
        return self.pool(x_pooled)  # [B, 256]
