import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Dict, Tuple


# 设置计算设备（GPU优先，没有则用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KAN_linear(nn.Module):
    """基于傅里叶变换的KAN线性层，用于替代传统的nn.Linear层"""
    def __init__(self, inputdim, outdim, gridsize, addbias=True):
        super(KAN_linear, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, outdim, inputdim, gridsize) / 
            (np.sqrt(inputdim) * np.sqrt(self.gridsize))
        )
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)

        k = torch.reshape(
            torch.arange(1, self.gridsize + 1, device=x.device),
            (1, 1, 1, self.gridsize)
        )
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)

        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))

        y = torch.einsum(
            "dbik,djik->bj", 
            torch.concat([c, s], axis=0),
            self.fouriercoeffs
        )

        if self.addbias:
            y += self.bias

        y = y.view(outshape)
        return y


class NaiveFourierKANLayer(nn.Module):
    """基于傅里叶KAN的图卷积层（适配自定义图结构）"""
    def __init__(self, in_feats, out_feats, gridsize, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        # 保持初始化逻辑不变
        self.gridsize = gridsize
        self.addbias = addbias
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, out_feats, in_feats, gridsize) / 
            (np.sqrt(in_feats) * np.sqrt(gridsize))
        )
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(out_feats))

    def forward(self, V: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        适配自定义图结构的前向传播（修复维度不匹配问题）
        :param V: 节点特征 [batch_size, num_nodes, in_feats]
        :param A: 邻接矩阵 [batch_size, num_nodes, adj_chans, num_nodes]
        :return: 更新后的节点特征 [batch_size, num_nodes, out_feats]
        """
        batch_size, num_nodes, adj_chans, _ = A.shape  # [B, N, L, N]
        
        # 1. 傅里叶变换处理节点特征（保持不变）
        k = torch.reshape(
            torch.arange(1, self.gridsize + 1, device=V.device),
            (1, 1, 1, 1, self.gridsize)
        )
        v_rshp = V.view(batch_size, num_nodes, 1, self.in_feats, 1)  # [B, N, 1, C, 1]
        
        cos_kx = torch.cos(k * v_rshp)  # [B, N, 1, C, K]
        sin_kx = torch.sin(k * v_rshp)  # [B, N, 1, C, K]
        
        # 2. 计算傅里叶特征变换（保持不变）
        cos_sin = torch.concat([cos_kx, sin_kx], dim=2)  # [B, N, 2, C, K]
        fourier_feat = torch.einsum(
            "bndck,dock->bno",  # [B, N, 2, C, K] → [B, N, out_feats]
            cos_sin,
            self.fouriercoeffs
        )  # [B, N, out_feats]

        # 3. 邻接矩阵聚合（修复维度不匹配）
        # 步骤1：调整A的维度顺序为 [B, N, N, L]（将边特征通道放在最后）
        A_permuted = A.permute(0, 1, 3, 2)  # [B, N, L, N] → [B, N, N, L]
        # 步骤2：重塑为 [B, N, N*L]（合并邻居节点和边特征通道）
        A_reshaped = A_permuted.reshape(batch_size, num_nodes, -1)  # [B, N, N*L]
        # 步骤3：转置为 [B, N*L, N]（满足bmm的维度要求：[B, n, m] × [B, m, p]）
        A_transposed = A_reshaped.transpose(1, 2)  # [B, N*L, N]
        # 步骤4：矩阵乘法聚合邻居特征 → [B, N*L, out_feats]
        aggregated = torch.bmm(A_transposed, fourier_feat)  # [B, N*L, F]
        # 步骤5：重塑为 [B, N, L, F] 并对边特征通道求和 → [B, N, F]
        aggregated = aggregated.reshape(batch_size, num_nodes, adj_chans, self.out_feats)  # [B, N, L, F]
        aggregated = aggregated.sum(dim=2)  # 合并边特征通道 → [B, N, F]

        # 4. 添加偏置（保持不变）
        if self.addbias:
            aggregated += self.bias

        return aggregated


class KA_GNN_two(nn.Module):
    """简化版KA-GNN模型（适配自定义图结构）"""
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(KA_GNN_two, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.layers = nn.ModuleList()

        self.leaky_relu = nn.LeakyReLU()
        self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)

        for _ in range(num_layers - 1):
            self.layers.append(
                NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias)
            )

        self.linear_1 = KAN_linear(hidden_feat, out, 1, addbias=True)
        
        # 池化操作（适配自定义图结构）
        self.pool_func = self._get_pool_func(pooling)

        self.Readout = nn.Sequential(
            self.linear_1,
            nn.Sigmoid()
        )

    def _get_pool_func(self, pooling: str) -> callable:
        """获取池化函数（考虑分子实际大小，忽略padding）"""
        def sum_pool(V: torch.Tensor, mol_size: torch.Tensor) -> torch.Tensor:
            return self._segment_operation(V, mol_size, operation=torch.sum)
        
        def avg_pool(V: torch.Tensor, mol_size: torch.Tensor) -> torch.Tensor:
            sum_v = self._segment_operation(V, mol_size, operation=torch.sum)
            return sum_v / mol_size.unsqueeze(1).float()
        
        def max_pool(V: torch.Tensor, mol_size: torch.Tensor) -> torch.Tensor:
            return self._segment_operation(V, mol_size, operation=torch.max)

        pool_map = {
            'sum': sum_pool,
            'avg': avg_pool,
            'max': max_pool
        }
        return pool_map[pooling]

    def _segment_operation(self, V: torch.Tensor, mol_size: torch.Tensor, operation: callable) -> torch.Tensor:
        """对每个分子的节点特征进行分段聚合（忽略padding）"""
        batch_size = V.shape[0]
        results = []
        for i in range(batch_size):
            # 截取实际分子节点（去除padding）
            actual_nodes = V[i, :mol_size[i], :]
            # 执行聚合操作
            if operation == torch.max:
                pooled = operation(actual_nodes, dim=0)[0]  # max返回(values, indices)
            else:
                pooled = operation(actual_nodes, dim=0)
            results.append(pooled)
        return torch.stack(results, dim=0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        :param batch: 包含以下键的字典
            - 'V': 节点特征 [batch_size, num_nodes, in_feat]
            - 'A': 邻接矩阵 [batch_size, num_nodes, adj_chans, num_nodes]
            - 'mol_size': 每个分子的实际节点数 [batch_size]
        :return: 图级输出 [batch_size, out]
        """
        V = batch['V']  # 节点特征
        A = batch['A']  # 邻接矩阵
        mol_size = batch['mol_size']  # 分子大小

        # 输入特征映射
        h = self.kan_line(V)

        # 图卷积层传播
        for layer in self.layers:
            m = layer(h, A)  # 使用邻接矩阵进行消息传递
            h = self.leaky_relu(torch.add(m, h))  # 残差连接

        # 图级特征聚合
        y = self.pool_func(h, mol_size)

        # 输出层
        out = self.Readout(y)
        return out

    def get_grad_norm_weights(self) -> nn.ParameterList:
        return self.parameters()


class KA_GNN(nn.Module):
    """完整版KA-GNN模型（适配自定义图结构）"""
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(KA_GNN, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)
        self.layers = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

        for _ in range(num_layers - 1):
            self.layers.append(
                NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias)
            )

        self.linear_1 = KAN_linear(hidden_feat, out_feat, grid_feat, addbias=use_bias)
        self.linear_2 = KAN_linear(out_feat, out, grid_feat, addbias=use_bias)
        
        # 池化函数
        self.pool_func = self._get_pool_func(pooling)

        self.Readout = nn.Sequential(
            self.linear_1,
            self.leaky_relu,
            self.linear_2,
            nn.Sigmoid()
        )

    def _get_pool_func(self, pooling: str) -> callable:
        """同KA_GNN_two的池化函数实现"""
        def sum_pool(V: torch.Tensor, mol_size: torch.Tensor) -> torch.Tensor:
            return self._segment_operation(V, mol_size, operation=torch.sum)
        
        def avg_pool(V: torch.Tensor, mol_size: torch.Tensor) -> torch.Tensor:
            sum_v = self._segment_operation(V, mol_size, operation=torch.sum)
            return sum_v / mol_size.unsqueeze(1).float()
        
        def max_pool(V: torch.Tensor, mol_size: torch.Tensor) -> torch.Tensor:
            return self._segment_operation(V, mol_size, operation=torch.max)

        pool_map = {
            'sum': sum_pool,
            'avg': avg_pool,
            'max': max_pool
        }
        return pool_map[pooling]

    def _segment_operation(self, V: torch.Tensor, mol_size: torch.Tensor, operation: callable) -> torch.Tensor:
        """同KA_GNN_two的分段聚合实现"""
        batch_size = V.shape[0]
        results = []
        for i in range(batch_size):
            actual_nodes = V[i, :mol_size[i], :]
            if operation == torch.max:
                pooled = operation(actual_nodes, dim=0)[0]
            else:
                pooled = operation(actual_nodes, dim=0)
            results.append(pooled)
        return torch.stack(results, dim=0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        :param batch: 包含'V', 'A', 'mol_size'的字典
        :return: 图级输出 [batch_size, out]
        """
        V = batch['V']
        A = batch['A']
        mol_size = batch['mol_size']

        h = self.kan_line(V)

        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                h = layer(h, A)  # 图卷积传播
            else:
                h = self.leaky_relu(layer(h, A))  # 最后一层额外激活

        # 图级聚合
        y = self.pool_func(h, mol_size)

        # 输出层
        out = self.Readout(y)
        return out

    def get_grad_norm_weights(self) -> nn.ParameterList:
        return self.parameters()