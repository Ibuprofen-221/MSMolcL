import torch
from torch import nn
import torch.nn.functional as F
from config import CFG
import utils
from GNN import layers as gly
from FragmentationTreeEncoder import GraphTransformerEncoder
import math
from GNN.layers import MultiHeadMSAttentionPool

class FCModule(nn.Module):
    def __init__(self, embed_dim=256, norm_type='layernorm'):
        super(FCModule, self).__init__()
        self.active_relu = nn.ReLU()
        self.emb = nn.Embedding(100, embed_dim)
        self.ln1 = nn.Linear(embed_dim, 4*embed_dim)
        self.ln2 = nn.Linear(4*embed_dim,embed_dim)

        self.norm_type = norm_type

        if norm_type == 'layernorm':
            self.nm1 = nn.LayerNorm(4*embed_dim)
            self.nm2 = nn.LayerNorm(embed_dim)
        elif norm_type == 'batchnorm':
            self.bn1 = nn.BatchNorm1d(4*embed_dim)
            self.bn2 = nn.BatchNorm1d(embed_dim)
        else:
            raise ValueError(f"FCModule: Unknown norm_type: {norm_type}")

        nn.init.xavier_uniform_(self.ln1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.ln2.weight, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, x):
        x_emb = self.emb(x)
        if self.norm_type == 'layernorm':
            h1 = self.nm1(self.active_relu(self.ln1(x_emb)))
            out = self.nm2(self.active_relu(x_emb + self.ln2(h1)))
        elif self.norm_type == 'batchnorm':
            h1 = self.bn1(self.active_relu(self.ln1(x_emb)))
            out = self.bn2(self.active_relu(x_emb + self.ln2(h1)))
        return out

class sinEmbedModule(nn.Module):
    def __init__(
        self,
        embedding_dim,
        dropout,
        dropout_rate,
        dropout_in_first_layer,
    ):
        super(sinEmbedModule, self).__init__()
        self.layers = nn.ModuleList()
        self.lambda_min = 10**-2.5
        self.lambda_max = 10**3.3
        self.x = torch.arange(0, embedding_dim, 2).to(torch.float64)
        self.x = (
            2
            * math.pi
            * (
                self.lambda_min
                * (self.lambda_max / self.lambda_min) ** (self.x / (embedding_dim - 2))
            )
            ** -1
        )
        dropout_starting_layer = 0 if dropout_in_first_layer else 1
        for i in range(2):
            self.layers.append(nn.Linear(embedding_dim, embedding_dim))
            if i==0:
                self.layers.append(nn.ReLU())
            self.layers.append(nn.LayerNorm(embedding_dim))
            if dropout and i >= dropout_starting_layer:
                self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, mz):
        self.x = self.x.to(mz.device)
        x = torch.einsum("bl,d->bld", mz, self.x)
        sinemb = torch.sin(x)
        cosemb = torch.cos(x)
        b, l, d = sinemb.shape
        x = torch.empty((b, l, 2 * d), dtype=mz.dtype, device=mz.device)
        x[:, :, ::2] = sinemb
        x[:, :, 1::2] = cosemb
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = layer(x)
        return x

class EmbedModule(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        dropout,
        dropout_rate,
        dropout_in_first_layer,
    ):
        super(EmbedModule, self).__init__()
        self.layers = nn.ModuleList()
        self.tkembd = nn.Embedding(20000, embedding_dim)
        self.sinemb = sinEmbedModule(
            embedding_dim, dropout, dropout_rate, dropout_in_first_layer
        )
        dropout_starting_layer = 0 if dropout_in_first_layer else 1
        for i in range(2):
            if i == 0:
                self.layers.append(nn.Linear(embedding_dim + 1, embedding_dim))
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Linear(embedding_dim, embedding_dim))
            self.layers.append(nn.LayerNorm(embedding_dim))
            if dropout and i >= dropout_starting_layer:
                self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, mz, intensity, precursor):
        mz = torch.cat([mz, precursor.unsqueeze(1)], dim=1)
        mzemb = self.sinemb(mz)
        intensity = torch.cat(
            [intensity, 2 * torch.ones((mzemb.shape[0], 1)).to(mz.device)], -1
        )
        x = torch.cat([mzemb, intensity.unsqueeze(-1)], dim=2)
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = layer(x)
        return x

class SinSiameseModel(nn.Module):
    # 新增 norm_type 参数
    def __init__(
        self,
        cfg,
        input_dim=100,
        embed_dim: int = 1024,
        project_size = 400,
        use_film_tokens: bool = False,
        norm_type: str = 'layernorm'
    ):
        super(SinSiameseModel, self).__init__()
        self.embed_dim = embed_dim

        # 读取排列策略：默认旧行为（intensity_desc）；配置为 topk_then_mz_asc 时开启运行时重排
        msenc_cfg = getattr(getattr(cfg, 'model', {}), 'ms_encoder', {})
        self._runtime_sort_to_mz = str(getattr(msenc_cfg, 'spectrum_order', 'intensity_desc')) == 'topk_then_mz_asc'

        # 统一 dropout 设置
        dr = float(getattr(msenc_cfg, 'dropout', 0.1) or 0.0)
        use_dropout = dr > 0.0
        self.embd_model = EmbedModule(
            input_dim, embed_dim,
            dropout=use_dropout,
            dropout_rate=dr,
            dropout_in_first_layer=False,
        )

        self.tranencoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embed_dim, 8, dropout=dr, batch_first=True, activation=F.gelu
            ),
            6,
        )

        self.trandecoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                embed_dim, 8, dropout=dr, batch_first=True, activation=F.gelu
            ),
            6,
        )

        self.fc_adduct = FCModule(embed_dim=embed_dim, norm_type=norm_type)
        self.use_film_tokens = use_film_tokens
        if self.use_film_tokens:
            self.film_from_cond = nn.Linear(embed_dim, 2 * embed_dim)
            nn.init.zeros_(self.film_from_cond.weight)
            nn.init.zeros_(self.film_from_cond.bias)
        self.project = nn.Linear(embed_dim, project_size)
        self.active = nn.SELU()

    @staticmethod
    def _sort_by_mz_asc(mz: torch.Tensor, intensity: torch.Tensor):
        """
        对每个样本将非零 m/z 的列按升序排序；padding(=0)保持在末尾。
        mz/intensity 形状: [B, L]
        """
        mask = mz > 0
        inf = torch.full_like(mz, float('inf'))
        mz_filled = torch.where(mask, mz, inf)
        sorted_mz, idx = torch.sort(mz_filled, dim=1, descending=False)
        # 恢复 padding 的 m/z 为 0
        sorted_mz = torch.where(torch.isinf(sorted_mz), torch.zeros_like(sorted_mz), sorted_mz)
        # 强制 intensity 按相同索引重排；padding 位置置零
        sorted_int = torch.gather(intensity, 1, idx)
        sorted_int = torch.where(sorted_mz > 0, sorted_int, torch.zeros_like(sorted_int))
        return sorted_mz, sorted_int

    def _get_padding_mask(self, mz, cp):
        B, L = mz.shape
        idx = torch.arange(L + 1, device=mz.device).unsqueeze(0).expand(B, -1)
        valid = idx < cp.unsqueeze(1)
        valid[:, -1] = True
        pad_mask = ~valid
        return pad_mask

    def inference(self, mz, intensity, precursor, cp, adduct):
        emb = self.embd_model(mz, intensity, precursor) # [B, L+1, D]
        pad_mask = self._get_padding_mask(mz, cp) # [B, L+1]

        # Encoder
        enc_out = self.tranencoder(emb, src_key_padding_mask=pad_mask)

        # Decoder
        param_emb = torch.stack([self.fc_adduct(adduct)], dim=1) # [B, 1, D]
        dec_out = self.trandecoder(param_emb, enc_out, memory_key_padding_mask=pad_mask) # [B, 1, D]

        # 池化和投影
        out = dec_out.squeeze(1) # [B, D]
        return self.active(self.project(out))

    def forward(self, spec, precursor, adduct):
        # .to(precursor.device) 确保张量在正确的设备上
        mz = spec[:, 0, :].to(precursor.device)
        intensity = spec[:, 1, :].to(precursor.device)
        # 运行时按 m/z 升序重排（仅对有效峰；padding 留尾部）
        if self._runtime_sort_to_mz:
            mz, intensity = self._sort_by_mz_asc(mz, intensity)
        cp = (mz > 0).sum(1)
        return self.inference(mz, intensity, precursor, cp, adduct)
    
    def encode_tokens(self, spec, precursor, adduct):
        mz = spec[:, 0, :].to(precursor.device)
        intensity = spec[:, 1, :].to(precursor.device)
        if self._runtime_sort_to_mz:
            mz, intensity = self._sort_by_mz_asc(mz, intensity)
        cp = (mz > 0).sum(1)

        emb = self.embd_model(mz, intensity, precursor)
        if self.use_film_tokens:
            cond_vec = self.fc_adduct(adduct)
            gamma, beta = self.film_from_cond(cond_vec).chunk(2, dim=-1)
            emb = emb * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        pad_mask = self._get_padding_mask(mz, cp)
        enc_out = self.tranencoder(emb, src_key_padding_mask=pad_mask)

        mz_with_precursor = torch.cat([mz, precursor.unsqueeze(1)], dim=1)

        return enc_out, pad_mask, mz_with_precursor


class MolGNNEncoder(nn.Module):
    def __init__(self, outdim, n_feats=74, n_filters_list=[256,256,256], n_head=4, mols=1, adj_chans=6, readout_layers=2, bias=True,
                 raw_in_feats=None):
        super().__init__()
        # raw_in_feats = 训练/推理时 V 的“实际输入维”；n_feats = 工作维(投影后的维)
        self._work_feats = n_feats
        self._expected_in_feats = raw_in_feats if (raw_in_feats is not None) else n_feats

        if raw_in_feats is not None and raw_in_feats != n_feats:
            self.input_proj = nn.Linear(raw_in_feats, n_feats, bias=True)
        else:
            self.input_proj = nn.Identity()

        n_filters_list = [i for i in n_filters_list if i is not None]
        lys = []
        if n_filters_list:
            nf1 = n_feats
            for i, nf in enumerate(n_filters_list):
                ly = gly.GConvBlockNoGF(nf1, nf, mols, adj_chans, bias)
                lys.append(ly)
                nf1 = nf
            self.block_layers = nn.ModuleList(lys)
            self.attention_layer = gly.MultiHeadGlobalAttention(nf, n_head=n_head, concat=True, bias=bias)
            readout_in_dim = nf * n_head
        else:
            self.block_layers = nn.ModuleList()
            self.attention_layer = None
            readout_in_dim = n_feats

        readout_layers_list = [nn.Linear(readout_in_dim, outdim, bias=bias)] + [nn.Linear(outdim, outdim) for _ in range(readout_layers - 1)]
        self.readout_layers = nn.ModuleList(readout_layers_list)
        self.gelu = nn.GELU()

    def forward(self, batch):
        V = batch['V']        # [B, N, C_in]
        A = batch['A']
        mol_size = batch['mol_size']

        if not isinstance(self.input_proj, nn.Identity):
            V = self.input_proj(V)

        for ly in self.block_layers:
            V = ly(V, A)

        X = self.attention_layer(V, mol_size) if self.attention_layer else V.sum(dim=1)
        for ly in self.readout_layers:
            X = self.gelu(ly(X))
        return X


class ProjectionHead(nn.Module):
    def __init__(self,
                 embedding_dim,
                 projection_dim,
                 dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        return self.net(x)

# 新增：InfoNCE损失函数计算
class InfoNCELoss(nn.Module):
    """
    同模态InfoNCE损失（分子-分子 或 质谱-质谱）
    需要排除自身，避免平凡解
    """
    def __init__(self, temperature=0.1, handling_mode='skip'):
        super().__init__()
        self.temperature = temperature
        self.handling_mode = handling_mode  # 'skip', 'soft', 'ignore'
        
    def forward(self, embeddings, labels):
        """
        embeddings: [B, D] - 同模态嵌入向量
        labels: [B] - 样本标签，相同标签表示相似样本
        """
        B = embeddings.size(0)
        
        # 归一化
        embeddings = F.normalize(embeddings, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # 创建正样本掩码（相同标签且非自身）
        pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        self_mask = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        pos_mask = pos_mask & ~self_mask
        
        # 检查每个样本是否有正样本
        has_positive = pos_mask.sum(dim=1) > 0
        
        if not has_positive.all():
            # 处理没有正样本的情况
            if self.handling_mode == 'skip':
                return self._forward_skip(sim_matrix, pos_mask, has_positive)
            elif self.handling_mode == 'soft':
                return self._forward_soft(sim_matrix, pos_mask, labels, has_positive)
            elif self.handling_mode == 'ignore':
                return self._forward_ignore(sim_matrix, pos_mask, has_positive)
        
        # 所有样本都有正样本的正常情况
        return self._compute_loss(sim_matrix, pos_mask, self_mask)
    
    def _compute_loss(self, sim_matrix, pos_mask, self_mask):
        """计算损失的核心逻辑"""
        # 排除自身后计算logsumexp
        logsumexp_pos = torch.logsumexp(
            sim_matrix.masked_fill(~pos_mask, float('-inf')), 
            dim=1
        )
        logsumexp_all = torch.logsumexp(
            sim_matrix.masked_fill(self_mask, float('-inf')), 
            dim=1
        )
        loss = -(logsumexp_pos - logsumexp_all).mean()
        return loss
    
    def _forward_skip(self, sim_matrix, pos_mask, has_positive):
        """跳过没有正样本的样本"""
        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=sim_matrix.device, requires_grad=True)
        
        # 只保留有正样本的样本
        mask = has_positive
        sim_matrix_sub = sim_matrix[mask][:, mask]
        pos_mask_sub = pos_mask[mask][:, mask]
        self_mask_sub = torch.eye(mask.sum(), device=sim_matrix.device, dtype=torch.bool)
        
        return self._compute_loss(sim_matrix_sub, pos_mask_sub, self_mask_sub)
    
    def _forward_soft(self, sim_matrix, pos_mask, labels, has_positive):
        """软处理：没有正样本时使用所有其他样本作为"软"负样本"""
        B = sim_matrix.size(0)
        
        # 对于没有正样本的样本，使用所有其他样本作为负样本
        soft_pos_mask = pos_mask.clone()
        no_pos_mask = ~has_positive
        
        if no_pos_mask.any():
            # 对于没有正样本的样本，将除自身外的所有样本都视为负样本
            # 这样他们的损失会鼓励他们与所有样本都不相似
            soft_pos_mask[no_pos_mask] = False  # 没有正样本，只有负样本
        
        self_mask = torch.eye(B, device=sim_matrix.device, dtype=torch.bool)
        return self._compute_loss(sim_matrix, soft_pos_mask, self_mask)
    
    def _forward_ignore(self, sim_matrix, pos_mask, has_positive):
        """忽略问题，直接计算（可能产生NaN）"""
        B = sim_matrix.size(0)
        self_mask = torch.eye(B, device=sim_matrix.device, dtype=torch.bool)
        
        # 直接计算，但处理可能的NaN
        loss = self._compute_loss(sim_matrix, pos_mask, self_mask)
        if torch.isnan(loss):
            return torch.tensor(0.0, device=sim_matrix.device, requires_grad=True)
        return loss

class FragSimiModel(nn.Module):
    def __init__(self, cfg, enable_compile: bool = True):
        super().__init__()
        self.cfg = cfg

        # --- 新增: 同模态损失配置 ---
        # 从配置中获取是否使用同模态损失
        self.use_mol_loss = getattr(cfg.training, 'same_modal_loss', 'none').lower() in ['mol', 'both']
        self.use_ms_loss = getattr(cfg.training, 'same_modal_loss', 'none').lower() in ['ms', 'both']
        
        # 初始化同模态InfoNCE损失函数
        self.mol_infonce_loss = InfoNCELoss(temperature=getattr(cfg.training, 'mol_temperature', 0.1)) if self.use_mol_loss else None
        self.ms_infonce_loss = InfoNCELoss(temperature=getattr(cfg.training, 'ms_temperature', 0.1)) if self.use_ms_loss else None
        
        # 同模态损失权重
        self.mol_loss_weight = getattr(cfg.training, 'mol_loss_weight', 1.0)
        self.ms_loss_weight = getattr(cfg.training, 'ms_loss_weight', 1.0)
        print(self.mol_loss_weight )
        print(self.ms_loss_weight)
        # --- 新增: torch.compile 配置 ---
        # 检查PyTorch版本是否支持compile，并设置编译模式
        # 'reduce-overhead' 模式编译时间较短，适合动态性较强的输入
        # 'max-autotune' 模式会花更多时间寻找最优kernel，适合固定尺寸的输入
        use_torch_compile = hasattr(torch, 'compile') and bool(enable_compile)
        if use_torch_compile:
            print("PyTorch 2.x compile feature detected. Applying to compatible modules.")

        def _fusion_from_cfg(c):
            # unified field
            fusion = getattr(c.model.ms_encoder, "fusion", None)
            if fusion is not None:
                return fusion
            # backward compat: fall back to old 'frag_tree.mode'
            ft_mode = getattr(getattr(c.model.ms_encoder, "frag_tree", {}), "mode", None)
            if ft_mode == "disable":
                return "clerms-only"
            # default safer choice when old config had trees enabled but no explicit fusion
            return "concat"

        self.fusion_mode = _fusion_from_cfg(cfg)
        cond_str = str(getattr(getattr(cfg.model, 'ms_encoder', {}), 'adduct_condition', '') or '').lower()
        self.use_film_tokens = ('film' in cond_str) or ('film_tokens' in cond_str)

        # -------- 分子编码器部分 --------
        self.mol_gnn_encoder = None
        mol_embedding_dim = 0
        if 'gnn' in self.cfg.model.mol_encoder.type:
            gnn_cfg = self.cfg.model.mol_encoder.gnn
            work_feats = int(getattr(gnn_cfg, 'in_feats', 74))
            use_sub = bool(getattr(gnn_cfg, 'use_subgraph_fp', False))
            sub_bits = int(getattr(gnn_cfg, 'subgraph_fp_nbits', 256))
            raw_in_feats = work_feats + (sub_bits if use_sub else 0)

            self.mol_gnn_encoder = MolGNNEncoder(
                outdim=self.cfg.model.mol_encoder.embedding_dim,
                n_feats=work_feats,
                raw_in_feats=raw_in_feats,
                n_filters_list=self.cfg.model.mol_encoder.gnn.n_filters_list,
                n_head=self.cfg.model.mol_encoder.gnn.n_head,
                readout_layers=self.cfg.model.mol_encoder.gnn.readout_layers
            )
            mol_embedding_dim += self.cfg.model.mol_encoder.embedding_dim

        if 'fp' in self.cfg.model.mol_encoder.type:
            mol_embedding_dim += self.cfg.model.mol_encoder.fp.nbits

        # -------- 质谱编码器部分 --------
        msenc_cfg = getattr(self.cfg.model, 'ms_encoder', {})
        xattn_cfg = getattr(msenc_cfg, 'xattn', {}) or {}
        tree_cfg   = getattr(msenc_cfg, 'tree_encoder', {}) or {}
        self.norm_type = getattr(msenc_cfg, 'norm_type', 'layernorm')

        spectrum_dim = int(getattr(msenc_cfg, 'spectrum_dim', 100))
        xattn_hidden = int(xattn_cfg.get('hidden', 512))

        # 原有：深度
        tree_depth    = int(tree_cfg.get('depth', 2))
        # 新增：隐藏宽度 / 头数 / 池化输出维
        tree_hidden   = int(tree_cfg.get('hidden_dim', 256))
        tree_heads1   = int(tree_cfg.get('heads1', 8))
        tree_heads2   = int(tree_cfg.get('heads2', 1))
        tree_pool_out = int(tree_cfg.get('pool_out_dim', 256))


        # 实例化 CLERMS_encoder
        self.CLERMS_encoder = SinSiameseModel(cfg, input_dim=spectrum_dim, use_film_tokens=self.use_film_tokens, 
                                              norm_type=self.norm_type)
        # 应用 torch.compile
        if use_torch_compile:
            self.CLERMS_encoder = torch.compile(self.CLERMS_encoder, mode="max-autotune")


        enhanced_tree = bool(tree_cfg.get('enhanced_features', False))
        node_dim = 18 if enhanced_tree else 15
        edge_dim = 14 if enhanced_tree else 13

        ms_feature_dim = 0

        if self.fusion_mode == "xattn":
            self.trees_encoder = GraphTransformerEncoder(
                input_dim=node_dim,
                edge_dim=edge_dim,
                total_depth=tree_depth,
                hidden_dim=tree_hidden,
                heads1=tree_heads1,
                heads2=tree_heads2,
                pool_out_dim=tree_pool_out,
                norm_type=self.norm_type
            )
            
            xattn_cfg = getattr(msenc_cfg, 'xattn', {}) or {}
            pre_self = bool(xattn_cfg.get('pre_self_attn', False))

            self.fusion = TreeMSCrossAttentionFusion(
                spec_dim=1024,
                tree_dim=tree_hidden,
                hidden=xattn_hidden,
                heads=8,
                pool_heads=4,
                dropout=self.cfg.model.ms_encoder.dropout,
                pre_self_attn=pre_self
            )
            if use_torch_compile:
                self.fusion = torch.compile(self.fusion, mode="reduce-overhead")

            ms_feature_dim += self.fusion.out_dim

                    
        elif self.fusion_mode == "concat":
            self.trees_encoder = GraphTransformerEncoder(
                input_dim=node_dim,
                edge_dim=edge_dim,
                total_depth=tree_depth,
                hidden_dim=tree_hidden,
                heads1=tree_heads1,
                heads2=tree_heads2,
                pool_out_dim=tree_pool_out,
                norm_type=self.norm_type,
            )
            ms_feature_dim += 400
            ms_feature_dim += tree_pool_out
        elif self.fusion_mode in ("tree-only", "tree_only"):
            self.trees_encoder = GraphTransformerEncoder(
                input_dim=node_dim,
                edge_dim=edge_dim,
                total_depth=tree_depth,
                hidden_dim=tree_hidden,
                heads1=tree_heads1,
                heads2=tree_heads2,
                pool_out_dim=tree_pool_out,
                norm_type=self.norm_type
            )
            ms_feature_dim += tree_pool_out  # 只用树的池化向量
        elif self.fusion_mode in ("clerms-only", "clerms_only"):
            ms_feature_dim += 400
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        self.ms_projection = ProjectionHead(
            embedding_dim=ms_feature_dim,
            projection_dim=self.cfg.model.projection_dim,
            dropout=self.cfg.model.ms_encoder.dropout
        )

        self.mol_projection = ProjectionHead(
            embedding_dim=mol_embedding_dim,
            projection_dim=self.cfg.model.projection_dim,
            dropout=self.cfg.model.ms_encoder.dropout
        )
        
        # 可学习温度（默认关闭）
        ls = getattr(cfg.training, 'learnable_sigma', False)
        self._learn_sigma = bool(ls if isinstance(ls, bool) else getattr(ls, 'enabled', False))
        if self._learn_sigma:
            if isinstance(ls, dict):
                min_s = float(ls.get('min', 0.05))
                max_s = float(ls.get('max', 0.2))
                init_s = float(ls.get('init', getattr(cfg.training, 'sigma', 0.1)))
            else:
                min_s, max_s, init_s = 0.05, 0.2, float(getattr(cfg.training, 'sigma', 0.1))
            self._sigma_min = min_s
            self._sigma_max = max_s
            init_log_inv = math.log(max(1e-6, 1.0 / max(1e-6, init_s)))
            self.log_inv_sigma = nn.Parameter(torch.tensor(init_log_inv, dtype=torch.float32))
        else:
            self.log_inv_sigma = None
        

    def forward(self, batch):
        device = next(self.parameters()).device

        # 获取分子嵌入
        mol_feat_list = []
        if 'gnn' in self.cfg.model.mol_encoder.type:
            mol_feat_list.append(self.mol_gnn_encoder(batch))
        if 'fp' in self.cfg.model.mol_encoder.type:
            mol_fps = batch["mol_fps"].to(device)
            mol_feat_list.append(mol_fps)
        mol_features = torch.cat(mol_feat_list, dim=1)
        mol_embeddings_proj = self.mol_projection(mol_features)

        # 获取质谱嵌入
        spec_tensor = batch['spec_tensor'].squeeze(1).to(device)
        adduct_indices = batch['adduct_type_idx'].to(device)
        precursor = batch['precursor_mz'].to(device)

        # 质谱分支：仅根据 fusion_mode 决定如何取特征
        if self.trees_encoder is not None and self.fusion_mode == "xattn":
            spec_seq, spec_pad_mask, _ = self.CLERMS_encoder.encode_tokens(spec_tensor, precursor, adduct_indices)
            pyg_data = batch['pyg_data'].to(device)
            tree_seq, tree_pad_mask = self.trees_encoder(pyg_data, return_nodes=True)
            ms_features = self.fusion(
                spec_seq, spec_pad_mask,
                tree_seq, tree_pad_mask
            )
        elif self.trees_encoder is not None and self.fusion_mode == "concat":
            spectrum_enc = self.CLERMS_encoder(spec_tensor, precursor, adduct_indices)
            pyg_data = batch['pyg_data'].to(device)
            tree_enc = self.trees_encoder(pyg_data)
            ms_features = torch.cat([spectrum_enc, tree_enc], dim=-1)
        elif self.trees_encoder is not None and self.fusion_mode in ("tree-only","tree_only"):
            pyg_data = batch['pyg_data'].to(device)
            ms_features = self.trees_encoder(pyg_data)
        else:
            ms_features = self.CLERMS_encoder(spec_tensor, precursor, adduct_indices)

        ms_embeddings_proj = self.ms_projection(ms_features)

        ms_embeddings = F.normalize(ms_embeddings_proj, dim=1)
        mol_embeddings = F.normalize(mol_embeddings_proj, dim=1)

        ids = batch['compound_id'].to(device)

        # 损失计算
        if self._learn_sigma and (self.log_inv_sigma is not None):
            sigma_unclamped = torch.exp(-self.log_inv_sigma)        # = sigma
            sigma = torch.clamp(sigma_unclamped, self._sigma_min, self._sigma_max)
        else:
            sigma = float(getattr(self.cfg.training, 'sigma', 0.1))

        sim_matrix = mol_embeddings @ ms_embeddings.t() / sigma

        pos_mask = ids.unsqueeze(0).eq(ids.unsqueeze(1))

        logsumexp_pos_row = torch.logsumexp(sim_matrix.masked_fill(~pos_mask, float('-inf')), dim=1)
        logsumexp_all_row = torch.logsumexp(sim_matrix, dim=1)
        loss_row = -(logsumexp_pos_row - logsumexp_all_row).mean()

        logsumexp_pos_col = torch.logsumexp(sim_matrix.t().masked_fill(~pos_mask.t(), float('-inf')), dim=1)
        logsumexp_all_col = torch.logsumexp(sim_matrix.t(), dim=1)
        loss_col = -(logsumexp_pos_col - logsumexp_all_col).mean()
        loss_con = (loss_row + loss_col) / 2

        # --- 新增: 同模态InfoNCE损失 ---
        loss_mol = 0.0
        loss_ms = 0.0
        
        # 分子-分子同模态损失
        if self.use_mol_loss and self.mol_infonce_loss is not None:
            loss_mol = self.mol_infonce_loss(mol_embeddings, ids) * self.mol_loss_weight
        
        # 质谱-质谱同模态损失
        if self.use_ms_loss and self.ms_infonce_loss is not None:
            loss_ms = self.ms_infonce_loss(ms_embeddings, ids) * self.ms_loss_weight

        # --- 统一由配置控制是否叠加 MSE ---
        # 优先读 training.loss；缺省则保持旧行为：用 alpha 控制（alpha<=0 即不加 MSE）
        loss_setting = getattr(self.cfg.training, 'loss', None)
        if loss_setting is None:
            # 旧逻辑：仅用 alpha 控制
            mse_weight = float(getattr(self.cfg.training, 'alpha', 100.0))
        else:
            loss_lc = str(loss_setting).lower()
            if loss_lc in ('infonce', 'info_nce', 'pure_infonce'):
                mse_weight = 0.0
            elif loss_lc in ('infonce+mse', 'infonce_mse', 'joint', 'combined'):
                mse_weight = float(getattr(self.cfg.training, 'alpha', 100.0))
            else:
                mse_weight = 0.0

        total_loss = loss_con + loss_mol + loss_ms  # 添加同模态损失
        if mse_weight > 0 and ('fp' in self.cfg.model.mol_encoder.type):
            # 按需计算 MSE（只有在需要、且模型包含 fp 时才计算）
            # 1) 指纹 Jaccard
            intersection = torch.matmul(mol_fps, mol_fps.t())
            sum_bits = mol_fps.sum(dim=1, keepdim=True) + mol_fps.sum(dim=1).unsqueeze(0)
            union = sum_bits - intersection
            jaccard_sim = intersection / (union + 1e-8)
            # 2) 跨模态余弦
            cross_modal_sim = torch.matmul(mol_embeddings, ms_embeddings.t())
            # 3) MSE
            mse_loss = F.mse_loss(cross_modal_sim, jaccard_sim)
            total_loss = total_loss + mse_weight * mse_loss

        # 返回所有损失分量，便于监控
        return {
            'total_loss': total_loss,
            'cross_loss': loss_con,
            'mol_loss': loss_mol if self.use_mol_loss else torch.tensor(0.0, device=device),
            'ms_loss': loss_ms if self.use_ms_loss else torch.tensor(0.0, device=device),
            'mse_loss': mse_loss if (mse_weight > 0 and 'fp' in self.cfg.model.mol_encoder.type) else torch.tensor(0.0, device=device)
        }

    def encode_mol(self, batch):
        device = next(self.parameters()).device
        mol_feat_list = []
        if 'gnn' in self.cfg.model.mol_encoder.type:
            gnn_batch = {k: v.to(device) for k, v in batch.items() if k in ['V', 'A', 'mol_size']}
            mol_feat_list.append(self.mol_gnn_encoder(gnn_batch))
        if 'fp' in self.cfg.model.mol_encoder.type:
            mol_feat_list.append(batch["mol_fps"].to(device))
        mol_features = torch.cat(mol_feat_list, dim=1)
        mol_embeddings_proj = self.mol_projection(mol_features)
        return mol_embeddings_proj

    def encode_ms(self, batch):
        device = next(self.parameters()).device
        spec_tensor = batch['spec_tensor'].squeeze(1).to(device)
        adduct_indices = batch['adduct_type_idx'].to(device)
        precursor = batch['precursor_mz'].to(device)

        if self.trees_encoder is not None and self.fusion_mode == "xattn":
            spec_seq, spec_pad_mask, _ = self.CLERMS_encoder.encode_tokens(spec_tensor, precursor, adduct_indices)
            pyg_data = batch['pyg_data'].to(device)
            tree_seq, tree_pad_mask = self.trees_encoder(pyg_data, return_nodes=True)
            ms_features = self.fusion(
                spec_seq, spec_pad_mask,
                tree_seq, tree_pad_mask
            )
        elif self.trees_encoder is not None and self.fusion_mode == "concat":
            spectrum_encodings = self.CLERMS_encoder(spec_tensor, precursor, adduct_indices)
            pyg_data = batch['pyg_data'].to(device)
            trees_encodings = self.trees_encoder(pyg_data)
            ms_features = torch.cat([spectrum_encodings, trees_encodings], dim=-1)
        elif self.trees_encoder is not None and self.fusion_mode in ("tree-only", "tree_only"):
            pyg_data = batch['pyg_data'].to(device)
            ms_features = self.trees_encoder(pyg_data)
        else:
            ms_features = self.CLERMS_encoder(spec_tensor, precursor, adduct_indices)

        ms_embeddings_proj = self.ms_projection(ms_features)
        return ms_embeddings_proj

    
    
    # 以下几个函数是新增的，预测的接口，给定谱图和分子候选池来进行检索
    
    @torch.no_grad()
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1)

    @torch.no_grad()
    def _auto_chunk_size(self, D: int, num_candidates_hint: int = 4096, safety: float = 0.6, hard_cap: int = 8192) -> int:
        """
        依据可用显存估计一个候选分块大小（仅用于相似度计算时 q @ cand^T）。
        """
        if torch.cuda.is_available():
            try:
                free, _total = torch.cuda.mem_get_info()
                bytes_per_cand = D * 4
                est = int(safety * free // bytes_per_cand)
                return max(1, min(est, hard_cap))
            except Exception:
                pass
        return 2048

    # 检索 (predict 分子结构) 的函数
    @torch.no_grad()
    def predict(
        self,
        query_batch=None,                                   # 允许 None（当传 preencoded_queries 时）
        *,
        candidate_loader=None,
        candidate_embeddings=None,
        candidate_indices=None,
        mode: str = "scores",                               # "scores" | "topk" | "best" | "rank"
        topk: int | None = None,
        device: torch.device | None = None,
        cand_chunk_size: int = 0,
        labels=None,                                        # 1D tensor/list，长度=B；每个query的真值在“全语料库”中的全局索引
        per_query_candidates=None,                          # 2D tensor [B, K]；每行候选索引(第0列=真值)
        preencoded_queries=None,                            # 可选：直接传入已预编码的 query 向量 [B, D]
        deterministic: bool = True,                         # 新增：默认开启确定性执行
    ):
        """
        通用预测接口：
        - mode="scores"：返回每个查询对所有候选的相似度（B×N）
        - mode="best"  ：返回每个查询的Top1（索引+分数）
        - mode="topk"  ：返回每个查询的TopK（索引+分数，降序）
        - mode="rank"  ：返回每个查询的排名（1=最好），可配合：
            * per_query_candidates=[B,K]（第0列为真值）——用于“1真+若干负例”的排名
            * 或 labels=[B] + 全语料库 candidate_embeddings ——用于“全语料库”排名（分块显存友好）
        """
        assert mode in ("scores", "topk", "best", "rank")
        if mode == "best":
            topk = 1
        if mode == "topk":
            if topk is None or topk <= 0:
                topk = 10

        if device is None:
            device = next(self.parameters()).device

        # --------- 确定性开关（默认开启，可关闭）---------
        # 进入时保存旧状态；离开时恢复，避免污染全局
        _old = {
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None,
            "tf32_cudnn":  torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None,
            "cudnn_bench": torch.backends.cudnn.benchmark if torch.cuda.is_available() else None,
            "deterministic": torch.are_deterministic_algorithms_enabled()
        }
        try:
            if deterministic:
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True)
            # ----------------------------------------------

            self.eval()

            # --------- 处理 Query 向量 ---------
            if preencoded_queries is not None:
                # 直接使用预编码向量（例如 encode_ms 的输出）
                q_emb = preencoded_queries.to(device)
            else:
                # 正常路径：从 batch 编码
                assert query_batch is not None, "需要 query_batch 或 preencoded_queries 之一"
                q_emb = self.encode_ms({k: (v.to(device) if hasattr(v, "to") else v) for k, v in query_batch.items()})

            q_emb = self._normalize(q_emb).contiguous()
            B, D = q_emb.shape

            # --------- 收集/构建候选块（显存友好）---------
            cand_blocks = []
            cand_block_indices = []
            N_total = 0

            def _append_block(c_emb, idx):
                nonlocal N_total
                if c_emb is None or c_emb.numel() == 0:
                    return
                # 统一归一化到CPU缓存，计算时再搬到GPU
                t = self._normalize(c_emb).cpu()
                cand_blocks.append(t)
                if idx is None:
                    idx = torch.arange(N_total, N_total + t.size(0), dtype=torch.long)
                cand_block_indices.append(idx.cpu())
                N_total += t.size(0)

            if candidate_embeddings is not None:
                c = candidate_embeddings
                if hasattr(c, "device") and c.device.type != "cpu":
                    c = c.cpu()
                _append_block(c, candidate_indices if candidate_indices is not None else None)
            else:
                if candidate_loader is not None:
                    for batch in candidate_loader:
                        if not batch:
                            continue
                        if "row_idx" in batch:
                            block_idx = batch["row_idx"]
                            del batch["row_idx"]
                        else:
                            block_idx = None
                        for k, v in batch.items():
                            if hasattr(v, "to"):
                                batch[k] = v.to(device)
                        c_emb = self.encode_mol(batch).cpu()
                        _append_block(c_emb, block_idx.cpu() if block_idx is not None else None)

            # --------- 提前处理块大小 ---------
            if cand_chunk_size and cand_chunk_size > 0:
                chunk_c = int(cand_chunk_size)
            else:
                chunk_c = self._auto_chunk_size(D, num_candidates_hint=4096, safety=0.6, hard_cap=8192)

            # ====================== 各模式分支 ======================
            # 1) 直接返回完整分数矩阵 B×N （拼接所有候选分块）
            if mode == "scores":
                if N_total == 0:
                    return torch.empty((B, 0), dtype=torch.float32)
                cols = []
                for c_cpu in cand_blocks:
                    for j0 in range(0, c_cpu.size(0), chunk_c):
                        j1 = min(j0 + chunk_c, c_cpu.size(0))
                        s = (q_emb @ c_cpu[j0:j1].to(device, non_blocking=True).t()).cpu()
                        cols.append(s)
                return torch.cat(cols, dim=1)

            # 2) 在线维护 TopK / Top1（合并分块后再 topk）
            if mode in ("topk", "best"):
                K = int(topk)
                topk_scores = torch.full((B, 0), float("-inf"))
                topk_indices = torch.empty((B, 0), dtype=torch.long)
                for c_cpu, idx_cpu in zip(cand_blocks, cand_block_indices):
                    Cc = c_cpu.size(0)
                    s_cols = []
                    for j0 in range(0, Cc, chunk_c):
                        j1 = min(j0 + chunk_c, Cc)
                        s_cols.append(q_emb @ c_cpu[j0:j1].to(device, non_blocking=True).t())
                    s = torch.cat(s_cols, dim=1)  # [B, Cc]

                    merged_scores = torch.cat([topk_scores.to(device), s], dim=1)
                    left_idx = topk_indices.to(device) if topk_indices.numel() else torch.empty((B, 0), dtype=torch.long, device=device)
                    right_idx = idx_cpu.to(device).unsqueeze(0).expand(B, Cc)
                    merged_indices = torch.cat([left_idx, right_idx], dim=1)

                    vals, inds = torch.topk(merged_scores, k=min(K, merged_scores.size(1)), dim=1, largest=True, sorted=True)
                    new_indices = torch.gather(merged_indices, 1, inds)

                    topk_scores = vals.cpu()
                    topk_indices = new_indices.cpu()

                out = []
                for i in range(B):
                    out.append(list(zip(topk_indices[i].tolist(), topk_scores[i].tolist())))
                return out

            # 3) 计算排名（rank），两种场景：
            #    a) per_query_candidates=[B,K]（第0列为真值）——“1真+若干负例”
            #    b) labels=[B] + 全量候选（分块扫描）——“全语料库”排名
            if mode == "rank":
                # 3a) 每个query有独立候选集合（例如 1真+99负）
                if per_query_candidates is not None:
                    assert candidate_embeddings is not None, "per_query_candidates 需要 candidate_embeddings 以按索引切片"
                    idx_all = per_query_candidates  # [B, K]，每行第0个是ground-truth
                    Btot, K = idx_all.shape
                    # 分 query 小块，避免一次性切 [B,K,D] 过大
                    ranks_buf = []
                    q_chunk = max(1, min(Btot, 2048))
                    for i0 in range(0, Btot, q_chunk):
                        i1 = min(i0 + q_chunk, Btot)
                        q = q_emb[i0:i1].to(device, non_blocking=True)                 # [b, D]

                        flat_idx = idx_all[i0:i1].reshape(-1)                                                  # [b*K]
                        cand_cpu = candidate_embeddings.index_select(0, flat_idx).view(i1 - i0, K, -1)         # [b, K, D]
                        cand_cpu = F.normalize(cand_cpu, dim=2)                                                # ★ 新增：对每条候选单位化

                        s = torch.bmm(q.unsqueeze(1), cand_cpu.to(device, non_blocking=True).transpose(1, 2)).squeeze(1)  # [b, K]
                        true = s[:, 0]
                        ranks_chunk = (s > true.unsqueeze(1)).sum(dim=1).add_(1).cpu()

                        ranks_buf.append(ranks_chunk)
                    return torch.cat(ranks_buf)

                # 3b) 全语料库排名（labels 指明每个query的真值在全量候选中的索引）
                assert labels is not None, "全库排名需要提供 labels（每个query的真值全局索引）"
                assert N_total > 0, "没有候选向量用于排名"
                labels_cpu = torch.as_tensor(labels, dtype=torch.long, device="cpu")

                # --- 建立 (块ID, 块内列号) 映射（沿用你原来的逻辑） ---
                block_lookup = {}
                for b_id, idx_cpu in enumerate(cand_block_indices):
                    for local_col, gidx in enumerate(idx_cpu.tolist()):
                        block_lookup[gidx] = (b_id, local_col)

                pairs = [block_lookup.get(int(g), (-1, -1)) for g in labels_cpu]
                bids  = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
                lcols = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
                B, D = q_emb.shape
                assert (bids >= 0).all(), "有 labels 未出现在候选块中，请检查语料与标签一致性。"

                # --------- 尝试极速单通道（单块且能装下时）---------
                use_fast_path = False
                if (not cand_chunk_size or cand_chunk_size <= 0) and \
                   len(cand_blocks) == 1 and candidate_indices is None and torch.cuda.is_available():
                    free, _ = torch.cuda.mem_get_info()
                    # 预留 256MB，取 60% 安全系数，仅以分数矩阵大小评估
                    reserve = 256 * 1024 * 1024
                    budget  = max(1, int((free - reserve) * 0.60))
                    Cc_fast = cand_blocks[0].size(0)
                    if B * Cc_fast * 4 <= budget:
                        cand = cand_blocks[0].to(device, non_blocking=True)
                        scores = q_emb @ cand.t()                                # [B, N]
                        true_scores = scores.gather(1, labels_cpu.to(device).view(-1,1)).squeeze(1)
                        ranks = 1 + (scores > true_scores.unsqueeze(1)).sum(dim=1)
                        return ranks.cpu()
                # 否则走双向分块的稳妥路径

                # --------- 双向分块（查询×候选）  # NEW ----------
                def _auto_q_chunk(c_cols: int) -> int:
                    """根据当前候选子块列数，估一个安全的查询行数 Q（控制 Q×c_cols 的分数矩阵）。"""
                    if not torch.cuda.is_available():
                        return min(B, 1024)
                    free, _ = torch.cuda.mem_get_info()
                    reserve = 256 * 1024 * 1024
                    # 60% 用于 scores，余量给激活/中间张量
                    budget  = max(1, int((free - reserve) * 0.60))
                    # 每生成一列分数需要 Q 行（float32，每元素4字节）
                    max_q = max(1, budget // (max(1, c_cols) * 4))
                    # 不必太小，给个上限以减少循环次数
                    return max(1, min(B, max_q, 8192))

                # 如果用户给了 cand_chunk_size，则优先作为候选子块列数的上限；否则用之前的估算
                if cand_chunk_size and cand_chunk_size > 0:
                    base_c_chunk = int(cand_chunk_size)
                else:
                    base_c_chunk = self._auto_chunk_size(D, safety=0.60, hard_cap=8192)

                true_scores = torch.empty(B, dtype=q_emb.dtype, device=device)

                # --- 第一遍：只抽取真值分数（逐块×逐候选子块×逐查询子块）  # NEW ---
                for b_id, c_cpu in enumerate(cand_blocks):
                    Cc = c_cpu.size(0)
                    # 进一步把该块的候选分成子块，避免一次性拷整个块上 GPU
                    for j0 in range(0, Cc, base_c_chunk):
                        j1 = min(j0 + base_c_chunk, Cc)
                        cand_sub = c_cpu[j0:j1].to(device, non_blocking=True)     # [Cc', D]
                        # 动态估算查询子块大小，控制 scores 的 [Q, Cc'] 占用
                        q_chunk = _auto_q_chunk(cand_sub.size(0))
                        for i0 in range(0, B, q_chunk):
                            i1 = min(i0 + q_chunk, B)
                            q = q_emb[i0:i1]                                       # [Q, D]
                            scores = q @ cand_sub.t()                              # [Q, Cc']
                            # 仅当真值在 (b_id, [j0,j1)) 这个候选子块中时才抽取
                            mask_local = (bids[i0:i1] == b_id)
                            if mask_local.any():
                                cols_full = lcols[i0:i1]
                                in_range  = (cols_full >= j0) & (cols_full < j1) & mask_local
                                if in_range.any():
                                    rows   = in_range.nonzero(as_tuple=False).squeeze(1)
                                    cols_l = (cols_full[rows] - j0)                # 子块内列号
                                    true_scores[i0:i1].index_copy_(0, rows, scores[rows, cols_l])

                # --- 第二遍：统计大于真值分数的个数（同样双向分块）  # NEW ---
                count_g = torch.zeros(B, dtype=torch.long, device=device)
                for c_cpu in cand_blocks:
                    Cc = c_cpu.size(0)
                    for j0 in range(0, Cc, base_c_chunk):
                        j1 = min(j0 + base_c_chunk, Cc)
                        cand_sub = c_cpu[j0:j1].to(device, non_blocking=True)
                        q_chunk = _auto_q_chunk(cand_sub.size(0))
                        for i0 in range(0, B, q_chunk):
                            i1 = min(i0 + q_chunk, B)
                            q = q_emb[i0:i1]
                            scores = q @ cand_sub.t()                              # [Q, Cc']
                            ts = true_scores[i0:i1].unsqueeze(1)                   # [Q, 1]
                            count_g[i0:i1] += (scores > ts).sum(dim=1)

                ranks = 1 + count_g
                return ranks.cpu()


        finally:
            # 恢复旧状态
            try:
                if _old["deterministic"] is not None:
                    torch.use_deterministic_algorithms(_old["deterministic"])
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = _old["tf32_matmul"]
                    torch.backends.cudnn.allow_tf32 = _old["tf32_cudnn"]
                    torch.backends.cudnn.benchmark = _old["cudnn_bench"]
            except Exception:
                pass

class TreeMSCrossAttentionFusion(nn.Module):
    def __init__(self, spec_dim=1024, tree_dim=256,
                 hidden=512, heads=8, pool_heads=4, dropout=0.1,
                 pre_self_attn: bool = False):
        super().__init__()
        self.heads = heads
        self.qs = nn.Linear(spec_dim, hidden)
        self.ks = nn.Linear(tree_dim, hidden)
        self.qt = nn.Linear(tree_dim, hidden)
        self.kt = nn.Linear(spec_dim, hidden)
        self.pre_self_attn = bool(pre_self_attn)

        if self.pre_self_attn:
            enc_layer_s = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads, batch_first=True, activation=nn.GELU(), dropout=dropout
            )
            enc_layer_t = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads, batch_first=True, activation=nn.GELU(), dropout=dropout
            )
            self.self_encoder_s = nn.TransformerEncoder(enc_layer_s, num_layers=1)
            self.self_encoder_t = nn.TransformerEncoder(enc_layer_t, num_layers=1)
        else:
            self.self_encoder_s = None
            self.self_encoder_t = None

        self.attn_s2t = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=dropout)
        self.attn_t2s = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=dropout)

        self.ln_s = nn.LayerNorm(hidden)
        self.ln_t = nn.LayerNorm(hidden)

        self.ffn_s = nn.Sequential(nn.Linear(hidden, 2*hidden), nn.GELU(), nn.Linear(2*hidden, hidden))
        self.ffn_t = nn.Sequential(nn.Linear(hidden, 2*hidden), nn.GELU(), nn.Linear(2*hidden, hidden))

        self.pool_spec = MultiHeadMSAttentionPool(n_feats=hidden, n_head=pool_heads, concat=True, bias=True)
        self.pool_tree = MultiHeadMSAttentionPool(n_feats=hidden, n_head=pool_heads, concat=True, bias=True)

        self.out_dim = hidden * pool_heads * 2

    def forward(self, spec_seq, spec_pad_mask, tree_seq, tree_pad_mask):
        S = self.qs(spec_seq)
        T = self.ks(tree_seq)

        if self.pre_self_attn:
            S = self.self_encoder_s(S, src_key_padding_mask=spec_pad_mask)
            T = self.self_encoder_t(T, src_key_padding_mask=tree_pad_mask)

        # spectra -> attend to tree
        S_enh, _ = self.attn_s2t(
            S, T, T,
            key_padding_mask=tree_pad_mask
        )
        S_enh = self.ln_s(S + S_enh)
        S_enh = S_enh + self.ffn_s(S_enh)

        # tree -> attend to spectra (query=tree)
        Tq = self.qt(tree_seq)
        K  = self.kt(spec_seq)
        T_enh, _ = self.attn_t2s(
            Tq, K, K,
            key_padding_mask=spec_pad_mask
        )
        T_enh = self.ln_t(Tq + T_enh)
        T_enh = T_enh + self.ffn_t(T_enh)

        spec_vec = self.pool_spec(S_enh, spec_pad_mask)
        tree_vec = self.pool_tree(T_enh, tree_pad_mask)
        return torch.cat([spec_vec, tree_vec], dim=-1)
