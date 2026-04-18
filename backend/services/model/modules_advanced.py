import torch
from torch import nn
import torch.nn.functional as F
from config import CFG
import utils_advanced
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
        # 步骤1：初始化时计算 x，并注册为 buffer（自动随模型设备迁移）
        # 优先使用 CUDA 设备，若无则用 CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.arange(0, embedding_dim, 2, dtype=torch.float64, device=device)
        x = (
            2
            * math.pi
            * (
                self.lambda_min
                * (self.lambda_max / self.lambda_min) ** (x / (embedding_dim - 2))
            )
            ** -1
        )
        # 注册为 buffer，避免被当作参数更新，且自动随模型移至设备
        self.register_buffer('x', x)
        dropout_starting_layer = 0 if dropout_in_first_layer else 1
        for i in range(2):
            self.layers.append(nn.Linear(embedding_dim, embedding_dim))
            if i==0:
                self.layers.append(nn.ReLU())
            self.layers.append(nn.LayerNorm(embedding_dim))
            if dropout and i >= dropout_starting_layer:
                self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, mz):
        # 步骤2：移除原地设备迁移，改为使用已注册的 buffer（自动匹配 mz 设备）
        x = self.x.to(mz.device, non_blocking=True)
        # 原计算逻辑保持不变
        x = torch.einsum("bl,d->bld", mz, x)
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
        embed_dim: int = 512,
        project_size = 200,
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
        # raw_in_feats = 训练/推理时 V 的"实际输入维"；n_feats = 工作维(投影后的维)
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

class FPPredictionHead(nn.Module):
    """分子指纹预测头"""
    def __init__(self,
                 input_dim,
                 fp_dim,
                 hidden_dims=[1024, 512],
                 dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, fp_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # 使用sigmoid输出每个位为1的概率
        return torch.sigmoid(self.net(x))

import torch
from torch import nn
import torch.nn.functional as F

class FragSimiModel(nn.Module):
    def __init__(self, cfg, enable_compile: bool = False):
        super().__init__()
        self.cfg = cfg

        # --- 新增: torch.compile 配置 ---
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
            mol_embedding_dim +=2048

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

            # 获取 CLERMS_encoder 的实际嵌入维度
            spec_embed_dim = self.CLERMS_encoder.embed_dim 

            self.fusion = TreeMSCrossAttentionFusion(
                spec_dim=spec_embed_dim,  # <--- 改为动态获取 (或者是 512)
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
            ms_feature_dim += 200
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
            ms_feature_dim += 200
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        # 新增：FP预测头（使用线性层，输出logits）
        fp_nbits = int(getattr(self.cfg.model.mol_encoder.fp, 'nbits', 2048)) * 2
        self.fp_prediction_head = nn.Linear(ms_feature_dim, fp_nbits)
        
        # 不再需要投影头和对比损失相关参数
        self.ms_projection = None
        self.mol_projection = None
        
        # 移除对比损失相关参数
        self._learn_sigma = False
        self.log_inv_sigma = None
        
        # 损失函数配置（借鉴MIST模型）
        self.binarization_thresh = float(getattr(cfg.training, 'binarization_thresh', 0.5))
        self.loss_fn = str(getattr(cfg.training, 'loss_fn', 'cosine')).lower()
        
        # 定义损失函数（与MIST模型一致）
        self.bce_loss = nn.BCELoss(reduction="none")
        
        # 余弦损失函数（与MIST模型一致）
        cosine_sim = nn.CosineSimilarity(dim=-1)
        self.cosine_loss = lambda x, y: 1 - cosine_sim(
            x.expand(y.shape), y.float()
        ).unsqueeze(-1)
        
        # MSE损失函数
        self.mse_loss = nn.MSELoss(reduction="none")
        
        # 设置主损失函数
        if self.loss_fn == "bce":
            self.main_loss_fn = self.bce_loss
        elif self.loss_fn == "mse":
            self.main_loss_fn = self.mse_loss
        elif self.loss_fn == "cosine":
            self.main_loss_fn = self.cosine_loss
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")
        
        # 初始化参数
        nn.init.xavier_uniform_(self.fp_prediction_head.weight)
        if self.fp_prediction_head.bias is not None:
            nn.init.zeros_(self.fp_prediction_head.bias)
        
    def compute_loss(self, pred_fp, target_fp, train_step=True):
        """计算损失函数（借鉴MIST模型的设计）"""
        ret_dict = {}
        
        # 计算主损失
        if self.loss_fn == "bce":
            # 对BCE损失需要sigmoid
            pred_probs = torch.sigmoid(pred_fp)
            loss_full = self.main_loss_fn(pred_probs, target_fp)
        else:
            # 对于MSE和cosine，直接使用原始输出
            loss_full = self.main_loss_fn(pred_fp, target_fp)
        
        # 在指纹维度上取平均
        loss = loss_full.mean(-1)
        
        # 计算批次平均损失
        ret_dict["fp_loss"] = loss.mean().item()
        
        # 如果是BCE损失，计算准确率
        if self.loss_fn == "bce":
            with torch.no_grad():
                pred_probs = torch.sigmoid(pred_fp)
                pred_binary = (pred_probs > self.binarization_thresh).float()
                correct = (pred_binary == target_fp).float()
                ret_dict["fp_accuracy"] = correct.mean().item()
        
        # 如果是cosine损失，计算余弦相似度
        if self.loss_fn == "cosine":
            with torch.no_grad():
                # 计算余弦相似度
                cosine_sim_value = F.cosine_similarity(pred_fp, target_fp, dim=-1)
                ret_dict["cosine_similarity"] = cosine_sim_value.mean().item()
        
        # 总损失
        total_loss = loss.mean()
        ret_dict["loss"] = total_loss.item()
        
        # 验证时额外计算其他损失指标
        if not train_step:
            # 验证时总是计算BCE和cosine损失
            with torch.no_grad():
                pred_probs = torch.sigmoid(pred_fp)
               
                
                # 余弦损失
                cosine_loss_val = self.cosine_loss(pred_fp, target_fp).mean()
                ret_dict["cos_loss"] = cosine_loss_val.item()
                
                # 余弦相似度
                cosine_sim_value = F.cosine_similarity(pred_fp, target_fp, dim=-1).mean()
                ret_dict["cosine_sim"] = cosine_sim_value.item()
                
                # 准确率
                pred_binary = (pred_probs > self.binarization_thresh).float()
                correct = (pred_binary == target_fp).float()
                ret_dict["accuracy"] = correct.mean().item()
        
        return total_loss, ret_dict

    def forward(self, batch, train_step=True):
        device = next(self.parameters()).device

        # 获取分子指纹（作为预测目标）
        if 'fp' in self.cfg.model.mol_encoder.type:
            target_fps = batch["mol_fps"].to(device).float()
        else:
            # 如果没有FP，创建全零目标（如果需要的话）
            fp_nbits = int(getattr(self.cfg.model.mol_encoder.fp, 'nbits', 2048))
            target_fps = torch.zeros((batch['spec_tensor'].shape[0], fp_nbits), device=device)

        # 获取质谱特征
        spec_tensor = batch['spec_tensor'].squeeze(1).to(device)
        adduct_indices = batch['adduct_type_idx'].to(device)
        precursor = batch['precursor_mz'].to(device)

        # 获取质谱特征（根据fusion_mode）
        if self.fusion_mode == "xattn":
            spec_seq, spec_pad_mask, _ = self.CLERMS_encoder.encode_tokens(spec_tensor, precursor, adduct_indices)
            pyg_data = batch['pyg_data'].to(device)
            tree_seq, tree_pad_mask = self.trees_encoder(pyg_data, return_nodes=True)
            ms_features = self.fusion(
                spec_seq, spec_pad_mask,
                tree_seq, tree_pad_mask
            )
        elif self.fusion_mode == "concat":
            spectrum_enc = self.CLERMS_encoder(spec_tensor, precursor, adduct_indices)
            pyg_data = batch['pyg_data'].to(device)
            tree_enc = self.trees_encoder(pyg_data)
            ms_features = torch.cat([spectrum_enc, tree_enc], dim=-1)
        elif self.fusion_mode in ("tree-only","tree_only"):
            pyg_data = batch['pyg_data'].to(device)
            ms_features = self.trees_encoder(pyg_data)
        else:
            ms_features = self.CLERMS_encoder(spec_tensor, precursor, adduct_indices)

        # 预测分子指纹（输出logits）
        fp_logits = self.fp_prediction_head(ms_features)
        
        # 计算损失
        total_loss, loss_dict = self.compute_loss(fp_logits, target_fps, train_step=train_step)
        
        # 如果需要概率输出（用于监控）
        with torch.no_grad():
            predicted_probs = torch.sigmoid(fp_logits)
        
        # 返回结果
        return {
            'loss': total_loss,
            'fp_loss': loss_dict.get('fp_loss', 0.0),
            'predicted_fps': predicted_probs,
            'target_fps': target_fps,
            'fp_logits': fp_logits,
            'loss_dict': loss_dict  
        }

    def encode_mol(self, batch):
        """编码分子（如果需要的话）"""
        device = next(self.parameters()).device
        mol_feat_list = []
        if 'gnn' in self.cfg.model.mol_encoder.type:
            gnn_batch = {k: v.to(device) for k, v in batch.items() if k in ['V', 'A', 'mol_size']}
            mol_feat_list.append(self.mol_gnn_encoder(gnn_batch))
        if 'fp' in self.cfg.model.mol_encoder.type:
            mol_feat_list.append(batch["mol_fps"].to(device))
        if mol_feat_list:
            mol_features = torch.cat(mol_feat_list, dim=1)
            return mol_features
        else:
            # 返回空张量或None
            return torch.empty((batch['spec_tensor'].shape[0], 0), device=device)

    def encode_ms(self, batch):
        """编码质谱，返回特征（不预测FP）"""
        device = next(self.parameters()).device
        spec_tensor = batch['spec_tensor'].squeeze(1).to(device)
        adduct_indices = batch['adduct_type_idx'].to(device)
        precursor = batch['precursor_mz'].to(device)

        if self.fusion_mode == "xattn":
            spec_seq, spec_pad_mask, _ = self.CLERMS_encoder.encode_tokens(spec_tensor, precursor, adduct_indices)
            pyg_data = batch['pyg_data'].to(device)
            tree_seq, tree_pad_mask = self.trees_encoder(pyg_data, return_nodes=True)
            ms_features = self.fusion(
                spec_seq, spec_pad_mask,
                tree_seq, tree_pad_mask
            )
        elif self.fusion_mode == "concat":
            spectrum_encodings = self.CLERMS_encoder(spec_tensor, precursor, adduct_indices)
            pyg_data = batch['pyg_data'].to(device)
            trees_encodings = self.trees_encoder(pyg_data)
            ms_features = torch.cat([spectrum_encodings, trees_encodings], dim=-1)
        elif self.fusion_mode in ("tree-only", "tree_only"):
            pyg_data = batch['pyg_data'].to(device)
            ms_features = self.trees_encoder(pyg_data)
        else:
            ms_features = self.CLERMS_encoder(spec_tensor, precursor, adduct_indices)
            
        return ms_features
    
    def predict_fp(self, batch):
        """预测分子指纹"""
        ms_features = self.encode_ms(batch)
        predicted_fps = self.fp_prediction_head(ms_features)
        return predicted_fps
    
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
        labels=None,                                        # 1D tensor/list，长度=B；每个query的真值在"全语料库"中的全局索引
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
            * per_query_candidates=[B,K]（第0列为真值）——用于"1真+若干负例"的排名
            * 或 labels=[B] + 全语料库 candidate_embeddings ——用于"全语料库"排名（分块显存友好）
        """
        # 注意：由于我们移除了对比学习，这个predict函数可能需要重新设计
        # 目前保留原接口，但实际可能不再需要
        print("Warning: predict() method may not work properly after removing contrastive learning.")
        return None


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