# a) 如果需要极致的确定性, 请启动如下代码块
'''
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # deterministic GEMMs
os.environ['PYTHONHASHSEED'] = '0'

import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try: torch.set_float32_matmul_precision('highest')
except: pass
'''

# b) 优化极致性能, 启动以下代码块, 接受统计意义上的确定
import os
import torch
# 启用 cuDNN benchmark 模式
torch.backends.cudnn.benchmark = True
# 在支持的硬件上启用 TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')  # 可选 'high' 或 'medium'
except:
    pass
# Flash SDP 核加速注意力计算
from torch.backends.cuda import sdp_kernel
sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)

from utils import *
import logging
from modules import * # 确保包含所需模型组件
import sys
import numpy as np
from tqdm import tqdm
import random
from torch import nn
from config import CFG
from dataset import PreprocessedDataset  # 确保数据集类正确导入
import torch.utils.data
import copy, json, pickle
import re
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch import amp
from torch.utils.data import Sampler
from torch_geometric.utils import to_dense_batch, to_dense_adj

# ===== Warmup + Cosine (per-step) LR scheduler =====
import math

class WarmupCosineLRScheduler:
    """简单的 per-batch 学习率调度器：前 warmup_steps 线性升温，后余弦退火"""
    def __init__(self, optimizer, total_steps, warmup_steps=0, max_lr=1e-4, min_lr=5e-6, last_step=0):
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.last_step = int(last_step)
        self._apply_lr(self.last_step)

    def _lr_at(self, s: int) -> float:
        if s < self.warmup_steps:
            return self.max_lr * (s / max(1, self.warmup_steps))
        progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def _apply_lr(self, s: int) -> float:
        lr = self._lr_at(s)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def step(self) -> float:
        self.last_step += 1
        return self._apply_lr(self.last_step)

    def state_dict(self):
        return {
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'max_lr': self.max_lr,
            'min_lr': self.min_lr,
            'last_step': self.last_step,
        }

    def load_state_dict(self, state: dict):
        self.total_steps = int(state.get('total_steps', self.total_steps))
        self.warmup_steps = int(state.get('warmup_steps', self.warmup_steps))
        self.max_lr = float(state.get('max_lr', self.max_lr))
        self.min_lr = float(state.get('min_lr', self.min_lr))
        self.last_step = int(state.get('last_step', self.last_step))
        self._apply_lr(self.last_step)


class EpochShuffleSampler(Sampler):
    """训练用可控打乱采样器：按 epoch 确定性打乱，恢复训练时顺序一致"""
    def __init__(self, data_source, seed: int):
        super().__init__(data_source)
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        import numpy as np
        n = len(self.data_source)
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = rng.permutation(n).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)


def seed_worker(worker_id):
    """DataLoader worker_init_fn：确保数据加载可复现"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_next_record_dir(basedir, prefix=''):
    path = '%s/%%s001/' % basedir
    n = 2
    while os.path.exists(path % prefix):
        path = '%s/%%s%.3d/' % (basedir, n)
        n += 1
    pth = path % prefix
    os.makedirs(pth)
    return pth

def setup_seed(seed):
    """设置全局随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_module_requires_grad(module, requires_grad):
    """设置模块参数是否可训练"""
    for param in module.parameters():
        param.requires_grad = requires_grad

def pad_spec_tensor(tensor, max_length):
    """谱图张量padding到固定长度"""
    current_length = tensor.size(1)
    if current_length < max_length:
        padding = torch.zeros((tensor.size(0), max_length - current_length, tensor.size(2)),
                              dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=1)
    return tensor


def my_collate(batch):
    """自定义DataLoader拼接函数：处理分子/质谱数据的变长padding"""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {}
    
    # 初始化所有可能的字段列表
    fields = {
        'ms_bins': [],
        'mol_fps': [],
        'mol_fmvec': [],
        'V': [],
        'A': [],
        'mol_size': [],
        'pyg_data': [],
        'spec_tensor': [],
        'compound_id': [],
        'adduct_type_idx': [],
        'precursor_mz':[],
    }
    
    # 收集批次中存在的字段
    for b in batch:
        for field in fields:
            if field in b:
                fields[field].append(b[field])
    
    bat = {}
    for field in fields:
        if fields[field]:
            if field == 'compound_id':
                bat[field] = torch.tensor(fields[field], dtype=torch.long)
            elif field == 'adduct_type_idx':
                bat[field] = torch.tensor(fields[field], dtype=torch.long)
            elif field == 'precursor_mz':
                bat[field]=torch.tensor(fields[field])
            elif field in ['ms_bins', 'mol_fps', 'mol_fmvec', 'V', 'A', 'spec_tensor']:
                if field == 'V':
                    max_n = max([v.shape[0] for v in fields['V']])
                    padded_V = [pad_V(v, max_n) for v in fields['V']]
                    bat['V'] = torch.stack(padded_V)
                elif field == 'A':
                    max_n = max([a.shape[0] for a in fields['A']])
                    padded_A = [pad_A(a, max_n) for a in fields['A']]
                    bat['A'] = torch.stack(padded_A)
                elif field == 'spec_tensor':
                    try:
                        bat['spec_tensor'] = torch.stack(fields['spec_tensor'])
                    except RuntimeError as e:
                        raise RuntimeError("谱图张量长度不一致，无法堆叠成批次，这很可能是因为你的预处理数据集中混合了用不同参数生成的文件。") from e
                else:
                    bat[field] = torch.stack(fields[field])
            elif field == 'mol_size':
                bat['mol_size'] = torch.cat(fields['mol_size'], dim=0)
            elif field == 'pyg_data':
                bat['pyg_data'] = Batch.from_data_list(fields['pyg_data'])
    return bat

class FixedSeedShuffleSampler(Sampler):
    """验证集固定顺序采样器：初始化时打乱一次，后续保持顺序一致"""
    def __init__(self, data_source, seed):
        super().__init__(data_source)
        self.data_source = data_source
        self.seed = seed
        rng = np.random.RandomState(self.seed)
        self.shuffled_indices = rng.permutation(len(self.data_source))

    def __iter__(self):
        return iter(self.shuffled_indices)

    def __len__(self):
        return len(self.data_source)


def build_loaders(processed_dir, mode, cfg, num_workers):
    """构建训练/验证DataLoader：训练用epoch打乱，验证用固定顺序"""
    dataset = PreprocessedDataset(processed_dir)
    
    # 可复现的随机生成器
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    
    sampler = None
    shuffle = False

    if mode == "train":
        sampler = EpochShuffleSampler(dataset, seed=cfg.seed)  # 按epoch可控打乱
    else:  # "valid" 模式
        sampler = FixedSeedShuffleSampler(dataset, seed=cfg.seed)  # 固定顺序
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=my_collate,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=(mode == "train"),  # 训练集丢弃最后不完整批次
    )
    return dataloader


def train_batch(model, batch, optimizer, scaler, use_amp, use_scaler, amp_dtype, cfg, ema=None):
    """单batch训练逻辑：抽离为独立函数，便于控制每batch后验证"""
    # 数据移至设备
    for k, v in batch.items():
        if hasattr(v, 'to'):
            batch[k] = v.to(CFG.device, non_blocking=True)
    
    # AMP自动混合精度训练
    with amp.autocast(device_type='cuda', enabled=use_amp, dtype=amp_dtype):
        loss_dict = model(batch)
        total_loss = loss_dict['total_loss']
    
    # 梯度更新
    optimizer.zero_grad(set_to_none=True)
    if use_scaler:
        scaler.scale(total_loss).backward()
        # 梯度裁剪（防止梯度爆炸）
        grad_clip = float(getattr(cfg.training, "grad_clip", -1))
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        grad_clip = float(getattr(cfg.training, "grad_clip", -1))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    
    # EMA模型更新（如果启用）
    if ema is not None:
        ema.update(model)
    
    # 计算当前batch的样本数（用于平均loss）
    count = batch['pyg_data'].num_graphs if "pyg_data" in batch else batch['mol_size'].size(0)
    return loss_dict, count


def valid_epoch(model, valid_loader, use_amp, amp_dtype, cfg, ema=None):
    """完整验证epoch：计算平均验证loss"""
    # 初始化多损失分量的平均器
    loss_meters = {
        'total_loss': AvgMeter(),
        'cross_loss': AvgMeter(),
        'mol_loss': AvgMeter(),
        'ms_loss': AvgMeter(),
        'mse_loss': AvgMeter()
    }
    
    tqdm_object = tqdm(valid_loader, total=len(valid_loader), desc="Validation")

    # 应用EMA模型（如果启用）
    applied_ema = False
    if ema is not None:
        ema.apply_to(model)
        applied_ema = True

    model.eval()
    try:
        with torch.no_grad():  # 验证阶段禁用梯度计算
            for batch in tqdm_object:
                if not batch:
                    continue
                # 数据移至设备
                for k, v in batch.items():
                    if hasattr(v, 'to'):
                        batch[k] = v.to(CFG.device, non_blocking=True)
                
                # AMP推理
                with amp.autocast(device_type='cuda', enabled=use_amp, dtype=amp_dtype):
                    loss_dict = model(batch)
                
                # 更新验证loss统计
                count = batch['pyg_data'].num_graphs if 'pyg_data' in batch else batch['mol_size'].size(0)
                for key, meter in loss_meters.items():
                    if key in loss_dict:
                        meter.update(loss_dict[key].item(), count)
                
                # 在进度条上显示所有损失
                postfix = {
                    'total': f"{loss_meters['total_loss'].avg:.4f}",
                    'cross': f"{loss_meters['cross_loss'].avg:.4f}"
                }
                if loss_meters['mol_loss'].count > 0:
                    postfix['mol'] = f"{loss_meters['mol_loss'].avg:.4f}"
                if loss_meters['ms_loss'].count > 0:
                    postfix['ms'] = f"{loss_meters['ms_loss'].avg:.4f}"
                if loss_meters['mse_loss'].count > 0:
                    postfix['mse'] = f"{loss_meters['mse_loss'].avg:.4f}"
                
                tqdm_object.set_postfix(** postfix)
    finally:
        # 恢复原模型（如果应用了EMA）
        if applied_ema:
            ema.restore(model)

    return loss_meters


def main(cfg, savedir='data/train'):
    setup_seed(cfg.seed)

    # 数据路径设置
    base_processed_dir = cfg.data.processed_dir_base
    train_processed_dir = os.path.join(base_processed_dir, 'train.lmdb')
    valid_processed_dir = os.path.join(base_processed_dir, 'valid.lmdb')

    # AMP配置与变量初始化
    start_epoch = 0
    global_train_batch = 0  # 全局训练batch计数（跨epoch）
    best_loss = float('inf')
    best_model_fns = []

    use_amp = cfg.training.use_amp
    amp_dtype_str_from_config = cfg.training.amp_dtype
    effective_amp_dtype_str = amp_dtype_str_from_config
    # 检查BF16支持性
    if use_amp and effective_amp_dtype_str == 'bfloat16' and not torch.cuda.is_bf16_supported():
        logging.warning("设备不支持 bfloat16，自动切换回 float16。")
        effective_amp_dtype_str = 'float16'
    amp_dtype = torch.bfloat16 if effective_amp_dtype_str == 'bfloat16' else torch.float16
    use_scaler = use_amp and effective_amp_dtype_str == 'float16'
    scaler = amp.GradScaler(enabled=use_scaler)

    logging.info(f"自动混合精度训练 (AMP) 已 {'启用' if use_amp else '禁用'}。")
    if use_amp:
        logging.info(f"AMP 数据类型: {effective_amp_dtype_str}。梯度缩放 (Scaler): {'启用' if use_scaler else '禁用'}")

    # 模型初始化
    model = FragSimiModel(cfg).to(cfg.device)
    logging.info(f"模型结构:\n{model}")

    # EMA模型配置
    ema_cfg = getattr(cfg.training, "ema", {})
    ema_enabled = bool(getattr(ema_cfg, "enabled", False))
    ema = None
    if ema_enabled:
        ema_decay = float(getattr(ema_cfg, "decay", 0.999))
        ema = ModelEMA(model, decay=ema_decay)
        logging.info(f"EMA 已启用: decay={ema_decay}")
    else:
        logging.info("EMA 未启用")

    # 数据加载器
    logging.info(f"从预处理目录加载训练数据: {train_processed_dir}")
    logging.info(f"从预处理目录加载验证数据: {valid_processed_dir}")

    num_workers = cfg.data.num_workers
    logging.info(f"使用 {num_workers} 个 worker 进程加载数据。")

    train_loader = build_loaders(train_processed_dir, "train", cfg, num_workers)
    valid_loader = build_loaders(valid_processed_dir, "valid", cfg, num_workers)

    # 优化器
    base_lr = float(getattr(cfg.training, "lr", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=cfg.training.weight_decay, fused=True)

    checkpoint = None

    # 从checkpoint恢复训练
    if cfg.training.resume_path:
        logging.info(f"尝试从 {cfg.training.resume_path} 恢复训练。")
        if not os.path.exists(cfg.training.resume_path):
            error_msg = f"错误: 恢复路径不存在: {cfg.training.resume_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        checkpoint = torch.load(cfg.training.resume_path, map_location=cfg.device, weights_only=False)

        # 加载模型状态
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        if missing or unexpected:
            logging.info(f"模型参数加载完成，缺失: {len(missing)}, 多余: {len(unexpected)}")
        
        # 加载优化器和训练状态
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        global_train_batch = checkpoint.get('global_train_batch', 0)  # 恢复全局batch计数
        best_loss = checkpoint.get('best_loss', float('inf'))
        
        if use_scaler and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
            logging.info("成功恢复GradScaler状态。")
        
        # 恢复EMA
        if ema_enabled and 'ema' in checkpoint and checkpoint['ema'] is not None:
            try:
                ema.load_state_dict(checkpoint['ema'])
                logging.info("已恢复EMA状态。")
            except Exception as e:
                logging.warning(f"恢复EMA失败：{e}")

        # 恢复RNG状态
        try:
            from utils import restore_rng_state
            if 'rng_state' in checkpoint:
                restore_rng_state(checkpoint['rng_state'])
                logging.info("已恢复随机数状态。")
        except Exception as e:
            logging.warning(f"恢复RNG状态失败：{e}")

        best_model_fns = checkpoint.get('best_models_meta', [])
        logging.info(f"从epoch {start_epoch + 1} 继续训练，当前最佳验证loss: {best_loss:.4f}")
    else:
        logging.info("未提供恢复路径，将从头开始训练")
    
    # 学习率调度器设置
    lr_mode = str(getattr(cfg.training, "lr_mode", "plateau")).lower()
    batch_scheduler = None
    plateau_scheduler = None
    
    steps_per_epoch = len(train_loader)
    total_steps = cfg.training.epochs * steps_per_epoch

    if checkpoint is not None:  # 恢复训练
        lr_mode_from_ckpt = checkpoint.get('lr_mode', 'plateau')
        logging.info(f"从checkpoint恢复调度器模式: {lr_mode_from_ckpt}")
        lr_mode = lr_mode_from_ckpt

        if lr_mode == "plateau":
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=cfg.training.patience, factor=cfg.training.factor
            )
            if 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler'] is not None:
                try:
                    plateau_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    logging.info("已恢复ReduceLROnPlateau状态。")
                except Exception as e:
                    logging.warning(f"恢复Plateau调度器失败: {e}")
        
        elif lr_mode == "cosine":
            warmup_pct = float(getattr(cfg.training, "warmup_pct", 0.03))
            max_lr = float(getattr(cfg.training, "max_lr", base_lr))
            min_lr = float(getattr(cfg.training, "min_lr", 5e-6))
            warmup_steps = int(round(total_steps * warmup_pct))
            
            batch_scheduler = WarmupCosineLRScheduler(
                optimizer=optimizer, total_steps=total_steps, warmup_steps=warmup_steps,
                max_lr=max_lr, min_lr=min_lr, last_step=0 
            )
            if 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler'] is not None:
                try:
                    batch_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    logging.info(f"已恢复Cosine调度器，当前step: {batch_scheduler.last_step}")
                except Exception as e:
                    logging.warning(f"恢复Cosine调度器失败: {e}")

    else:  # 从头训练
        logging.info(f"使用学习率调度器模式: {lr_mode}")
        if lr_mode == "plateau":
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=cfg.training.patience, factor=cfg.training.factor
            )
            logging.info("使用ReduceLROnPlateau调度器")

        elif lr_mode == "fixed":
            logging.info(f"使用固定学习率: {base_lr:.8f}")

        elif lr_mode == "cosine":
            warmup_pct = float(getattr(cfg.training, "warmup_pct", 0.03))
            max_lr = float(getattr(cfg.training, "max_lr", base_lr))
            min_lr = float(getattr(cfg.training, "min_lr", 5e-6))
            warmup_steps = int(round(total_steps * warmup_pct))
            
            batch_scheduler = WarmupCosineLRScheduler(
                optimizer=optimizer, total_steps=total_steps, warmup_steps=warmup_steps,
                max_lr=max_lr, min_lr=min_lr, last_step=0
            )
            logging.info(f"使用Warmup+Cosine调度: total_steps={total_steps}, warmup_steps={warmup_steps}")

        else:
            logging.warning(f"未知lr_mode='{lr_mode}'，回退为plateau")
            lr_mode = "plateau"
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=cfg.training.patience, factor=cfg.training.factor
            )

    # 验证间隔配置（每1000个batch验证一次）
    VALID_BATCH_INTERVAL = 1000
    logging.info(f"验证策略：每训练{VALID_BATCH_INTERVAL}个batch或完成一个epoch后进行验证")

    # 主训练循环
    for epoch in range(start_epoch, cfg.training.epochs):
        logging.info(f"========== Epoch: {epoch + 1}/{cfg.training.epochs} ==========")
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        # 初始化多损失分量的平均器
        train_loss_meters = {
            'total_loss': AvgMeter(),
            'cross_loss': AvgMeter(),
            'mol_loss': AvgMeter(),
            'ms_loss': AvgMeter(),
            'mse_loss': AvgMeter()
        }
        
        epoch_batch_count = 0  # 当前epoch内的batch计数

        # 遍历训练batch，将tqdm迭代器命名为pbar以避免与batch变量混淆
        pbar = tqdm(train_loader, total=len(train_loader), desc="Training")
        for batch in pbar:
            if not batch:
                continue
            
            # 学习率调度（cosine模式按batch更新）
            if lr_mode == "cosine" and batch_scheduler is not None:
                current_lr = batch_scheduler.step()
            else:
                current_lr = optimizer.param_groups[0]['lr']

            # 单batch训练
            loss_dict, count = train_batch(
                model, batch, optimizer, scaler, use_amp, use_scaler, amp_dtype, cfg, ema=ema
            )
            
            # 更新训练统计
            for key, meter in train_loss_meters.items():
                if key in loss_dict:
                    meter.update(loss_dict[key].item(), count)
            
            epoch_batch_count += 1
            global_train_batch += 1

            

            # 在进度条上显示所有损失
            postfix = {
                'total': f"{train_loss_meters['total_loss'].avg:.4f}",
                'cross': f"{train_loss_meters['cross_loss'].avg:.4f}",
                'lr': f"{current_lr:.6f}"
            }
            if train_loss_meters['mol_loss'].count > 0:
                postfix['mol'] = f"{train_loss_meters['mol_loss'].avg:.4f}"
            if train_loss_meters['ms_loss'].count > 0:
                postfix['ms'] = f"{train_loss_meters['ms_loss'].avg:.4f}"
            if train_loss_meters['mse_loss'].count > 0:
                postfix['mse'] = f"{train_loss_meters['mse_loss'].avg:.4f}"
            
            # 使用tqdm迭代器对象pbar来设置后缀，而不是batch
            pbar.set_postfix(** postfix)

            # 每1000个batch验证一次
            if global_train_batch % VALID_BATCH_INTERVAL == 0:
                logging.info(f"\n全局第{global_train_batch}个batch，开始验证...")
                valid_loss_meters = valid_epoch(model, valid_loader, use_amp, amp_dtype, cfg, ema=ema)
                valid_loss = valid_loss_meters['total_loss'].avg
                logging.info(f"验证结果：valid_loss={valid_loss:.4f}")

                # 保存最佳模型
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model_fn = f"{savedir}/model_batch-{global_train_batch}_vloss-{best_loss:.4f}.pth"
                    
                    checkpoint = {
                        'epoch': epoch + 1,
                        'global_train_batch': global_train_batch,
                        'best_loss': best_loss,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': (
                            plateau_scheduler.state_dict() if lr_mode == "plateau"
                            else (batch_scheduler.state_dict() if lr_mode == "cosine" else None)
                        ),
                        'lr_mode': lr_mode,
                        'scaler': scaler.state_dict() if use_scaler else None,
                        'rng_state': capture_rng_state(),
                        'best_models_meta': best_model_fns,
                        'config': dict(CFG),
                        'ema': ema.state_dict() if ema_enabled and (ema is not None) else None,
                    }
                    torch.save(checkpoint, best_model_fn)
                    logging.info(f"保存新最佳模型到: {best_model_fn} (valid_loss={best_loss:.4f})")
                    
                    # 维护最佳模型列表（只保留最新的几个）
                    best_model_fns.append((best_loss, best_model_fn))
                    best_model_fns.sort(key=lambda x: x[0])  # 按loss升序排序
                    
                    models_to_keep = cfg.training.keep_best_models_num
                    if len(best_model_fns) > models_to_keep:
                        _, fn_to_remove = best_model_fns.pop(-1)
                        if os.path.exists(fn_to_remove):
                            os.remove(fn_to_remove)
                            logging.info(f"删除旧模型: {fn_to_remove}")

                # 验证后切换回训练模式
                model.train()

        # 每个epoch结束后也进行一次验证
        logging.info(f"\nEpoch {epoch + 1} 训练完成，开始epoch验证...")
        valid_loss_meters = valid_epoch(model, valid_loader, use_amp, amp_dtype, cfg, ema=ema)
        valid_loss = valid_loss_meters['total_loss'].avg
        logging.info(f"Epoch {epoch + 1} 验证结果：valid_loss={valid_loss:.4f}")


        # 学习率调度（plateau模式按epoch更新）
        if lr_mode == "plateau" and plateau_scheduler is not None:
            plateau_scheduler.step(valid_loss)

        # 保存epoch级最佳模型
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_fn = f"{savedir}/model_epoch-{epoch+1}_vloss-{best_loss:.4f}.pth"
            
            checkpoint = {
                'epoch': epoch + 1,
                'global_train_batch': global_train_batch,
                'best_loss': best_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': (
                    plateau_scheduler.state_dict() if lr_mode == "plateau"
                    else (batch_scheduler.state_dict() if lr_mode == "cosine" else None)
                ),
                'lr_mode': lr_mode,
                'scaler': scaler.state_dict() if use_scaler else None,
                'rng_state': capture_rng_state(),
                'best_models_meta': best_model_fns,
                'config': dict(CFG),
                'ema': ema.state_dict() if ema_enabled and (ema is not None) else None,
            }
            torch.save(checkpoint, best_model_fn)
            logging.info(f"保存新最佳模型到: {best_model_fn} (valid_loss={best_loss:.4f})")
            
            best_model_fns.append((best_loss, best_model_fn))
            best_model_fns.sort(key=lambda x: x[0])
            
            models_to_keep = cfg.training.keep_best_models_num
            if len(best_model_fns) > models_to_keep:
                _, fn_to_remove = best_model_fns.pop(-1)
                if os.path.exists(fn_to_remove):
                    os.remove(fn_to_remove)
                    logging.info(f"删除旧模型: {fn_to_remove}")

    best_model_fnl = [fn for _, fn in best_model_fns]
    logging.info(f"\n训练结束。最佳验证损失: {best_loss:.4f}")
    logging.info(f"最终保留的模型: {best_model_fnl}")
    return best_model_fnl, best_losss

if __name__ == "__main__":
    default_config_file = 'cfgbest.json'
    config_to_load = default_config_file

    # 命令行参数优先级高于默认配置
    if len(sys.argv) > 1:
        config_to_load = sys.argv[1]
        print(f"使用命令行指定的配置文件: {config_to_load}")
    else:
        print(f"未指定配置文件，使用默认: {default_config_file}")

    try:
        CFG.load(config_to_load)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保cfgbest.json存在或通过命令行提供正确路径")
        sys.exit(1)
    
    # 设置保存目录
    savedir = 'outputs/'
    os.makedirs(savedir, exist_ok=True)

    project_name = getattr(CFG, 'project_name', 'training_run')
    base_path = os.path.join(savedir, project_name)
    subdir = base_path
    counter = 1
    while os.path.exists(subdir):
        subdir = f"{base_path}_{counter}"
        counter += 1
    os.makedirs(subdir)

    # 配置日志
    log_file_path = os.path.join(subdir, 'training.log')
    setup_logging(log_file_path)  # 确保utils中有此函数
    CFG.save(f'{subdir}/config.json')

    logging.info("--- 当前运行配置 ---")
    for key, value in sorted(CFG.items()):
        logging.info(f"{key:<25}: {value}")
    logging.info("--------------------------\n")

    # 启动训练
    main(CFG, subdir)
    