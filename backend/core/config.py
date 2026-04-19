import os
from pathlib import Path

# 单文件大小限制：100MB
file_size_limit = 100 * 1024 * 1024

# 服务 IP（健康检查返回字段）
server_ip = "127.0.0.1"

# Web 服务监听配置
# 设为 0.0.0.0 以接收外部容器或宿主机转发访问
backend_host = "0.0.0.0"
backend_port = 6008
backend_reload = False

# 目录基准（固定约定：web 与 autodl-tmp 同级）
backend_dir = Path(__file__).resolve().parents[1]
web_dir = backend_dir.parent
autodl_tmp_dir = web_dir.parent / "autodl-tmp"

# 上传目录（临时文件目录）
temp_dir = backend_dir / "temp"

# 用户数据根目录（多用户隔离目录）
user_data_root = backend_dir / "user_data"
# 兼容旧变量名（建议新代码使用 user_data_root）
user_data_dir = user_data_root

# 认证与数据库配置
database_url = os.getenv("DATABASE_URL", f"sqlite:///{backend_dir / 'users.db'}")
jwt_secret_key = os.getenv("JWT_SECRET_KEY", "please-change-this-secret")
jwt_algorithm = "HS256"
jwt_access_token_expire_minutes = 60 * 24

# 任务空间根目录映射：当前统一到 temp（advanced 兼容映射到同目录）
task_storage_roots = {
    "normal": user_data_dir,
    "advanced": user_data_dir,
}

# 历史记录文件（先使用 JSON 管理固定用户）
history_records_path = user_data_dir / "history_records.json"

# smiles图片缓存目录
smiles_image_dir = backend_dir / "smiles_image"

# 允许上传的后缀（小写）
allowed_mgf_extensions = {"mgf", "txt"}
allowed_json_extensions = {"json"}

# 预处理输出文件路径
valid_pairs_spectra_path = temp_dir / "valid_pairs_spectra.mgf"
valid_pairs_fragtrees_path = temp_dir / "valid_pairs_fragtrees.json"
statas_path = temp_dir / "statas.json"

# 说明文档配置
# 源文档（可手动修改）
docs_source_path = backend_dir / "services" / "文档说明.md"
# 解析后的结构化结果缓存（保存到 temp 目录）
docs_structured_cache_path = temp_dir / "docs_content.json"

# 检索模型与候选库路径配置
retrieve_model_weight_path_pos = autodl_tmp_dir / "model_epoch-36_vloss-0.1429.pth"
retrieve_model_weight_path_neg = autodl_tmp_dir / "model_epoch-43_vloss-0.2149.pth"
# 兼容旧逻辑（默认正离子模型）
retrieve_model_weight_path = retrieve_model_weight_path_pos
retrieve_model_weight_paths = {
    "pos": retrieve_model_weight_path_pos,
    "neg": retrieve_model_weight_path_neg,
}
retrieve_pubchem_parquet_path = autodl_tmp_dir / "pubchem_final_with_formula.parquet"
retrieve_database_root = autodl_tmp_dir / "database"
retrieve_shared_smiles_txt_path = backend_dir / "testdata" / "common.txt"

# 高级检索模型路径配置
retrieve_advanced_model_weight_path_pos = autodl_tmp_dir / "ft_e8_loss1.0037.pth"
retrieve_advanced_model_weight_path_neg = autodl_tmp_dir / "ft_e5_loss1.0547.pth"
retrieve_advanced_model_weight_paths = {
    "pos": retrieve_advanced_model_weight_path_pos,
    "neg": retrieve_advanced_model_weight_path_neg,
}

# 检索运行参数
# 保留该参数用于兼容旧逻辑；单GPU队列模式下固定为1
retrieve_workers = 1
retrieve_max_pending_jobs = 512
retrieve_task_keep_seconds = 3600
retrieve_mp_start_method = "spawn"
# 结果队列消费线程轮询间隔（秒）
retrieve_queue_wait_timeout = 1

# 高级检索运行参数
retrieve_advanced_max_pending_jobs = 256
retrieve_advanced_task_keep_seconds = 3600
retrieve_advanced_mp_start_method = "spawn"
retrieve_advanced_queue_wait_timeout = 0.2

# CORS 配置（用于前端跨域访问）
# Electron 打包后常见 Origin 为 "null"；AutoDL 外网转发为 https://*.bjb2.seetacloud.com:8443
cors_allow_origins = [
    "http://127.0.0.1:6008",
    "http://localhost:6008",
    "https://u742891-8b98-66121702.bjb2.seetacloud.com:8443",
    "https://uu742891-8b98-66121702.bjb2.seetacloud.com:8443",
    "https://uu742891-9109-afb63769.bjb2.seetacloud.com:8443",
    "https://u742891-9109-afb63769.bjb2.seetacloud.com:8443",
]
# 允许同域名前缀变化（例如实例变更后的子域名）
cors_allow_origin_regex = r"https://[a-z0-9-]+\.bjb2\.seetacloud\.com:8443"
cors_allow_credentials = False
cors_allow_methods = ["*"]
cors_allow_headers = ["*"]

# 频率限制配置（slowapi）
# 同一IP每分钟允许请求次数
rate_limit_per_minute = 60
# 默认限流规则（按IP）
rate_limit_default_rule = f"{rate_limit_per_minute}/minute"
# 上传与检索接口限流规则（可按需拆分不同值）
rate_limit_upload_rule = rate_limit_default_rule
rate_limit_retrieve_rule = rate_limit_default_rule
rate_limit_retrieve_status_rule = rate_limit_default_rule
# 反向代理场景下，优先取这些请求头作为真实来源IP
rate_limit_ip_header_priority = ("x-forwarded-for", "x-real-ip")

# 检索算法参数
retrieve_pubchem_ppm = 1.0
retrieve_batch_size = 256
retrieve_seed = 42
retrieve_eval_mode = "full"
retrieve_missing_tree_policy = "discard"
retrieve_ema_mode = "auto"
