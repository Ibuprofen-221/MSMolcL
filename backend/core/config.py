import os
from pathlib import Path

# ── Upload constraints ──────────────────────────────────────────────
file_size_limit = 10 * 1024 * 1024       # 10 MB per file
upload_max_total_files = 50              # max total files per upload

# ── Server ──────────────────────────────────────────────────────────
server_ip = os.getenv("SERVER_IP", "127.0.0.1")
backend_host = os.getenv("BACKEND_HOST", "0.0.0.0")
backend_port = int(os.getenv("BACKEND_PORT", "6008"))
backend_reload = os.getenv("BACKEND_RELOAD", "false").lower() == "true"

# ── Directory layout (convention: web/ and autodl-tmp/ are siblings) ─
backend_dir = Path(__file__).resolve().parents[1]
web_dir = backend_dir.parent
autodl_tmp_dir = web_dir.parent / "autodl-tmp"

temp_dir = backend_dir / "temp"
user_data_root = backend_dir / "user_data"
user_data_dir = user_data_root        # legacy alias

# ── Auth & database ─────────────────────────────────────────────────
database_url = os.getenv("DATABASE_URL", f"sqlite:///{backend_dir / 'users.db'}")
jwt_secret_key = os.getenv("JWT_SECRET_KEY", "please-change-this-secret-in-production")
jwt_algorithm = "HS256"
jwt_access_token_expire_minutes = 60 * 24

# ── Task storage ────────────────────────────────────────────────────
task_storage_roots = {
    "normal": user_data_dir,
    "advanced": user_data_dir,
}
history_records_path = user_data_dir / "history_records.json"
smiles_image_dir = backend_dir / "smiles_image"

# ── Allowed upload extensions ──────────────────────────────────────
allowed_mgf_extensions = {"mgf", "txt"}
allowed_json_extensions = {"json"}

# ── Preprocess output paths ────────────────────────────────────────
valid_pairs_spectra_path = temp_dir / "valid_pairs_spectra.mgf"
valid_pairs_fragtrees_path = temp_dir / "valid_pairs_fragtrees.json"
statas_path = temp_dir / "statas.json"

# ── Docs ────────────────────────────────────────────────────────────
docs_source_path = backend_dir / "services" / "文档说明.md"
docs_structured_cache_path = temp_dir / "docs_content.json"

# ── Retrieve: model weights & databases ─────────────────────────────
retrieve_model_weight_path_pos = autodl_tmp_dir / "model_epoch-36_vloss-0.1429.pth"
retrieve_model_weight_path_neg = autodl_tmp_dir / "model_epoch-43_vloss-0.2149.pth"
retrieve_model_weight_path = retrieve_model_weight_path_pos   # legacy default
retrieve_model_weight_paths = {
    "pos": retrieve_model_weight_path_pos,
    "neg": retrieve_model_weight_path_neg,
}
retrieve_pubchem_parquet_path = autodl_tmp_dir / "pubchem_final_with_formula.parquet"
retrieve_database_root = autodl_tmp_dir / "database"
retrieve_shared_smiles_txt_path = backend_dir / "testdata" / "common.txt"

# ── Advanced-retrieve model weights ─────────────────────────────────
retrieve_advanced_model_weight_path_pos = autodl_tmp_dir / "ft_e8_loss1.0037.pth"
retrieve_advanced_model_weight_path_neg = autodl_tmp_dir / "ft_e5_loss1.0547.pth"
retrieve_advanced_model_weight_paths = {
    "pos": retrieve_advanced_model_weight_path_pos,
    "neg": retrieve_advanced_model_weight_path_neg,
}

# ── Retrieve runtime parameters ─────────────────────────────────────
retrieve_workers = 1
retrieve_max_pending_jobs = 512
retrieve_task_keep_seconds = 3600
retrieve_mp_start_method = "spawn"
retrieve_queue_wait_timeout = 1

retrieve_advanced_max_pending_jobs = 256
retrieve_advanced_task_keep_seconds = 3600
retrieve_advanced_mp_start_method = "spawn"
retrieve_advanced_queue_wait_timeout = 0.2

# ── CORS ────────────────────────────────────────────────────────────
# Set CORS_ALLOW_ORIGINS as a comma-separated list, e.g.:
#   export CORS_ALLOW_ORIGINS="http://localhost:6006,https://your.domain.com"
_cors_env = os.getenv("CORS_ALLOW_ORIGINS", "")
cors_allow_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else [
        "http://127.0.0.1:6006",
        "http://localhost:6006",
        "http://127.0.0.1:6008",
        "http://localhost:6008",
    ]
)
cors_allow_origin_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX", "")
cors_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
cors_allow_methods = ["*"]
cors_allow_headers = ["*"]

# ── Rate limiting (slowapi) ────────────────────────────────────────
rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
rate_limit_default_rule = f"{rate_limit_per_minute}/minute"
rate_limit_upload_rule = rate_limit_default_rule
rate_limit_retrieve_rule = rate_limit_default_rule
rate_limit_retrieve_status_rule = rate_limit_default_rule
rate_limit_ip_header_priority = ("x-forwarded-for", "x-real-ip")

# ── Sirius worker queue (mgf-only async json generation) ───────────
sirius_celery_broker_url = os.getenv("SIRIUS_CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")
sirius_celery_result_backend = os.getenv("SIRIUS_CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/1")
sirius_celery_task_name = os.getenv("SIRIUS_CELERY_TASK_NAME", "tasks.process_mgf_task")
sirius_queue_fast = os.getenv("SIRIUS_QUEUE_FAST", "Queue_Fast")
sirius_queue_slow = os.getenv("SIRIUS_QUEUE_SLOW", "Queue_Slow")
sirius_fast_threshold = int(os.getenv("SIRIUS_FAST_THRESHOLD", "50"))

# ── Retrieve algorithm parameters ───────────────────────────────────
retrieve_pubchem_ppm = 1.0
retrieve_batch_size = 256
retrieve_seed = 42
retrieve_eval_mode = "full"
retrieve_missing_tree_policy = "discard"
retrieve_ema_mode = "auto"
