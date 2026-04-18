# 内存缓存：供后续流程复用有效对结果
processed_cache = {
    "valid_pairs_spectra_content": "",
    "valid_pairs_fragtrees_content": {},
    "statas": {},
}

# 自定义候选库内存缓存（仅在内存使用，不落盘）
custom_lib_cache: dict[str, list[str]] = {}


def update_processed_cache(spectra_content: str, fragtrees_content: dict, statas: dict) -> None:
    """更新预处理结果内存缓存。"""
    processed_cache["valid_pairs_spectra_content"] = spectra_content
    processed_cache["valid_pairs_fragtrees_content"] = fragtrees_content
    processed_cache["statas"] = statas


def set_custom_lib_cache(file_name: str, smiles_list: list[str]) -> None:
    """写入自定义候选库到内存缓存。"""
    custom_lib_cache[file_name] = smiles_list


def get_custom_lib_cache(file_name: str) -> list[str] | None:
    """从内存缓存读取自定义候选库。"""
    return custom_lib_cache.get(file_name)
