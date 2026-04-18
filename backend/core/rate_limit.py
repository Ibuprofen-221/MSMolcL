from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.config import rate_limit_ip_header_priority


def get_request_client_ip(request: Request) -> str:
    """优先从代理头读取真实IP，读取失败时回退到客户端地址。"""
    for header_name in rate_limit_ip_header_priority:
        raw_value = (request.headers.get(header_name) or "").strip()
        if not raw_value:
            continue

        # X-Forwarded-For 可能是逗号分隔的IP链，取首个非空值
        first_ip = raw_value.split(",")[0].strip()
        if first_ip:
            return first_ip

    return get_remote_address(request)


limiter = Limiter(key_func=get_request_client_ip)
