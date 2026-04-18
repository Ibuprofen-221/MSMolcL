def success_response(data: dict | None = None, message: str = "success") -> dict:
    """统一成功响应。"""
    return {
        "status": "success",
        "message": message,
        "data": data or {},
    }


def error_response(message: str, code: int = 400) -> dict:
    """统一失败响应。"""
    return {
        "status": "error",
        "code": code,
        "message": message,
    }
