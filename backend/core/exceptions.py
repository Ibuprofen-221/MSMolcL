class FileMissingError(BaseException):
    """上传文件缺失异常。"""


class FileFormatError(BaseException):
    """上传文件格式异常。"""


class FileSizeError(BaseException):
    """上传文件大小超限异常。"""


class ProcessingError(BaseException):
    """文件处理异常。"""
