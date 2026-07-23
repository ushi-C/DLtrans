# ==========================================
# api_client.py · API 客户端 & Token 消耗估计 & 智能重试
# ==========================================
import time
from getpass import getpass
from typing import List, Tuple

from openai import OpenAI

from config import (
    OPENAI_BASE_URL,
    MODEL_NAME,
    RETRY_MAX_ATTEMPTS,
    RETRY_WAIT_MIN,
    RETRY_WAIT_MAX,
)


# ==========================================
# 异常定义
# ==========================================

class LLMCallError(Exception):
    """LLM 调用失败异常，携带分类后的原因信息。

    Attributes:
        reason:     人类可读的失败原因。
        retryable:  该错误是否属于可重试类型。
        attempts:   实际尝试次数。
        original:   原始异常对象。
    """

    def __init__(self, reason: str, retryable: bool, attempts: int, original: Exception = None):
        self.reason = reason
        self.retryable = retryable
        self.attempts = attempts
        self.original = original
        tag = "可重试" if retryable else "不可重试"
        super().__init__(f"[{tag}] {reason}（已尝试 {attempts} 次）")


# ==========================================
# Token 估计
# ==========================================

class TokenCounter:
    def __init__(self):
        self.total_tokens = 0

    def add(self, text: str):
        self.total_tokens += int(len(text) * 1.3)


usage_stats = TokenCounter()


# ==========================================
# 客户端初始化
# ==========================================

def init_openai_client() -> OpenAI:
    api_key = getpass("请输入代理 API Key: ").strip()
    return OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)


# ==========================================
# 错误分类
# ==========================================

# 可重试的 HTTP 状态码
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# 动态收集 openai 异常类型（兼容不同库版本）
_RETRYABLE_TYPES: list = []
_NON_RETRYABLE_TYPES: list = []

try:
    from openai import APITimeoutError, APIConnectionError
    _RETRYABLE_TYPES.extend([APITimeoutError, APIConnectionError])
except ImportError:
    pass

try:
    from openai import RateLimitError
    _RETRYABLE_TYPES.append(RateLimitError)
except ImportError:
    pass

try:
    from openai import InternalServerError
    _RETRYABLE_TYPES.append(InternalServerError)
except ImportError:
    pass

try:
    from openai import AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError
    _NON_RETRYABLE_TYPES.extend([AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError])
except ImportError:
    pass

try:
    from openai import UnprocessableEntityError
    _NON_RETRYABLE_TYPES.append(UnprocessableEntityError)
except ImportError:
    pass

try:
    from openai import APIStatusError
except ImportError:
    APIStatusError = None

try:
    from openai import APIError
except ImportError:
    APIError = None


def _classify_error(e: Exception) -> Tuple[str, bool]:
    """分析异常，返回 (原因描述, 是否可重试)。

    分类逻辑：
      1. 明确不可重试：401 认证 / 400 请求格式 / 404 模型不存在 / 403 权限 / 422 内容无法处理
      2. 明确可重试：网络超时 / 连接失败 / 429 速率限制 / 5xx 服务器错误
      3. 其他 API 错误：按状态码判断，无状态码则默认可重试
      4. 非 openai 异常（网络层等）：默认可重试
    """
    # --- 明确不可重试 ---
    for t in _NON_RETRYABLE_TYPES:
        if isinstance(e, t):
            status = getattr(e, "status_code", None) or ""
            name = type(e).__name__
            return f"{name}（HTTP {status}）: {_safe_str(e)}", False

    # --- 明确可重试 ---
    for t in _RETRYABLE_TYPES:
        if isinstance(e, t):
            status = getattr(e, "status_code", None) or ""
            name = type(e).__name__
            return f"{name}（HTTP {status}）: {_safe_str(e)}", True

    # --- APIStatusError：按状态码判断 ---
    if APIStatusError and isinstance(e, APIStatusError):
        status = getattr(e, "status_code", None)
        if status and status in _RETRYABLE_STATUS:
            return f"服务器错误（HTTP {status}）: {_safe_str(e)}", True
        return f"API 状态错误（HTTP {status}）: {_safe_str(e)}", False

    # --- 其他 APIError ---
    if APIError and isinstance(e, APIError):
        status = getattr(e, "status_code", None)
        if status and status in _RETRYABLE_STATUS:
            return f"API 错误（HTTP {status}）: {_safe_str(e)}", True
        if status:
            return f"API 错误（HTTP {status}）: {_safe_str(e)}", False
        return f"API 错误（无状态码）: {_safe_str(e)}", True

    # --- 非 openai 异常（网络层等），默认可重试 ---
    return f"{type(e).__name__}: {_safe_str(e)}", True


def _safe_str(e: Exception) -> str:
    """安全提取异常信息，避免某些异常 __str__ 抛出二次异常。"""
    try:
        s = str(e)
        return s[:200] if len(s) > 200 else s
    except Exception:
        return type(e).__name__


# ==========================================
# LLM 调用（智能重试）
# ==========================================

def call_llm_api(client: OpenAI, messages: List[dict], temp: float = 0.2) -> str:
    """调用 LLM API，带错误分类的智能重试。

    - 可重试错误（网络超时、429 速率限制、5xx 服务器错误）→ 指数退避自动重试
    - 不可重试错误（401 认证、400 请求格式、404 模型不存在等）→ 立即抛出 LLMCallError
    - 重试耗尽 → 抛出 LLMCallError

    Returns:
        模型回复文本。

    Raises:
        LLMCallError: 当不可重试错误发生，或可重试错误重试耗尽时。
    """
    usage_stats.add(str(messages))

    last_reason = ""
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, temperature=temp
            )
            content = response.choices[0].message.content
            if not content:
                # 部分模型/供应商会返回空 content，视为可重试
                raise RuntimeError("API 返回空内容（content 为空）")
            usage_stats.add(content)
            return content

        except Exception as e:
            reason, retryable = _classify_error(e)
            last_reason = reason

            # 不可重试 → 立即抛出
            if not retryable:
                raise LLMCallError(reason, retryable=False, attempts=attempt, original=e)

            # 可重试但已耗尽 → 抛出
            if attempt >= RETRY_MAX_ATTEMPTS:
                raise LLMCallError(
                    f"重试 {RETRY_MAX_ATTEMPTS} 次后仍失败: {reason}",
                    retryable=True,
                    attempts=attempt,
                    original=e,
                )

            # 可重试，等待后重试（指数退避: WAIT_MIN, WAIT_MIN*2, WAIT_MIN*4, ...）
            wait = min(RETRY_WAIT_MIN * (2 ** (attempt - 1)), RETRY_WAIT_MAX)
            print(f"   ⏳ 第 {attempt}/{RETRY_MAX_ATTEMPTS} 次重试（等待 {wait}s）: {reason}")
            time.sleep(wait)

    # 理论上不会到达
    raise LLMCallError(last_reason or "未知失败", retryable=False, attempts=RETRY_MAX_ATTEMPTS)
