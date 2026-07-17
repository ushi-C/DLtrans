# ==========================================
# api_client.py · API 客户端 & Token 消耗估计
# ==========================================
from getpass import getpass
from typing import List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import OPENAI_BASE_URL, MODEL_NAME, RETRY_MAX_ATTEMPTS


# ---------- Token 估计 ----------
class TokenCounter:
    def __init__(self):
        self.total_tokens = 0

    def add(self, text: str):
        self.total_tokens += int(len(text) * 1.3)


usage_stats = TokenCounter()


# ---------- 客户端初始化 ----------
def init_openai_client() -> OpenAI:
    api_key = getpass("请输入代理 API Key: ").strip()
    return OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)


# ---------- LLM 调用（含重试）----------
@retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=2, min=4, max=15),
)
def call_llm_api(client: OpenAI, messages: List[dict], temp: float = 0.2):
    usage_stats.add(str(messages))
    response = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, temperature=temp
    )
    content = response.choices[0].message.content
    usage_stats.add(content)
    return content
