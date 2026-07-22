# ==========================================
# utils.py · 自建工具函数
# ==========================================
import re


def time_to_seconds(t):
    """将时间字符串或数值统一转换为秒（float）。

    兼容 SRT 时间戳格式（逗号分隔毫秒，如 00:00:03,120）。
    """
    if isinstance(t, (int, float)):
        return float(t)
    # SRT 使用逗号分隔毫秒，统一替换为小数点
    t = str(t).strip().replace(",", ".")
    parts = t.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except Exception:
        return 0.0


def format_srt_time(seconds):
    """将秒数格式化为 SRT 时间戳字符串 HH:MM:SS,mmm。"""
    ms = int(abs(seconds % 1) * 1000)
    full_sec = int(abs(seconds))
    m, s = divmod(full_sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def extract_mapping(content: str) -> dict:
    """从 LLM 返回文本中解析 {SID: 译文} 映射。
    支持 S00001、[S00001]、(S00001) 等格式。
    """
    mapping = {}
    for line in content.splitlines():
        # 匹配 S00001, [S00001], (S00001) 等
        m = re.search(r"(S\d+)", line)
        if m:
            sid = m.group(1)
            # 提取 ID 后面的内容：分割 ID 及其后的符号
            text_part = re.split(r"S\d+[\s\]:：]*", line, maxsplit=1)[-1]
            mapping[sid] = text_part.strip()
    return mapping
