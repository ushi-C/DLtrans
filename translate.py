# ==========================================
# translate.py · 翻译
# ==========================================
# 依据背景信息（故事背景 / 角色名称）进行日中翻译，
# 分块并发，失败自动重试。
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from api_client import call_llm_api
from utils import extract_mapping
from config import (
    MAX_WORKERS,
    MAX_CHARS_PER_CHUNK,
    TRANSLATE_SYSTEM_PROMPT,
    TRANSLATE_USER_TEMPLATE,
)
from background import AudioBackground


def _translate_worker(
    client: OpenAI,
    chunk: List[Tuple[str, str]],
    idx: int,
    total: int,
    system_prompt: str,
) -> Dict[str, str]:
    """翻译单个分块。

    Args:
        client:       OpenAI 客户端实例。
        chunk:        [(sid, text), ...] 待翻译的条目列表。
        idx:          当前分块序号（1-based）。
        total:        总分块数。
        system_prompt: 已格式化的系统 prompt（含背景信息）。

    Returns:
        {sid: 中文译文} 映射。
    """
    input_block = "\n".join([f"[{sid}] {txt}" for sid, txt in chunk])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": TRANSLATE_USER_TEMPLATE.format(input_block=input_block)},
    ]
    try:
        content = call_llm_api(client, messages)
        mapping = extract_mapping(content)

        # 检测解析失败的 ID，逐条单独重试
        missing = [(sid, txt) for sid, txt in chunk if sid not in mapping]
        if missing:
            print(f"   ⚠️ chunk {idx}/{total}: {len(missing)} 条解析失败，逐条重试...")
            for sid, txt in missing:
                try:
                    single = call_llm_api(client, [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"只输出中文译文，不要任何其他内容：{txt}"},
                    ])
                    mapping[sid] = single.strip()
                except Exception:
                    mapping[sid] = txt  # 保留日语原文

        return mapping

    except Exception as e:
        print(f"   ❌ chunk {idx}/{total} 整体失败: {e}，保留原文")
        return {sid: txt for sid, txt in chunk}  # 整体失败时保留原文，不返回空字典


def run_parallel_translation(
    client: OpenAI,
    segments: List[dict],
    background: AudioBackground = None,
    filename: str = "",
) -> List[dict]:
    """并发翻译校对后的日语文本。

    Args:
        client:    OpenAI 客户端实例。
        segments:  校对后的片段列表，每项含 {"start", "end", "ja"}。
        background: 音声作品背景信息（故事背景、角色名称等），用于提供翻译上下文。
        filename:  当前音频文件名，用于提取该段落的独立描述。

    Returns:
        添加了 "zh" 字段的片段列表。
    """
    print(f"🚀 [Step 3/3] 启动并发翻译 (并发: {MAX_WORKERS})...")

    # 预格式化系统 prompt（含背景信息），避免每个 worker 重复计算
    if background is not None:
        bg_text = background.format_for_prompt(filename)
        system_prompt = TRANSLATE_SYSTEM_PROMPT.format(background=bg_text)
    else:
        system_prompt = TRANSLATE_SYSTEM_PROMPT.format(background="无")

    items = [(f"S{i + 1:05d}", s["ja"]) for i, s in enumerate(segments)]

    chunks, cur_chunk, cur_len = [], [], 0
    for sid, txt in items:
        line = f"[{sid}] {txt}"
        if cur_chunk and cur_len + len(line) > MAX_CHARS_PER_CHUNK:
            chunks.append(cur_chunk)
            cur_chunk, cur_len = [], 0
        cur_chunk.append((sid, txt))
        cur_len += len(line)
    if cur_chunk:
        chunks.append(cur_chunk)

    all_zh: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_translate_worker, client, c, i + 1, len(chunks), system_prompt): i
            for i, c in enumerate(chunks)
        }
        for f in as_completed(futures):
            all_zh.update(f.result())

    # 仍缺失的 ID 用日语原文填充
    failed = 0
    for i, s in enumerate(segments):
        sid = f"S{i + 1:05d}"
        s["zh"] = all_zh.get(sid) or s["ja"]
        if not all_zh.get(sid):
            failed += 1

    if failed:
        print(f"   ⚠️ 最终仍有 {failed} 条未翻译，已用日语原文填充")
    else:
        print("   ✅ 全部翻译完成")
    return segments
