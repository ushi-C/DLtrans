# ==========================================
# proofread.py · 背景辅助 ASR 校对
# ==========================================
# 依据 DLsite 音声作品背景信息（故事背景 / 角色名称 / 段落介绍），
# 批量调用 LLM 对 ASR 结果进行智能校对。
from typing import List

from openai import OpenAI

from api_client import call_llm_api, LLMCallError
from utils import extract_mapping
from config import (
    PROOFREAD_BATCH_SIZE,
    PROOFREAD_SYSTEM_PROMPT,
    PROOFREAD_USER_TEMPLATE,
)
from background import AudioBackground


def run_smart_proofread(
    client: OpenAI,
    asr_data: List[dict],
    background: AudioBackground,
    filename: str = "",
) -> List[dict]:
    """依据背景信息，批量调用 LLM 对 ASR 结果进行智能校对。

    Args:
        client:   OpenAI 客户端实例。
        asr_data: ASR 识别结果，每项为 {"start", "end", "text"}。
        background: 音声作品背景信息（故事背景、角色名称等）。
        filename:  当前音频文件名，用于提取该段落的独立描述。

    Returns:
        校对后的片段列表，每项为 {"start", "end", "ja"}。
    """
    print("📡 [Step 2/3] 正在执行智能校对...")

    bg_text = background.format_for_prompt(filename)
    system_prompt = PROOFREAD_SYSTEM_PROMPT.format(background=bg_text)

    final: List[dict] = []
    total = len(asr_data)
    matched_count = 0

    for i in range(0, total, PROOFREAD_BATCH_SIZE):
        batch = asr_data[i: i + PROOFREAD_BATCH_SIZE]

        asr_in = "\n".join(
            [f"[S{i + idx + 1:05d}] {s['text']}" for idx, s in enumerate(batch)]
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": PROOFREAD_USER_TEMPLATE.format(asr_text=asr_in)},
        ]

        batch_num = i // PROOFREAD_BATCH_SIZE + 1
        try:
            content = call_llm_api(client, messages)
            mapping = extract_mapping(content)
            for idx, s in enumerate(batch):
                tid = f"S{i + idx + 1:05d}"
                res_text = mapping.get(tid, s["text"])
                if res_text != s["text"]:
                    matched_count += 1
                final.append({"start": s["start"], "end": s["end"], "ja": res_text})
        except LLMCallError as e:
            print(f"   ⚠️ 批次 {batch_num} 校对失败（{e.attempts}次）: {e.reason}，该批保留原文")
            for s in batch:
                final.append({"start": s["start"], "end": s["end"], "ja": s["text"]})
        except Exception as e:
            print(f"   ⚠️ 批次 {batch_num} 未知错误: {e}，该批保留原文")
            for s in batch:
                final.append({"start": s["start"], "end": s["end"], "ja": s["text"]})

    print(f"✅ 校对完成，共订正 {matched_count} 处。")
    return final
