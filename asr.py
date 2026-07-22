# ==========================================
# asr.py · ASR 识别（支持直接模式 & 预设轴模式）
# ==========================================
# 模式说明：
#   direct     — Whisper 自行分句，适用于无参考 SRT 的场景
#   srt_preset — 以参考 SRT 时间轴为基准，逐段切片后交给 Whisper 识别
#                时间轴完全沿用参考 SRT，仅替换识别文本
import gc
import os
import re
import shutil
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

import torch
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from tqdm import tqdm

from config import (
    WHISPER_MODEL_SIZE,
    ASR_DEVICE,
    ASR_COMPUTE_TYPE,
    ASR_LANGUAGE,
    ASR_BEAM_SIZE,
    ASR_TEMPERATURE,
    ASR_WORD_TIMESTAMPS,
    ASR_LOG_PROGRESS,
    ASR_CHUNK_LENGTH,
    ASR_VAD_FILTER,
    ASR_NO_SPEECH_THRESHOLD,
    VAD_PARAMS,
    ASR_NO_REPEAT_NGRAM_SIZE,
    ASR_MODE,
    ASR_TEMP_DIR,
    ASR_MIN_CLIP_DURATION,
    ASR_FALLBACK_TO_ORIGINAL,
)
from utils import time_to_seconds

_SAMPLE_RATE = 16000

# ==========================================
# 直接模式（Whisper 自行分句）
# ==========================================


def _collect_segments(segments):
    res = []
    for s in segments:
        text = s.text.strip()
        if text:
            res.append({"start": s.start, "end": s.end, "text": text})
    return res


def _transcribe(model, audio):
    segments, _ = model.transcribe(
        audio,
        language=ASR_LANGUAGE,
        beam_size=ASR_BEAM_SIZE,

        # ===== decoding =====
        temperature=ASR_TEMPERATURE,
        word_timestamps=ASR_WORD_TIMESTAMPS,
        log_progress=ASR_LOG_PROGRESS,

        # ===== ngram =====
        no_repeat_ngram_size=ASR_NO_REPEAT_NGRAM_SIZE,

        # ===== chunk =====
        chunk_length=ASR_CHUNK_LENGTH,

        # ===== VAD =====
        vad_filter=ASR_VAD_FILTER,
        vad_parameters=VAD_PARAMS,
        no_speech_threshold=ASR_NO_SPEECH_THRESHOLD,
    )
    return _collect_segments(segments)


def _find_repeat_anchor(res):
    """若存在相邻高度相似片段，返回重复段第一句的索引；否则 None。"""
    for i in range(1, len(res)):
        prev, cur = res[i - 1], res[i]
        sim = SequenceMatcher(None, prev["text"], cur["text"]).ratio()
        if sim >= 0.9 and (cur["start"] - prev["end"]) <= 0.5:
            return i - 1
    return None


def _run_asr_direct(audio_path):
    """直接模式：对完整音频执行 Whisper ASR，返回去重后的片段列表。"""
    print("🎧 [Step 1/3] 正在 ASR 识别（直接模式）")

    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=ASR_DEVICE,
        compute_type=ASR_COMPUTE_TYPE,
    )

    res = _transcribe(model, audio_path)

    # 去重逻辑：检测到重复时，从该部分第一句开始重听一次
    if res:
        anchor = _find_repeat_anchor(res)
        if anchor is not None:
            retry_from = res[anchor]["start"]
            print(f"⚠️ 检测到重复片段，从 {retry_from:.2f}s（该段第一句）重听一次")
            kept = res[:anchor]
            audio = decode_audio(audio_path, sampling_rate=_SAMPLE_RATE)
            start_sample = int(retry_from * _SAMPLE_RATE)
            retry = _transcribe(model, audio[start_sample:])
            for seg in retry:
                seg["start"] += retry_from
                seg["end"] += retry_from
            res = kept + retry

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


# ==========================================
# 预设轴模式（基于参考 SRT 时间轴切片识别）
# ==========================================

_SRT_TIME_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


def parse_srt_blocks(path: str) -> List[dict]:
    """解析 SRT，返回 [{index, start, end, original_text}] 列表。

    start / end 保留 SRT 原始时间字符串（如 "00:00:03,120"），
    使用 time_to_seconds() 可转为秒。
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.strip().split("\n")
        # 找到包含时间轴的行
        idx = next((i for i, l in enumerate(lines) if _SRT_TIME_RE.search(l)), -1)
        if idx == -1:
            continue
        m = _SRT_TIME_RE.search(lines[idx])
        blocks.append({
            "index":         len(blocks) + 1,
            "start":         m.group(1),
            "end":           m.group(2),
            "original_text": "\n".join(lines[idx + 1:]).strip(),
        })
    return blocks


def _split_audio_block(audio_path: str, block: dict, idx: int) -> Optional[str]:
    """用 ffmpeg 从音频中切出一段 WAV 切片。

    Args:
        audio_path: 源音频路径。
        block:      SRT 解析出的段信息（含 start / end 时间字符串）。
        idx:        切片序号（用于命名输出文件）。

    Returns:
        切片文件路径；切片失败或时长过短时返回 None。
    """
    start = time_to_seconds(block["start"])
    end = time_to_seconds(block["end"])
    dur = end - start
    if dur < ASR_MIN_CLIP_DURATION:
        return None
    out = os.path.join(ASR_TEMP_DIR, f"clip_{idx:04d}.wav")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", audio_path,
        "-t", str(dur),
        "-ac", "1",
        "-ar", str(_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-avoid_negative_ts", "make_zero",
        out,
    ]
    try:
        subprocess.run(
            cmd, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        if os.path.exists(out) and os.path.getsize(out) > 2048:
            return out
    except subprocess.CalledProcessError:
        pass
    return None


def _transcribe_clip(model: WhisperModel, clip_path: str) -> str:
    """识别单个音频切片，返回纯文本。

    预设轴模式不启用 VAD 和 chunk_length（切片已按 SRT 轴精确裁剪）。
    """
    try:
        segs, _ = model.transcribe(
            clip_path,
            language=ASR_LANGUAGE,
            beam_size=ASR_BEAM_SIZE,
            temperature=ASR_TEMPERATURE,
            no_repeat_ngram_size=ASR_NO_REPEAT_NGRAM_SIZE,
        )
        return "".join(s.text for s in segs).strip()
    except Exception:
        return ""


def run_asr_from_srt(audio_path: str, srt_path: str) -> List[dict]:
    """预设轴模式：以参考 SRT 时间轴为基准，逐段切片后交给 Whisper 识别。

    时间轴完全沿用参考 SRT，仅用 Whisper 识别结果替换文本。
    若某段识别为空且 ASR_FALLBACK_TO_ORIGINAL 为 True，则回退到参考 SRT 原文。

    Args:
        audio_path: 音频文件路径。
        srt_path:   参考 SRT 文件路径。

    Returns:
        [{"start": float, "end": float, "text": str}, ...]
    """
    print("🎧 [Step 1/3] 正在 ASR 识别（预设轴模式）")
    blocks = parse_srt_blocks(srt_path)
    print(f"   📋 共 {len(blocks)} 个时间段")

    if not blocks:
        print("   ⚠️ 参考 SRT 为空，回退到直接模式")
        return _run_asr_direct(audio_path)

    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=ASR_DEVICE,
        compute_type=ASR_COMPUTE_TYPE,
    )
    print(f"   🤖 模型 {WHISPER_MODEL_SIZE} ({ASR_DEVICE}) 已加载")

    os.makedirs(ASR_TEMP_DIR, exist_ok=True)
    result: List[dict] = []
    fallback_count = 0

    for i, block in enumerate(tqdm(blocks, desc="   识别进度")):
        clip = _split_audio_block(audio_path, block, i)
        if clip:
            txt = _transcribe_clip(model, clip)
            try:
                os.remove(clip)
            except OSError:
                pass
        else:
            txt = ""

        # 识别为空时的回退处理
        if not txt and ASR_FALLBACK_TO_ORIGINAL:
            txt = block["original_text"]
            fallback_count += 1

        result.append({
            "start": time_to_seconds(block["start"]),
            "end":   time_to_seconds(block["end"]),
            "text":  txt,
        })

    # 清理临时目录
    shutil.rmtree(ASR_TEMP_DIR, ignore_errors=True)

    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if fallback_count:
        print(f"   ⚠️ {fallback_count} 段识别为空，已回退到参考 SRT 原文")
    print(f"   ✅ ASR 完成，共 {len(result)} 条")
    return result


# ==========================================
# SRT 文件查找与匹配
# ==========================================


def find_srt_for_audio(audio_path: str) -> Optional[str]:
    """在音频文件同目录下查找同名 SRT 文件。

    匹配规则：音频文件名去扩展名后 + .srt / .SRT
    例：audio/01_intro.wav → audio/01_intro.srt
    """
    stem = Path(audio_path).stem
    audio_dir = Path(audio_path).parent
    for ext in (".srt", ".SRT"):
        srt_path = audio_dir / f"{stem}{ext}"
        if srt_path.exists():
            return str(srt_path)
    return None


def match_srt_to_audio(
    audio_files: List[str],
    srt_files: Optional[List[str]] = None,
) -> dict:
    """将 SRT 文件匹配到音频文件。

    Args:
        audio_files: 音频文件路径列表。
        srt_files:   SRT 文件路径列表。为 None 时在音频同目录自动查找。

    Returns:
        {audio_path: srt_path} 映射。未匹配的音频不在字典中。
    """
    mapping = {}
    if srt_files is None:
        for audio in audio_files:
            srt = find_srt_for_audio(audio)
            if srt:
                mapping[audio] = srt
    else:
        srt_by_stem = {Path(f).stem: f for f in srt_files}
        for audio in audio_files:
            stem = Path(audio).stem
            if stem in srt_by_stem:
                mapping[audio] = srt_by_stem[stem]
    return mapping


# ==========================================
# 统一入口（调度器）
# ==========================================


def run_asr(audio_path: str, srt_path: Optional[str] = None) -> List[dict]:
    """ASR 识别入口，根据 ASR_MODE 自动调度。

    Args:
        audio_path: 音频文件路径。
        srt_path:   参考 SRT 路径（预设轴模式必需，直接模式忽略）。

    Returns:
        [{"start": float, "end": float, "text": str}, ...]
    """
    if ASR_MODE == "srt_preset" and srt_path:
        return run_asr_from_srt(audio_path, srt_path)
    elif ASR_MODE == "srt_preset" and not srt_path:
        print("⚠️ 预设轴模式但未提供参考 SRT，自动回退到直接模式")
    return _run_asr_direct(audio_path)
