# ==========================================
# asr.py · ASR 识别
# ==========================================
import gc
from difflib import SequenceMatcher

import torch
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

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
)

_SAMPLE_RATE = 16000


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


def run_asr(audio_path):
    """对音频文件执行 Whisper ASR，返回去重后的片段列表。"""
    print("🎧 [Step 1/3] 正在 ASR 识别")

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
