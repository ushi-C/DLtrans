# ==========================================
# main.py · 入口执行
# ==========================================
# DLsite 音声作品自动化翻译：
#   多音频文件 → 共用背景 → 各自 ASR / 校对 / 翻译 → 双语 SRT
from pathlib import Path

from api_client import init_openai_client, usage_stats
from asr import run_asr, match_srt_to_audio
from background import AudioBackground
from config import ASR_MODE
from proofread import run_smart_proofread
from translate import run_parallel_translation
from utils import format_srt_time

# 支持的音频扩展名
_AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".mp4", ".flac", ".aac", ".ogg"}
_SRT_EXTS = {".srt", ".SRT"}


def process_single_audio(client, audio_path, background, filename, srt_path=None):
    """处理单个音频文件：ASR → 校对 → 翻译 → 写 SRT。

    Args:
        client:    OpenAI 客户端实例。
        audio_path: 音频文件路径。
        background: 共享的背景信息。
        filename:  音频文件名（用于提取段落描述）。
        srt_path:  参考 SRT 路径（预设轴模式使用，直接模式忽略）。

    Returns:
        生成的 SRT 文件名。
    """
    print(f"\n{'='*50}")
    print(f"🎬 处理: {filename}")
    if srt_path:
        print(f"   📝 参考 SRT: {Path(srt_path).name}")
    print(f"{'='*50}")

    # Step 1: ASR
    raw_asr = run_asr(audio_path, srt_path)
    if not raw_asr:
        print(f"⚠️ {filename} 未识别到任何内容，跳过")
        return None

    # Step 2: 校对（背景辅助）
    proofed_data = run_smart_proofread(client, raw_asr, background, filename)

    # Step 3: 翻译
    final_data = run_parallel_translation(client, proofed_data, background, filename)

    # Step 4: 写 SRT
    srt_file = f"{Path(audio_path).stem}_bilingual.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, s in enumerate(final_data, 1):
            f.write(
                f"{i}\n"
                f"{format_srt_time(s['start'])} --> {format_srt_time(s['end'])}\n"
                f"{s['ja']}\n{s['zh']}\n\n"
            )

    print(f"📄 已生成: {srt_file}")
    return srt_file


def _print_srt_match_table(audio_files, srt_mapping):
    """打印音频 → SRT 匹配对照表。"""
    print(f"\n--- SRT 匹配对照表（模式: {ASR_MODE}）---")
    for i, audio_path in enumerate(audio_files, 1):
        fname = Path(audio_path).name
        srt = srt_mapping.get(audio_path)
        if srt:
            print(f"  [{i}] {fname}")
            print(f"      → {Path(srt).name}")
        else:
            tag = "⚠️ 无匹配 SRT" if ASR_MODE == "srt_preset" else "（直接模式，无需 SRT）"
            print(f"  [{i}] {fname}  {tag}")
    print()


def main():
    # 延迟导入，避免非 Colab 环境下模块级导入失败
    from google.colab import files

    client = init_openai_client()

    # --- 背景信息（所有音频共用） ---
    background = AudioBackground.input_interactive()

    # --- 上传文件（音频 + SRT，支持多个）---
    print("\n--- 请上传文件（音频 + 参考 SRT，可多选）---")
    if ASR_MODE == "srt_preset":
        print("   ℹ️ 预设轴模式：请同时上传音频文件和对应的参考 SRT 文件")
        print("   ℹ️ SRT 文件名需与音频文件名一致（如 01_intro.wav ↔ 01_intro.srt）")
    uploaded = files.upload()

    # 按扩展名分离音频文件和 SRT 文件
    audio_files = []
    srt_files = []
    for fname in uploaded.keys():
        ext = Path(fname).suffix
        if ext in _AUDIO_EXTS:
            audio_files.append(fname)
        elif ext in _SRT_EXTS:
            srt_files.append(fname)

    if not audio_files:
        print("⚠️ 未上传任何音频文件，程序结束")
        return

    print(f"\n✅ 共接收 {len(audio_files)} 个音频文件"
          + (f"，{len(srt_files)} 个 SRT 文件" if srt_files else ""))

    # 匹配 SRT 到音频
    if srt_files:
        srt_mapping = match_srt_to_audio(audio_files, srt_files)
    else:
        srt_mapping = {}

    _print_srt_match_table(audio_files, srt_mapping)

    # --- 粘贴トラックリスト → 解析对齐到各音频 ---
    track_descs = AudioBackground.input_track_descriptions(audio_files)
    background.track_descriptions = track_descs

    # --- 逐个处理 ---
    srt_files_out = []
    for audio_path in audio_files:
        filename = Path(audio_path).name
        srt_path = srt_mapping.get(audio_path)
        out_file = process_single_audio(
            client, audio_path, background, filename, srt_path
        )
        if out_file:
            srt_files_out.append(out_file)

    # --- 汇总 ---
    print(f"\n{'='*50}")
    print(f"✅ 全部完成！共生成 {len(srt_files_out)} 个 SRT 文件")
    print(f"   Token 消耗估算: {usage_stats.total_tokens}")
    print(f"{'='*50}")

    # 逐个下载
    for srt_file in srt_files_out:
        files.download(srt_file)


if __name__ == "__main__":
    main()
