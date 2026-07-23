# ==========================================
# config.py · 全局配置（所有配置参数 & Prompt 模板）
# ==========================================
# 参数前置原则：所有可调参数集中在此文件，其他模块仅引用不定义。

# --- API 配置 ---
OPENAI_BASE_URL = "https://llm-api.net/v1/chat/completions"
MODEL_NAME = "gemini-2.5-pro"

# --- 并发与重试 ---
MAX_WORKERS = 4            # 并发翻译线程数
RETRY_MAX_ATTEMPTS = 3     # API 失败重试次数

# ==========================================
# ASR 配置
# ==========================================

WHISPER_MODEL_SIZE = "large-v3"

# ----- mode -----
# "direct"     = 直接识别（Whisper 自行分句，适用于无参考 SRT 的场景）
# "srt_preset" = 预设轴模式（以参考 SRT 时间轴切片后逐段识别，保留原始轴）
ASR_MODE = "srt_preset"

# ----- device -----
ASR_DEVICE = "cuda"
ASR_COMPUTE_TYPE = "float16"

# ----- decoding -----
ASR_LANGUAGE = "ja"
ASR_BEAM_SIZE = 5
ASR_TEMPERATURE = [0.0, 0.2, 0.4, 0.6]
ASR_WORD_TIMESTAMPS = False
ASR_LOG_PROGRESS = False

# ----- chunk -----
ASR_CHUNK_LENGTH = 30

# ----- ngram -----
ASR_NO_REPEAT_NGRAM_SIZE = 6

# ----- VAD -----
ASR_VAD_FILTER = True
ASR_NO_SPEECH_THRESHOLD = 0.55

VAD_PARAMS = {
    "threshold": 0.16,
    "max_speech_duration_s": 30,
    "min_speech_duration_ms": 500,
    "min_silence_duration_ms": 800,
    "speech_pad_ms": 200,
}

# ----- SRT 预设轴模式专用 -----
ASR_TEMP_DIR = "_asr_clips"          # 音频切片临时目录
ASR_MIN_CLIP_DURATION = 0.3          # 最小切片时长（秒），低于此值跳过识别
ASR_FALLBACK_TO_ORIGINAL = True      # ASR 识别为空时，回退到参考 SRT 原文

# ==========================================
# 校对 & 翻译配置
# ==========================================

MAX_CHARS_PER_CHUNK = 1500  # 翻译分块最大字符数
PROOFREAD_BATCH_SIZE = 100  # 单批校对轴数

# ==========================================
# Prompt 模板（校对 & 翻译共用）
# ==========================================
# {background} 占位符由 background.AudioBackground.format_for_prompt() 填充。

# --- 校对 Prompt ---
PROOFREAD_SYSTEM_PROMPT = (
    "执行日语 ASR 文本校对任务。你是专业的日语语音识别校对专家，"
    "擅长 DLsite 音声作品的文本修正。\n\n"
    "请根据以下背景信息，修正 ASR 识别结果中的错误：\n"
    "【背景信息】\n{background}\n\n"
    "校对规则：\n"
    "1. 依据角色名称修正语音识别中的人名错误（最优先）。\n"
    "2. 依据故事背景修正专有名词、地名、术语等错误。\n"
    "3. 依据本段内容介绍理解音频主题，辅助判断语义。\n"
    "4. 保留 [Sxxxxx] 标签格式，逐行输出校对结果。\n"
    "5. 无需修改的行原样返回，不要遗漏任何一行。\n"
    "6. 禁止输出任何解释说明。"
)

PROOFREAD_USER_TEMPLATE = "[待校对ASR]\n{asr_text}"

# --- 翻译 Prompt ---
TRANSLATE_SYSTEM_PROMPT = (
    "执行字幕翻译任务：将日语翻译为中文。"
    "你是专业的日中翻译专家，擅长 DLsite 音声作品的翻译。\n\n"
    "请根据以下背景信息进行翻译：\n"
    "【背景信息】\n{background}\n\n"
    "翻译规则：\n"
    "1. 根据背景信息和角色名称确保翻译准确性和一致性。\n"
    "2. 角色名称统一翻译，首次出现时标注日语原文。\n"
    "3. 人名和自造词保留日语原文。\n"
    "4. 保持原文的语气和情感色彩。\n"
    "5. 必须严格保持并输出所有 [Sxxxxx] ID 标签。"
)

TRANSLATE_USER_TEMPLATE = (
    "请逐行将日语翻译为中文。根据上下文语境纠正突兀之处，"
    "人名和自造词保留日语原文，必须严格保持并输出所有 ID。\n"
    "格式：[ID] 中文翻译\n\n{input_block}"
)
