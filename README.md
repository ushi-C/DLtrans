# DLsite 音声作品自动翻译

DLsite 日语音声作品 → 中日双语 SRT 字幕。推荐在 Google Colab（CUDA / T4 GPU）运行。

```
共享背景（标题 + 故事 + 角色）           ← 所有音频共用
    ↓
音频文件 ×N
    +
トラックリスト（一次粘贴，按文件名对齐） ← 每轨独立「本段内容」
    ↓
[1] ASR 识别        faster-whisper large-v3，日语
    ├─ 预设轴模式：按参考 SRT 时间轴切片 → 逐段识别（保留原始轴）
    └─ 直接模式：Whisper 自行分句，VAD 过滤 + 去重
    ↓
[2] 背景校对        依据故事/角色/本段内容，LLM 批量纠正 ASR 错误
    ↓
[3] AI 翻译         LLM 日语 → 中文，分块并发，保留人名/自造词
    ↓
输出 *_bilingual.srt（日文 + 中文双轨）× N 个文件
```

---

## 文件结构

```
DLsiteAudioTranslator/
├── config.py          # 所有可调参数 & Prompt 模板
├── utils.py           # 工具函数：时间转换、SRT 格式化、ID 解析
├── api_client.py      # OpenAI 客户端、重试器、Token 统计
├── asr.py             # Whisper ASR（预设轴模式 / 直接模式）
├── background.py      # 背景信息：共享背景 + トラックリスト解析/对齐
├── proofread.py       # LLM 辅助 ASR 校对（背景驱动）
├── translate.py       # LLM 翻译（背景驱动）
├── main.py            # 入口：多音频流水线，输出 SRT
├── run.ipynb          # Colab 笔记本版（推荐）
└── requirements.txt
```

---

## 快速开始

### 1. 环境

推荐 Google Colab（已预装 CUDA / torch）。打开 `run.ipynb`，硬件加速器选 **T4 GPU**，按格子依次运行。

本地或 Colab 安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 参数配置

编辑 `config.py`，按需调整：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `OPENAI_BASE_URL` | API 代理地址 | `https://api.chatanywhere.tech/v1` |
| `MODEL_NAME` | 使用的模型 | `gpt-5.5` |
| `MAX_WORKERS` | 并发翻译线程数 | `4` |
| `RETRY_MAX_ATTEMPTS` | API 失败重试次数 | `3` |
| `WHISPER_MODEL_SIZE` | Whisper 模型规格 | `large-v3` |
| `ASR_MODE` | ASR 模式：`srt_preset`（预设轴）/ `direct`（直接） | `srt_preset` |
| `ASR_TEMP_DIR` | 预设轴模式切片临时目录 | `_asr_clips` |
| `ASR_MIN_CLIP_DURATION` | 预设轴最小切片时长（秒） | `0.3` |
| `ASR_FALLBACK_TO_ORIGINAL` | 预设轴识别为空时回退到参考 SRT 原文 | `True` |
| `VAD_PARAMS` | VAD 静音检测参数（直接模式） | 见文件 |
| `MAX_CHARS_PER_CHUNK` | 单次翻译最大字符数 | `1500` |
| `PROOFREAD_BATCH_SIZE` | 单批校对轴数 | `100` |
| `PROOFREAD_SYSTEM_PROMPT` | 校对系统 Prompt 模板 | 见文件 |
| `PROOFREAD_USER_TEMPLATE` | 校对用户 Prompt 模板 | 见文件 |
| `TRANSLATE_SYSTEM_PROMPT` | 翻译系统 Prompt 模板 | 见文件 |
| `TRANSLATE_USER_TEMPLATE` | 翻译用户 Prompt 模板 | 见文件 |

### 3. 运行

**Colab（推荐）**：打开 `run.ipynb`，依次执行各单元格。

**脚本**：

```bash
python main.py
```

运行后依次提示：

1. 输入 API Key  
2. 输入共享背景（作品标题、故事背景、登场角色）  
3. 上传音频文件（可多选）  
4. 粘贴トラックリスト，确认对齐对照表（可跳过）  
5. 自动处理：ASR → 校对 → 翻译 → 写 SRT  

完成后下载每个音频对应的 `<音频文件名>_bilingual.srt`。

---

## ASR 模式

项目支持两种 ASR 模式，通过 `config.py` 中的 `ASR_MODE` 切换：

### 预设轴模式（`srt_preset`，默认）

以参考 SRT 文件的时间轴为基准，将音频逐段切片后交给 Whisper 识别。

- **时间轴**：完全沿用参考 SRT，不会偏移
- **文本**：用 Whisper 重新识别替换，比原 SRT 更准确
- **回退**：某段识别为空时，自动回退到参考 SRT 原文（`ASR_FALLBACK_TO_ORIGINAL`）
- **适用**：已有粗略 SRT（如其他工具生成的日文字幕），需要提高文本质量

使用方式：将音频文件和同名 SRT 文件放在一起（如 `01_intro.wav` ↔ `01_intro.srt`），程序自动匹配。

### 直接模式（`direct`）

Whisper 自行分句并生成时间轴，附带 VAD 过滤和重复检测去重。

- **适用**：没有参考 SRT，从零开始识别

---

## 背景信息说明

背景分两层：

| 层级 | 内容 | 作用范围 |
|---|---|---|
| 共享背景 | 作品标题、故事背景、登场角色 | 所有音频的校对 / 翻译 |
| 段落描述 | 当前轨的曲名 + 标签（本段内容） | 仅当前正在处理的音频 |

校对与翻译的 system prompt 都会注入：共享背景 +（若有）本段内容。

### 共享背景

- **作品标题**：音声作品名称（可选）  
- **故事背景**：世界观设定、剧情前提等  
- **登场角色**：`角色名:描述`，多个用逗号分隔；也可只写角色名  

示例：

```
作品标题: 癒やしの森の夢眠案内
故事背景: 聴者は森の奥にある小さな小屋に迷い込み、妖精に眠りを導かれる物語。
登场角色: リリス:森の妖精、語り手, ミスト:夢の案内人
```

### 段落描述：トラックリスト

上传音频后，**一次粘贴整份トラックリスト**（不要逐文件手填）。流程：

1. **正则拆轨**：按 `序号. 曲名 [时长]` 识别轨标题，后续非空行作为标签  
2. **对齐文件**：优先匹配文件名中的轨号；否则按文件名自然序与轨序号一一对应  
3. **确认对照表**：打印「文件 → 本段内容」；确认后写入，输入 `n` 可回退逐文件手填  

交互输入时，粘贴结束后单独一行输入 `END`；直接 `END` 可跳过段落描述。

#### 支持的リスト格式

```
トラックリスト
1. 物語の導入 [03:20]
リリスが聴者を出迎える

2. 眠りへの誘導 [25:10]
子守歌/吐息

3. 朝の目覚め [05:00]
終わりの挨拶
```

也支持全角标点，如 `1．タイトル（36:52）`。时长可省略。

写入 prompt 的「本段内容」形如：

```
物語の導入 / リリスが聴者を出迎える
```

#### 文件名与对齐

| 规则 | 说明 |
|---|---|
| 优先 | 文件名带轨号：`01_intro.wav`、`track02.mp3`、`3_xxx.m4a` |
| 回退 | 轨数 = 文件数时，按自然序（`01` < `2` < `10`）一一对应 |
| 建议 | 上传前把文件名改成带序号，避免错轨 |

#### Notebook 非交互写法

不必走 `input()`，可直接赋值：

```python
TRACKLIST = """
1. 物語の導入 [03:20]
リリスが聴者を出迎える

2. 眠りへの誘導 [25:10]
子守歌/吐息
"""
track_descs = AudioBackground.apply_tracklist(TRACKLIST, audio_files)
background.track_descriptions = track_descs
```

---

## 输入格式

**音频**：faster-whisper 支持的格式（mp3、m4a、wav、mp4、flac 等），一次可上传多个文件。

**トラックリスト**：纯文本，见上文格式。

---

## 输出格式

标准 SRT，每条字幕为日文原文 + 中文译文双行：

```
1
00:00:03,120 --> 00:00:05,880
今日も来てくれてありがとう！
今天也谢谢你能来！

2
00:00:06,200 --> 00:00:09,440
...
```

每个音频文件生成一个独立的 SRT：`<原文件名去扩展名>_bilingual.srt`。

---

## 依赖

| 库 | 用途 |
|---|---|
| `faster-whisper` | ASR 识别 |
| `tqdm` | 进度条 |
| `openai` | LLM 调用 |
| `tenacity` | API 重试 |
| `torch` | GPU 推理后端 |

---

## 注意

**本项目代码由 AI 辅助生成。**

生成的 SRT 主要用于提高校对效率，不保证零人工修改。不懂日语请谨慎使用。

1. 需要能访问 DLsite 的网络（获取作品与トラックリスト）。  
2. 需要 Google Colab 账号（有 Google 账号即可）。  
3. 需要大语言模型 API，示例使用 [GPT_API_free](https://github.com/chatanywhere/GPT_API_free)。  
   - Token 消耗受字幕条数与背景信息长度影响。  
   - 其他 API 请修改 `config.py` 中的 `OPENAI_BASE_URL`。  
4. Colab 免费 GPU 时长有限，单轨超过约 2 小时请慎重。  
