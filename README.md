# 🎵 Lyrical-Aligner

> **自动化歌词生成工具** — 输入任意语言的音频，自动识别并生成带毫秒级时间戳的 `.lrc` 歌词文件，支持翻译为指定语言输出

---

## 项目目录结构

```
Lyrical-Aligner/
│
├── pipeline.py              ← 主入口：端到端流水线
├── config.py                ← 全局配置（路径、模型参数、后处理阈值）
├── download_models.py       ← 模型下载 / 本地加载脚本
│
├── vocal_extractor.py       ← Step 1 人声分离（Demucs）
├── transcription_engine.py  ← Step 2 语音识别（faster-whisper）
├── postprocessor.py         ← Step 3 ASR后处理（规则修正）
├── translator.py            ← Step 4 多语言翻译（可选）
├── lrc_generator.py         ← Step 5 LRC文件生成
│
├── requirements.txt         ← Python依赖清单
│
├── input/                   ← 放置原始音频文件
├── output/                  ← 生成的 .lrc 与中间 JSON 文件
├── models/                  ← faster-whisper 模型缓存目录
└── temp/                    ← Demucs 中间音频临时目录
```

---

## 技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 人声分离 | [Demucs (htdemucs)](https://github.com/facebookresearch/demucs) | Facebook Research 混合 Transformer 模型，同时输出纯人声与伴奏 |
| 语音识别 | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2 后端，支持 99 种语言自动识别，速度约为原版 4× |
| 歌词翻译 | [deep-translator](https://github.com/nidhaloff/deep-translator) | 识别后翻译为任意目标语言；封装 Google / DeepL / 离线 argostranslate |
| LRC 生成 | 内置 | 输出标准 LRC 或词级 Enhanced LRC（A2）格式 |
| 硬件加速 | CUDA (NVIDIA GPU) | 自动回退到 CPU |
| 语言 | Python 3.10+ | |

---

## 快速开始

### 1. 环境搭建（推荐使用 Conda）

> **为什么推荐 Conda？**  
> Demucs 和 faster-whisper 对 Python 版本与 CUDA 工具链较敏感，Conda 可以同时管理
> Python、CUDA 运行库（`cudatoolkit`）和系统级依赖，避免环境冲突。

#### 1-A. 安装 Conda

如果尚未安装，请下载 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（推荐）
或 [Anaconda](https://www.anaconda.com/download)，安装后重启终端。

```bash
# 验证安装
conda --version
```

#### 1-B. 创建并激活专用虚拟环境

```bash
# 创建名为 lyrical-aligner 的环境，指定 Python 3.10
conda create -n lyrical-aligner python=3.10 -y

# 激活环境（后续所有命令均在此环境下执行）
conda activate lyrical-aligner
```

> 💡 退出环境使用 `conda deactivate`；
> 彻底删除环境使用 `conda env remove -n lyrical-aligner`。

#### 1-C. 安装 ffmpeg（通过 Conda 管理，无需手动配置 PATH）

```bash
conda install -c conda-forge ffmpeg -y
```

#### 1-D. 安装 PyTorch（根据 CUDA 版本选择）

```bash
# --- CUDA 12.1（推荐，适配 RTX 30/40 系列）---
conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# --- CUDA 11.8（适配 RTX 20/30 系列旧驱动）---
# conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# --- CPU only（无 GPU 环境）---
# conda install pytorch torchaudio cpuonly -c pytorch -y
```

验证 GPU 是否可用：

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### 1-E. 安装项目依赖

```bash
# 确保在项目根目录下
cd Lyrical-Aligner

pip install -r requirements.txt
```

### 2. 下载 / 验证模型

```bash
# 下载默认模型（large-v3 + htdemucs）
python download_models.py

# 指定模型大小
python download_models.py --model medium

# 从本地路径加载 Whisper 模型（离线环境）
python download_models.py --local D:/models/faster-whisper-large-v3
```

### 3. 配置 `config.py`

运行前打开 `config.py`，根据需要修改对应属性即可。以下是最常用的两个场景示例：

**场景 A — 中文歌曲，直接识别，输出中文 LRC**
```python
# config.py
WHISPER_LANGUAGE        = "zh"          # 强制中文，识别更快
```

**场景 B — 日文歌曲，自动识别，翻译为中文**
```python
# config.py
TRANSLATION_TARGET_LANG = "zh"          # 翻译目标语言
TRANSLATION_BACKEND     = "google"      # 免费在线翻译
LRC_ARTIST              = "歌手名"
```

**场景 C — 高质量离线处理（无网络）**
```python
# config.py
WHISPER_MODEL_SIZE      = "large-v3"
DEMUCS_MODEL            = "htdemucs_ft" # 更高质量人声分离
WHISPER_DEVICE          = "cuda"
WHISPER_LOCAL_MODEL_PATH = r"D:\models\faster-whisper-large-v3"
TRANSLATION_BACKEND     = "argos"       # 完全离线翻译
```

**场景 D — 已有纯人声文件，CPU 运行**
```python
# config.py
SKIP_SEPARATION         = True          # 跳过 Demucs
WHISPER_MODEL_SIZE      = "small"
WHISPER_DEVICE          = "cpu"
WHISPER_COMPUTE_TYPE    = "int8"
```

**场景 E — 只需要人声 / 伴奏，不生成 LRC**
```bash
# 直接调用人声分离模块，输出 vocals.wav + accompaniment.wav
python vocal_extractor.py input/song.mp3

# 高质量模型（htdemucs_ft 人声残留更少）
python vocal_extractor.py input/song.mp3 --model htdemucs_ft
```

### 4. 运行流水线

CLI 只需传入音频文件路径，支持一次传入多个文件批量处理：

```bash
# 单个文件
python pipeline.py input/song.mp3

# 批量处理
python pipeline.py input/song1.mp3 input/song2.wav input/song3.flac
```

输出示例：
```
output/
├── song.lrc              ← 生成的歌词文件（含毫秒级时间戳）
└── song_segments.json    ← 中间段落数据（SAVE_INTERMEDIATES = True 时生成）
```

---

## 各模块独立使用

### 人声分离 / 伴奏提取

Demucs 在分离人声的同时会同时输出伴奏轨，两个文件保存在 `temp/demucs/` 下：

```
temp/demucs/
├── song_vocals.wav        ← 纯人声
└── song_accompaniment.wav ← 伴奏（消人声）
```

只需要人声或伴奏、不需要生成 LRC 时，可以直接单独调用 `vocal_extractor.py`：

```bash
# 基本用法（输出人声 + 伴奏）
python vocal_extractor.py input/song.mp3

# 使用更高质量模型（速度较慢）
python vocal_extractor.py input/song.mp3 --model htdemucs_ft

# CPU 运行（无 GPU 环境）
python vocal_extractor.py input/song.mp3 --device cpu
```

> 💡 **伴奏用途**：消人声后的伴奏轨可直接用于卡拉OK/KTV 练唱，效果与许多商业伴奏制作方式相同。  
> 人声轨则可用于二次创作、混音或单独的语音分析。

### 语音识别

```bash
python transcription_engine.py temp/demucs/song_vocals.wav
python transcription_engine.py vocals.wav --language ja --out-json segments.json
```

### 后处理

```bash
python postprocessor.py segments.json --out cleaned.json --merge-gap 0.4
```

### 翻译

```bash
# 翻译为中文（Google，免费）
python translator.py segments.json --target zh --out translated.json

# 翻译为英语（DeepL）
python translator.py segments.json --target en --backend deepl

# 离线翻译（argostranslate）
python translator.py segments.json --target ja --backend argos

# 预先下载 argos 语言包（首次使用前需执行一次）
python translator.py --install-argos en zh
```

### LRC 生成

```bash
python lrc_generator.py cleaned.json --out song.lrc --title "My Song" --by-word
```

---

## 评测管线 (`eval/`)

内置四维度评测框架，按 testset JSON 驱动，各阶段有对应 ground truth 才运行，没有则自动跳过。

| 维度 | 指标 | 参考标准 |
|------|------|---------|
| 人声分离 | SDR（dB） | > 10 dB 为良好 |
| 语音识别 | WER / CER | 越低越好 |
| 时间戳精度 | MAE（秒）/ Acc@±0.3s | MAE ≤ 0.3 s 为合格 |
| 翻译质量 | BLEU / chrF | 中文等 CJK 语言以 chrF 为主参考 |

```bash
# 使用示例 testset 运行（跳过人声分离阶段，复用已有 vocals）
python eval/eval_pipeline.py --testset eval/testset_example.json --skip-separation

# 保存评测报告
python eval/eval_pipeline.py --testset eval/testset_example.json --output eval/report.json
```

**testset 格式**（`eval/testset_example.json`）：每条记录对应一首歌，可选提供 `vocals_ref`（人声参考，用于 SDR）、`segments_ref`（转录参考，用于 WER/MAE）、`translation_ref`（翻译参考，用于 BLEU/chrF）。

---

## 配置说明 (`config.py`)

> 所有运行行为均通过修改 `config.py` 控制，**无需任何命令行参数**。

### 模型设置

| 参数 | 默认山 | 说明 |
|------|--------|------|
| `DEMUCS_MODEL` | `htdemucs` | Demucs 模型；`htdemucs_ft` 质量更高但更慢 |
| `DEMUCS_SEGMENT` | `7.8` | 分块长度（秒），VRAM 不足时调小 |
| `DEMUCS_DEVICE` | `cuda` | `cuda` 或 `cpu` |
| `WHISPER_MODEL_SIZE` | `large-v3` | Whisper 模型大小 |
| `WHISPER_DEVICE` | `cuda` | `cuda` 或 `cpu` |
| `WHISPER_COMPUTE_TYPE` | `float16` | GPU:`float16` / CPU:`int8` |
| `WHISPER_LOCAL_MODEL_PATH` | `None` | 本地模型路径（设置后不从网络下载） |

### ASR 识别设置

| 参数 | 默认山 | 说明 |
|------|--------|------|
| `WHISPER_LANGUAGE` | `None` | `None` = 自动检测；强制指定如 `"zh"` |
| `WHISPER_BEAM_SIZE` | `5` | Beam-search 宽度，越大越准确也越慢 |
| `WHISPER_VAD_FILTER` | `True` | Silero VAD 过滤，减少幻觉输出 |

### LRC 输出设置

| 参数 | 默认山 | 说明 |
|------|--------|------|
| `LRC_TITLE` | `""` | `[ti:]` 标签（空则自动使用文件名） |
| `LRC_ARTIST` | `""` | `[ar:]` 标签 |
| `LRC_ALBUM` | `""` | `[al:]` 标签 |
| `LRC_OFFSET` | `0` | 全局时间偏移（毫秒） |
| `LRC_BY_WORD` | `False` | `True` = Enhanced LRC 词级时间戳格式 |

### 流水线行为设置

| 参数 | 默认山 | 说明 |
|------|--------|------|
| `SKIP_SEPARATION` | `False` | `True` = 跳过 Demucs（输入已是纯人声） |
| `SAVE_INTERMEDIATES` | `True` | 是否保存 `_segments.json` 中间文件 |

### 翻译设置

| 参数 | 默认山 | 说明 |
|------|--------|------|
| `TRANSLATION_TARGET_LANG` | `None` | 目标语言（ISO-639-1），不设则不翻译 |
| `TRANSLATION_BACKEND` | `google` | `google` / `deepl` / `argos` |
| `TRANSLATION_SOURCE_LANG` | `auto` | 源语言，`auto` = 复用 ASR 检测结果 |
| `TRANSLATION_BATCH_DELAY` | `0.5` | API 调用间隔（秒），防止限流 |
| `DEEPL_API_KEY` | `""` | DeepL 密钥（推荐用环境变量 `DEEPL_API_KEY` 设置） |

### 后处理设置

| 参数 | 默认山 | 说明 |
|------|--------|------|
| `PP_MERGE_GAP_THRESHOLD` | `0.30` | 合并间距 ≤ 此山（秒）的相邻段落 |
| `PP_MAX_CHARS_PER_LINE` | `50` | 每行 LRC 的最大字符数 |
| `PP_MIN_SEGMENT_DURATION` | `0.3` | 丢弃时长 < 此山（秒）的段落 |

---

## 🌐 指定语言歌词输出

工具的核心能力之一：识别任意语言的歌曲，将歌词翻译后以目标语言生成 LRC，无需事先找到对应的歌词文本。全部翻译选项均在 `config.py` 中配置，当检测语言与 `TRANSLATION_TARGET_LANG` 相同时，翻译步骤会被自动跳过。

### 支持的翻译后端

| 后端 | `TRANSLATION_BACKEND` 山 | 优点 | 要求 |
|------|---------|------|------|
| **Google Translate** | `"google"` | 免费，无需 API Key | 需要网络 |
| **DeepL** | `"deepl"` | 翻译质量高 | DeepL API Key |
| **Argostranslate** | `"argos"` | 完全离线 | 首次下载语言包 |

### 配置示例

**Google 免费翻译（默认）**
```python
# config.py
TRANSLATION_TARGET_LANG = "zh"      # 翻译为简体中文
TRANSLATION_BACKEND     = "google"  # 免费，无需 Key
```

**DeepL 高质量翻译**
```python
# config.py
TRANSLATION_TARGET_LANG = "en"
TRANSLATION_BACKEND     = "deepl"
DEEPL_API_KEY           = "your-deepl-api-key"  # 或设置环境变量 DEEPL_API_KEY
```

**Argostranslate 离线翻译**
```python
# config.py
TRANSLATION_TARGET_LANG = "zh"
TRANSLATION_BACKEND     = "argos"
```
首次使用前预先下载语言包（仅需执行一次）：
```bash
python translator.py --install-argos ja zh   # 例：日语 → 中文
```

**禁用翻译**
```python
# config.py
TRANSLATION_TARGET_LANG = None   # 默认山，不翻译
```

### 翻译后 LRC 行为说明

> 翻译后词级时间戳会被自动清除（因为翻译后的单词不再与原始音频对齐）。
> 段落级的 `start` / `end` 时间戳完整保留，LRC 文件准确性不受影响。
> 开启 `LRC_BY_WORD = True` 时，翻译内容将自动回退为段落级时间戳格式。

---

## LRC 格式示例

### 标准（段落级）

```lrc
[ti:My Song]
[ar:Artist Name]
[al:]
[offset:0]
[by:Lyrical-Aligner]

[00:12.34]第一句歌词
[00:16.78]第二句歌词
[00:21.05]第三句歌词
```

### 增强（词级 Enhanced LRC / A2）

```lrc
[00:12.34]<00:12.34>第一 <00:13.10>句 <00:13.80>歌词
[00:16.78]<00:16.78>第二 <00:17.50>句 <00:18.20>歌词
```

---

## 后处理规则说明

`postprocessor.py` 按以下顺序执行：

1. **短段过滤** — 丢弃时长 < 0.3 s 或空文本的段落
2. **幻觉清除** — 移除 "Thanks for watching" 等 Whisper 常见幻觉
3. **文本清洗** — 去除首尾乱序标点，合并多余空格
4. **重复词消除** — `"love love you"` → `"love you"`
5. **段落合并** — 间距 < 0.3 s 且合并后不超字符上限的相邻段落合为一行
6. **长句切分** — 超过字符上限的段落按词边界拆分

---

## 硬件建议

| 配置 | 推荐模型 | 预计速度（3分钟歌曲） |
|------|----------|----------------------|
| GPU ≥ 8 GB VRAM | `large-v3` + `htdemucs` | ~60 s |
| GPU 4–8 GB VRAM | `medium` + `htdemucs` | ~90 s |
| CPU only | `small` + `htdemucs` | ~5–10 min |

> 💡 **VRAM 不足时**：在 `config.py` 中将 `DEMUCS_SEGMENT` 调低至 `4.0`，
> 并将 `WHISPER_COMPUTE_TYPE` 改为 `"int8_float16"`。

---

## Conda 环境管理速查

| 操作 | 命令 |
|------|------|
| 创建环境 | `conda create -n lyrical-aligner python=3.10 -y` |
| 激活环境 | `conda activate lyrical-aligner` |
| 退出环境 | `conda deactivate` |
| 查看所有环境 | `conda env list` |
| 导出环境配置 | `conda env export > environment.yml` |
| 从配置文件还原 | `conda env create -f environment.yml` |
| 删除环境 | `conda env remove -n lyrical-aligner` |
| 更新 Conda 自身 | `conda update -n base -c defaults conda -y` |

> ⚠️ **Windows 用户注意**：请在 **Anaconda Prompt** 或已启用 conda 的 PowerShell /
> CMD 中运行上述命令。若 PowerShell 提示 `conda` 未识别，执行一次
> `conda init powershell` 后重启终端即可。

---

## License

MIT
