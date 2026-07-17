# ==========================================
# background.py · DLsite 音声作品背景信息
# ==========================================
# 背景 = 故事背景 + 角色名称 + 每段音频内容介绍
# 多个音频文件共用同一背景，各自拥有独立的段落描述。
#
# 段落描述推荐流程：
#   粘贴整份トラックリスト → 正则拆轨 → 与文件名对齐 → 确认后写入
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# 轨标题行：1. タイトル [01:17]  /  1．タイトル（01:17）  /  1、タイトル
_TRACK_HEADER = re.compile(
    r"^(\d+)\s*[.．、]\s*(.+?)(?:\s*[\[（(](\d{1,2}:\d{2}(?::\d{2})?)[\]）)])?\s*$"
)

# 文件名中的轨号：01_xxx / 1.xxx / track02 / trk-3
_FILE_TRACK_NUM = re.compile(
    r"(?:^|[_\-\s])(?:track|trk|曲)?[_\-\s]*(\d{1,3})(?=[_\-\s.]|$)",
    re.IGNORECASE,
)
_FILE_LEADING_NUM = re.compile(r"^(\d{1,3})")


class AudioBackground:
    """DLsite 音声作品的背景信息。

    Attributes:
        title:              作品标题。
        story:              故事背景 / 世界观设定。
        characters:         角色列表，每项为 {"name": str, "description": str}。
        track_descriptions: 各音频文件的内容描述，键为文件名。
    """

    def __init__(
        self,
        title: str = "",
        story: str = "",
        characters: Optional[List[dict]] = None,
        track_descriptions: Optional[Dict[str, str]] = None,
    ):
        self.title = title
        self.story = story
        self.characters = characters if characters is not None else []
        self.track_descriptions = track_descriptions if track_descriptions is not None else {}

    # ---------- 查询 ----------

    def get_track_description(self, filename: str) -> str:
        """获取指定音频文件的内容描述。"""
        return self.track_descriptions.get(filename, "")

    def has_track_description(self, filename: str) -> bool:
        """判断是否存在指定音频文件的内容描述。"""
        return bool(self.track_descriptions.get(filename))

    # ---------- 格式化 ----------

    def format_for_prompt(self, filename: Optional[str] = None) -> str:
        """将背景信息格式化为 LLM prompt 中的文本段落。

        Args:
            filename: 当前处理的音频文件名。传入时追加该段落的独立描述。

        Returns:
            格式化后的背景信息字符串。
        """
        parts: List[str] = []

        if self.title:
            parts.append(f"作品标题：{self.title}")

        if self.story:
            parts.append(f"故事背景：{self.story}")

        if self.characters:
            char_parts = []
            for c in self.characters:
                name = c.get("name", "")
                desc = c.get("description", "")
                if desc:
                    char_parts.append(f"{name}（{desc}）")
                else:
                    char_parts.append(name)
            parts.append(f"登场角色：{'、'.join(char_parts)}")

        if filename:
            desc = self.get_track_description(filename)
            if desc:
                parts.append(f"本段内容：{desc}")

        return "\n".join(parts) if parts else "无"

    # ---------- 交互式输入（Colab） ----------

    @staticmethod
    def input_interactive() -> "AudioBackground":
        """交互式输入共享背景信息（用于 Colab / 终端）。

        依次收集作品标题、故事背景、登场角色。
        段落级描述请在上传音频后调用 input_track_descriptions()。
        """
        print("=== DLsite 音声作品背景信息 ===")

        title = input("1. 作品标题（可留空）: ").strip()

        print("2. 故事背景 / 世界观设定（单行输入，可用 \\n 表示换行）:")
        story_raw = input().strip()
        story = story_raw.replace("\\n", "\n") if story_raw else ""

        print("3. 登场角色（格式：角色名:描述，多个角色用逗号分隔）")
        print("   示例：涼海ネモ:清楚系Vtuber, あんず:妹角色")
        print("   也可只输入角色名，用逗号分隔：涼海ネモ, あんず")
        char_input = input().strip()
        characters = AudioBackground._parse_characters(char_input)

        bg = AudioBackground(
            title=title,
            story=story,
            characters=characters,
        )
        print(f"\n✅ 背景信息已记录（角色 {len(characters)} 名）")
        return bg

    @staticmethod
    def input_track_descriptions(filenames: List[str]) -> Dict[str, str]:
        """粘贴整份トラックリスト，解析并对齐到音频文件。

        流程：粘贴列表 → 正则拆轨 → 与文件名对齐 → 打印对照表确认。
        确认失败或解析失败时可回退到逐文件手填。

        Args:
            filenames: 音频文件名列表。

        Returns:
            {filename: description} 字典。
        """
        if not filenames:
            return {}

        print(f"\n=== トラックリスト → {len(filenames)} 个音频文件 ===")
        print("请粘贴整份トラックリスト（含序号、曲名、标签）。")
        print("单独一行输入 END 结束；直接 END 可跳过。\n")

        raw = AudioBackground._read_multiline_until_end()
        if not raw.strip():
            print("⏭️ 已跳过段落描述")
            return {}

        tracks = AudioBackground.parse_tracklist(raw)
        if not tracks:
            print("⚠️ 未能解析トラックリスト，改为逐文件手填")
            return AudioBackground._input_track_descriptions_manual(filenames)

        print(f"✅ 解析到 {len(tracks)} 轨")
        mapping, warnings = AudioBackground.align_tracks_to_files(tracks, filenames)
        AudioBackground.print_track_alignment(mapping, filenames, tracks, warnings)

        confirm = input("\n对照表是否正确？[Y/n]（n = 改为逐文件手填）: ").strip().lower()
        if confirm in ("n", "no"):
            return AudioBackground._input_track_descriptions_manual(filenames)

        print(f"✅ 已记录 {len(mapping)} 个文件的描述")
        return mapping

    @staticmethod
    def apply_tracklist(text: str, filenames: List[str]) -> Dict[str, str]:
        """非交互：解析トラックリスト并对齐到文件名（供 notebook / 脚本调用）。

        Args:
            text:      整份トラックリスト原文。
            filenames: 音频文件名列表。

        Returns:
            {filename: description}。解析失败时返回空字典。
        """
        tracks = AudioBackground.parse_tracklist(text)
        if not tracks:
            return {}
        mapping, warnings = AudioBackground.align_tracks_to_files(tracks, filenames)
        AudioBackground.print_track_alignment(mapping, filenames, tracks, warnings)
        return mapping

    # ---------- 解析 / 对齐 ----------

    @staticmethod
    def parse_tracklist(text: str) -> List[dict]:
        """用正则将トラックリスト拆成结构化轨信息。

        支持格式示例：
            1. タイトル [01:17]
            タグ/タグ

            2．タイトル（36:52）
            タグ

        Returns:
            [{"index", "title", "duration", "tags", "description"}, ...]
        """
        if not text or not text.strip():
            return []

        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        headers: List[Tuple[int, int, str, str]] = []  # (line_idx, track_no, title, duration)

        for i, line in enumerate(lines):
            m = _TRACK_HEADER.match(line.strip())
            if not m:
                continue
            track_no = int(m.group(1))
            title = m.group(2).strip()
            duration = (m.group(3) or "").strip()
            headers.append((i, track_no, title, duration))

        if not headers:
            return []

        tracks: List[dict] = []
        for h_i, (line_idx, track_no, title, duration) in enumerate(headers):
            end = headers[h_i + 1][0] if h_i + 1 < len(headers) else len(lines)
            tag_lines = []
            for raw in lines[line_idx + 1 : end]:
                s = raw.strip()
                if s:
                    tag_lines.append(s)
            tags = " ".join(tag_lines)
            if tags:
                description = f"{title} / {tags}"
            else:
                description = title
            tracks.append(
                {
                    "index": track_no,
                    "title": title,
                    "duration": duration,
                    "tags": tags,
                    "description": description,
                }
            )

        tracks.sort(key=lambda t: t["index"])
        return tracks

    @staticmethod
    def align_tracks_to_files(
        tracks: List[dict],
        filenames: List[str],
    ) -> Tuple[Dict[str, str], List[str]]:
        """将解析出的轨描述对齐到音频文件名。

        优先级：
          1. 文件名中的轨号 ↔ リスト序号
          2. 数量相等时按文件名自然序与轨序号一一对应

        Returns:
            (mapping, warnings)
        """
        warnings: List[str] = []
        if not tracks or not filenames:
            return {}, ["无轨信息或无音频文件"]

        by_index = {t["index"]: t for t in tracks}
        mapping: Dict[str, str] = {}
        used_indices = set()

        # Pass 1: 文件名轨号匹配
        numbered: List[Tuple[str, int]] = []
        unnumbered: List[str] = []
        for fname in filenames:
            num = AudioBackground._extract_track_number(fname)
            if num is not None:
                numbered.append((fname, num))
            else:
                unnumbered.append(fname)

        for fname, num in numbered:
            track = by_index.get(num)
            if track:
                mapping[fname] = track["description"]
                used_indices.add(num)
            else:
                warnings.append(f"{fname}: 文件名轨号 {num} 在リスト中不存在")

        # Pass 2: 剩余文件按自然序 ↔ 剩余轨
        remaining_files = [f for f in AudioBackground._natural_sort(filenames) if f not in mapping]
        remaining_tracks = [t for t in tracks if t["index"] not in used_indices]

        if remaining_files and remaining_tracks:
            if len(remaining_files) == len(remaining_tracks):
                for fname, track in zip(remaining_files, remaining_tracks):
                    mapping[fname] = track["description"]
                    used_indices.add(track["index"])
                if unnumbered or (numbered and remaining_files):
                    warnings.append("部分文件按排序顺序与剩余轨一一对应")
            else:
                n = min(len(remaining_files), len(remaining_tracks))
                for fname, track in zip(remaining_files[:n], remaining_tracks[:n]):
                    mapping[fname] = track["description"]
                    used_indices.add(track["index"])
                warnings.append(
                    f"文件数与轨数不完全匹配"
                    f"（剩余文件 {len(remaining_files)}，剩余轨 {len(remaining_tracks)}），"
                    f"已按顺序对齐前 {n} 个"
                )

        for fname in filenames:
            if fname not in mapping:
                warnings.append(f"{fname}: 未能对齐到任何轨")

        for t in tracks:
            if t["index"] not in used_indices:
                warnings.append(f"轨 {t['index']}「{t['title']}」未对齐到任何文件")

        return mapping, warnings

    @staticmethod
    def print_track_alignment(
        mapping: Dict[str, str],
        filenames: List[str],
        tracks: List[dict],
        warnings: Optional[List[str]] = None,
    ) -> None:
        """打印文件 ↔ 轨描述对照表。"""
        print("\n--- 对齐对照表 ---")
        for i, fname in enumerate(AudioBackground._natural_sort(filenames), 1):
            desc = mapping.get(fname, "（未对齐）")
            # 截断过长描述，便于终端阅读
            shown = desc if len(desc) <= 80 else desc[:77] + "..."
            print(f"  [{i}] {fname}")
            print(f"      → {shown}")

        if len(tracks) != len(filenames):
            print(f"\n⚠️ リスト {len(tracks)} 轨 / 音频 {len(filenames)} 个，数量不一致")

        if warnings:
            print("\n注意：")
            for w in warnings:
                print(f"  · {w}")

    # ---------- 内部方法 ----------

    @staticmethod
    def _read_multiline_until_end(sentinel: str = "END") -> str:
        """读取多行输入，直到单独一行等于 sentinel。"""
        lines: List[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == sentinel:
                break
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _input_track_descriptions_manual(filenames: List[str]) -> Dict[str, str]:
        """逐文件手填描述（解析失败或用户拒绝自动对齐时的回退）。"""
        print(f"\n=== 逐文件输入内容描述（{len(filenames)} 个）===")
        print("（直接回车可跳过该文件）\n")

        track_descs: Dict[str, str] = {}
        for i, fname in enumerate(filenames, 1):
            desc = input(f"  [{i}/{len(filenames)}] {fname}: ").strip()
            if desc:
                track_descs[fname] = desc

        print(f"✅ 已记录 {len(track_descs)} 个文件的描述")
        return track_descs

    @staticmethod
    def _extract_track_number(filename: str) -> Optional[int]:
        """从文件名提取轨号。优先前缀数字，其次 track/曲 + 数字。"""
        stem = Path(filename).stem
        m = _FILE_LEADING_NUM.match(stem)
        if m:
            return int(m.group(1))
        m = _FILE_TRACK_NUM.search(stem)
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _natural_sort(filenames: List[str]) -> List[str]:
        """按文件名自然序排序（01 < 2 < 10）。"""

        def key(name: str):
            parts = re.split(r"(\d+)", name.lower())
            return [int(p) if p.isdigit() else p for p in parts]

        return sorted(filenames, key=key)

    @staticmethod
    def _parse_characters(text: str) -> List[dict]:
        """解析角色输入文本。

        支持格式：
          "涼海ネモ:清楚系, あんず:妹角色"  -> [{"name": "涼海ネモ", "description": "清楚系"}, ...]
          "涼海ネモ, あんず"                 -> [{"name": "涼海ネモ", "description": ""}, ...]
        """
        if not text:
            return []

        characters: List[dict] = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                name, desc = part.split(":", 1)
                characters.append({"name": name.strip(), "description": desc.strip()})
            else:
                characters.append({"name": part, "description": ""})

        return characters
