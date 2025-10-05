import argparse
import gradio as gr
import json
import os
import sys
import hashlib
import time
import math
import re
from typing import Any, List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from diffsinger_utau.voice_bank import PredAll
from diffsinger_utau.voice_bank.commons.ds_reader import DSReader
from diffsinger_utau.voice_bank.commons.phome_num_counter import Phome
from pypinyin import pinyin, Style
from pypinyin.constants import RE_HANS

# —— 文本预处理：相邻纯汉字不加空格，其余保留空格 ——
def _is_hans_token(s: str) -> bool:
    try:
        return bool(RE_HANS and RE_HANS.fullmatch(s))
    except Exception:
        return False

def preprocess_zh_spaces(text: str) -> str:
    parts = [p for p in (text or "").split(" ") if p != ""]
    if not parts:
        return ""
    out = []
    for i, part in enumerate(parts):
        if i == 0:
            out.append(part)
        else:
            prev = parts[i - 1]
            if _is_hans_token(prev) and _is_hans_token(part):
                out[-1] = out[-1] + part
            else:
                out.append(" " + part)
    return "".join(out)


def validate_lyric_format(modified_text: str, original_text: str) -> Tuple[bool, str]:
    """
    校验歌词格式是否与原始文本匹配
    返回: (是否匹配, 渲染后的原始文本或空字符串)
    """
    if not original_text:
        return True, ""
    
    # 去掉空格后比较
    modified_clean = re.sub(r'\s+', '', modified_text)
    original_clean = re.sub(r'\s+', '', original_text)
    
    # 长度检查
    if len(modified_clean) != len(original_clean):
        return False, render_original_with_highlights(original_text, modified_text)
    
    # AP/SP 位置检查
    modified_ap_sp_positions = []
    original_ap_sp_positions = []
    
    # 找到修改后文本中的 AP/SP 位置
    for match in re.finditer(r'\b(AP|SP)\b', modified_text):
        modified_ap_sp_positions.append((match.start(), match.group()))
    
    # 找到原始文本中的 AP/SP 位置  
    for match in re.finditer(r'\b(AP|SP)\b', original_text):
        original_ap_sp_positions.append((match.start(), match.group()))
    
    # 比较 AP/SP 的数量和类型
    if len(modified_ap_sp_positions) != len(original_ap_sp_positions):
        return False, render_original_with_highlights(original_text, modified_text)
    
    # 检查每个 AP/SP 的相对位置是否一致
    for (mod_pos, mod_type), (orig_pos, orig_type) in zip(modified_ap_sp_positions, original_ap_sp_positions):
        if mod_type != orig_type:
            return False, render_original_with_highlights(original_text, modified_text)
        
        # 计算相对位置（在去空格后的字符串中）
        mod_relative_pos = len(re.sub(r'\s+', '', modified_text[:mod_pos]))
        orig_relative_pos = len(re.sub(r'\s+', '', original_text[:orig_pos]))
        
        if mod_relative_pos != orig_relative_pos:
            return False, render_original_with_highlights(original_text, modified_text)
    
    return True, ""


def render_original_with_highlights(original_text: str, modified_text: str) -> str:
    """
    渲染原始文本，用灰色字体显示，位置不一致的 AP/SP 用红色标记
    """
    # 找到修改后和原始文本中的 AP/SP 位置
    modified_ap_sp = set()
    original_ap_sp = set()
    
    for match in re.finditer(r'\b(AP|SP)\b', modified_text):
        pos = len(re.sub(r'\s+', '', modified_text[:match.start()]))
        modified_ap_sp.add((pos, match.group()))
    
    result_parts = []
    i = 0
    clean_pos = 0
    
    while i < len(original_text):
        # 检查当前位置是否是 AP 或 SP
        if original_text[i:i+2] in ['AP', 'SP'] and (i == 0 or not original_text[i-1].isalnum()) and (i+2 >= len(original_text) or not original_text[i+2].isalnum()):
            ap_sp = original_text[i:i+2]
            # 检查这个 AP/SP 在修改后的文本中是否在相同位置
            if (clean_pos, ap_sp) not in modified_ap_sp:
                result_parts.append(f'<span style="color: red;">{ap_sp}</span>')
            else:
                result_parts.append(ap_sp)
            i += 2
            clean_pos += 2
        elif original_text[i].isspace():
            result_parts.append(original_text[i])
            i += 1
        else:
            result_parts.append(original_text[i])
            i += 1
            clean_pos += 1
    
    return f'<span style="color: gray;">{"".join(result_parts)}</span>'

# 试图导入 diffsinger-utau（按要求使用该库，而非自行实现）
try:
    import diffsinger_utau  # 类型: 忽略
except Exception as e:
    diffsinger_utau = None

ROOT = Path(__file__).parent.resolve()
MODELS_DIR = ROOT / "models"
PUBLIC_TEMPLATES_DIR = ROOT / "templates" / "public"
USER_TEMPLATES_DIR = ROOT / "templates" / "user"
OUTPUT_DIR = ROOT / "output" / "pred_all"
CACHE_DIR = ROOT / "cache"

AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
# 预创建最大可编辑句子数，避免在事件中动态创建组件
MAX_LINES = 200


def ensure_dirs():
    for p in [MODELS_DIR, PUBLIC_TEMPLATES_DIR, USER_TEMPLATES_DIR, OUTPUT_DIR, CACHE_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def list_model_choices() -> List[str]:
    # 模型以目录名为选择项；也允许单文件模型
    choices = []
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.iterdir()):
            if p.is_dir():
                choices.append(str(p.relative_to(ROOT)))
            elif p.is_file():
                # 单文件权重
                choices.append(str(p.relative_to(ROOT)))
    return choices


def find_templates() -> Dict[str, Path]:
    """
    返回 {模板名（不含扩展名）: 模板路径}
    用户目录覆盖公开目录
    """
    results: Dict[str, Path] = {}
    # 先公开
    if PUBLIC_TEMPLATES_DIR.exists():
        for p in PUBLIC_TEMPLATES_DIR.glob("*.ds"):
            results[p.stem] = p
    # 后用户（覆盖）
    if USER_TEMPLATES_DIR.exists():
        for p in USER_TEMPLATES_DIR.glob("*.ds"):
            results[p.stem] = p
    return results


def bgm_path_for(template_path: Path) -> Optional[Path]:
    base = template_path.with_suffix("")
    for ext in AUDIO_EXTS:
        cand = Path(str(base) + ext)
        if cand.exists():
            return cand
    return None


def load_ds(template_path: Path):
    ds = DSReader(template_path).read_ds()
    return ds

def audiosegment_from_file(path: Path) -> AudioSegment:
    return AudioSegment.from_file(str(path))


def export_wav(seg: AudioSegment, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    seg.export(str(path), format="wav")


def overlay_bgm_snippet(vocal_wav: Path, bgm_audio: AudioSegment, offset_sec: float, bgm_volume: float = 1.0, vocal_gain_db: float = 0.0):
    vocal = audiosegment_from_file(vocal_wav)
    if vocal_gain_db != 0.0:
        vocal = vocal + vocal_gain_db
    start_ms = max(int(offset_sec * 1000), 0)
    # 应用 BGM 音量倍率
    if bgm_volume <= 0.0:
        base = AudioSegment.silent(duration=start_ms + len(vocal))
    else:
        gain_db = 20.0 * math.log10(bgm_volume)
        bgm_audio = bgm_audio + gain_db
        base = bgm_audio
    # 确保底轨足够长
    if len(base) < start_ms + len(vocal):
        pad_ms = start_ms + len(vocal) - len(base)
        base = base + AudioSegment.silent(duration=pad_ms)
    mixed = base.overlay(vocal, position=start_ms)
    return mixed[start_ms : start_ms + len(vocal)]


def concat_with_offsets(clips: List[Tuple[AudioSegment, float]]) -> AudioSegment:
    # 根据 offset 将多个片段放置在时间线上，输出包含至最大结束时间
    if not clips:
        return AudioSegment.silent(duration=0)
    max_end_ms = 0
    for seg, offset in clips:
        start = int(max(offset, 0) * 1000)
        max_end_ms = max(max_end_ms, start + len(seg))
    timeline = AudioSegment.silent(duration=max_end_ms)
    for seg, offset in clips:
        start = int(max(offset, 0) * 1000)
        timeline = timeline.overlay(seg, position=start)
    return timeline


def mix_full_song(vocal: AudioSegment, bgm: AudioSegment, bgm_volume: float = 1.0) -> AudioSegment:
    # 保证两者同长度
    if len(bgm) < len(vocal):
        bgm = bgm + AudioSegment.silent(duration=(len(vocal) - len(bgm)))
    else:
        vocal = vocal + AudioSegment.silent(duration=(len(bgm) - len(vocal)))
    # 应用 BGM 音量倍率
    if bgm_volume <= 0.0:
        bgm_adj = AudioSegment.silent(duration=len(bgm))
    else:
        gain_db = 20.0 * math.log10(bgm_volume)
        bgm_adj = bgm + gain_db
    return bgm_adj.overlay(vocal)


def param_hash(model_sel: str, speaker: str, key_shift: int, steps: int, text: str) -> str:
    s = json.dumps(
        {
            "model": model_sel,
            "speaker": speaker or "",
            "key_shift": int(key_shift),
            "steps": int(steps),
            "text": text or "",
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


class DiffSingerEngine:
    def __init__(self):
        self.impl = diffsinger_utau
        self.entry = None
        if self.impl is not None:
            # 尝试发现可用入口函数（不同版本可能不同）
            candidates = ["synthesize", "synth", "infer", "generate", "tts"]
            for name in candidates:
                fn = getattr(self.impl, name, None)
                if callable(fn):
                    self.entry = fn
                    break

    def is_ready(self) -> bool:
        return self.impl is not None and self.entry is not None

    def synth_once(
        self,
        model_path: Path,
        text: str,
        speaker: Optional[str],
        key_shift: int,
        steps: int,
        out_wav: Path,
    ) -> None:
        """
        使用 diffsinger-utau 渲染单句音频到 out_wav。
        如果不同版本签名不同，将尝试多种参数形式。
        """
        if not self.is_ready():
            raise RuntimeError(
                "未找到可用的 diffsinger-utau 推理入口。请确认已安装并与 torch==1.13.1 兼容。"
            )

        out_wav.parent.mkdir(parents=True, exist_ok=True)

        tried = []

        def call_or_record(fn, kwargs):
            tried.append({"fn": fn.__name__, "kwargs": list(kwargs.keys())})
            return fn(**kwargs)

        # 常见签名尝试
        errors = []
        for kwargs in [
            dict(model=str(model_path), text=text, speaker=speaker, key_shift=key_shift, steps=steps, out=str(out_wav)),
            dict(model=str(model_path), text=text, speaker=speaker, key_shift=key_shift, acoustic_steps=steps, out=str(out_wav)),
            dict(model_path=str(model_path), text=text, speaker=speaker, key_shift=key_shift, steps=steps, output=str(out_wav)),
            dict(model=str(model_path), text=text, key_shift=key_shift, steps=steps, out=str(out_wav)),
            dict(model=str(model_path), text=text, out=str(out_wav)),
        ]:
            # 移除 None
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            try:
                ret = call_or_record(self.entry, kwargs)
                # 若函数返回波形和采样率，也直接落盘
                if ret is not None and isinstance(ret, (tuple, list)) and len(ret) >= 2:
                    wav, sr = ret[0], ret[1]
                    import soundfile as sf  # 懒加载
                    sf.write(str(out_wav), np.asarray(wav, dtype=np.float32), int(sr))
                # 若 out_wav 成功生成，结束
                if out_wav.exists() and out_wav.stat().st_size > 0:
                    return
            except Exception as e:
                errors.append(f"{e}")

        raise RuntimeError(
            "调用 diffsinger-utau 失败。已尝试多种签名："
            + json.dumps(tried, ensure_ascii=False)
            + f"；错误示例：{errors[-1] if errors else '未知'}"
        )


engine = DiffSingerEngine()

class DSUEngine:
    """
    基于 voice_bank.PredAll 的推理引擎；若不可用则回退到 DiffSingerEngine。
    """
    def __init__(self, old_engine: DiffSingerEngine):
        self.old = old_engine
        self.available = PredAll is not None and DSReader is not None
        self.predictors: Dict[str, PredAll] = {}  # model_path -> PredAll 实例

    def is_ready(self) -> bool:
        return self.available or self.old.is_ready()

    def _get_predictor(self, model_path: Path):
        key = str(model_path.resolve())
        if key not in self.predictors:
            self.predictors[key] = PredAll(Path(key))
        return self.predictors[key]

    def synth_line(
        self,
        model_path: Path,
        template_path: Path,
        line_index: int,
        text: str,
        speaker: Optional[str],
        key_shift: int,
        steps: int,
        out_wav: Path,
    ) -> None:
        if self.available:
            predictor = self._get_predictor(model_path)
            # 读取 ds，并替换目标行文本与必要的音素
            ds_list = DSReader(template_path).read_ds()
            if not (0 <= line_index < len(ds_list)):
                raise IndexError("行索引越界")
            ds = ds_list[line_index]
            ds.replace(text)

            # 选择说话人
            spk = speaker
            try:
                if (not spk) and getattr(predictor, "available_speakers", None):
                    av = predictor.available_speakers
                    if isinstance(av, (list, tuple)) and len(av) > 0:
                        spk = av[0]
            except Exception:
                pass

            out_wav.parent.mkdir(parents=True, exist_ok=True)
            results = predictor.predict_full_pipeline(
                ds=ds,
                lang="zh",
                speaker=spk,
                key_shift=int(key_shift),
                pitch_steps=10,
                variance_steps=10,
                acoustic_steps=int(steps),
                gender=0.0,
                output_dir=str(out_wav.parent),
                save_intermediate=False,
            )
            # 拷贝/重命名输出为指定文件名
            audio_path = results.get("audio_path") if isinstance(results, dict) else None
            if not audio_path:
                raise RuntimeError("predict_full_pipeline 未返回 audio_path")
            src = Path(audio_path)
            if src.resolve() != out_wav.resolve():
                if src.exists():
                    src.replace(out_wav)
            if not out_wav.exists() or out_wav.stat().st_size == 0:
                raise RuntimeError("未能生成音频文件")
        else:
            # 回退旧引擎（不依赖 ds）
            self.old.synth_once(model_path, text, speaker, key_shift, steps, out_wav)

# 用 DSUEngine 覆盖默认引擎
engine = DSUEngine(engine)


def get_template_choices_and_bgm_visible():
    mapping = find_templates()
    names = sorted(mapping.keys())
    # 根据当前选择动态决定 BGM 开关可见与否，默认False由前端逻辑控制
    return names


def on_select_template(template_name: str):
    mapping = find_templates()
    if not template_name or template_name not in mapping:
        return gr.update(visible=False, value=0.0), [], []
    p = mapping[template_name]
    bgm = bgm_path_for(p)
    ds = load_ds(p)
    lines = [preprocess_zh_spaces(item.get("text", "")) for item in ds]
    offsets = [float(item.get("offset", 0.0)) for item in ds]
    bgm_update = gr.update(visible=(bgm is not None), value=(1.0 if bgm is not None else 0.0))
    return bgm_update, lines, offsets


def render_single_line(
    model_sel: str,
    template_name: str,
    line_index: int,
    new_text: str,
    speaker: str,
    key_shift: int,
    steps: int,
    bgm_volume: float,
) -> Tuple[Optional[str], Optional[str]]:
    """
    渲染单句，返回：(音频路径, 错误消息)
    """
    try:
        if not model_sel:
            return None, "请先选择模型"
        mapping = find_templates()
        if template_name not in mapping:
            return None, "未找到模板"
        template_path = mapping[template_name]
        ds = load_ds(template_path)
        if not (0 <= line_index < len(ds)):
            return None, "行索引越界"
        # 更新该句的文本（仅用于本次渲染；不写回文件）
        text = new_text.strip()
        if not text:
            return None, "文本为空"

        # 缓存键
        h = param_hash(model_sel, speaker, key_shift, steps, text)
        cache_dir = OUTPUT_DIR / template_name / h
        cache_dir.mkdir(parents=True, exist_ok=True)
        wav_out = cache_dir / f"line_{line_index+1}.wav"

        if not wav_out.exists():
            # 执行推理
            model_path = ROOT / model_sel
            engine.synth_line(
                model_path=model_path,
                template_path=template_path,
                line_index=line_index,
                text=text,
                speaker=speaker or None,
                key_shift=int(key_shift),
                steps=int(steps),
                out_wav=wav_out,
            )

        # 是否混音预览
        mapping_bgm = bgm_path_for(template_path)
        if (bgm_volume and bgm_volume > 0.0) and mapping_bgm and mapping_bgm.exists():
            bgm_audio = audiosegment_from_file(mapping_bgm)
            offset = float(ds[line_index].get("offset", 0.0))
            mixed_seg = overlay_bgm_snippet(wav_out, bgm_audio, offset, bgm_volume)
            preview_wav = cache_dir / f"line_{line_index+1}_preview.wav"
            export_wav(mixed_seg, preview_wav)
            return str(preview_wav), None

        return str(wav_out), None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def generate_full_song(
    model_sel: str,
    template_name: str,
    lines: List[str],
    speaker: str,
    key_shift: int,
    steps: int,
    bgm_volume: float,
) -> Tuple[Optional[str], Optional[str]]:
    """
    生成整曲，返回：(主输出音频路径, 混音输出音频路径或错误消息字符串)
    """
    try:
        if not model_sel:
            return None, "请先选择模型"
        mapping = find_templates()
        if template_name not in mapping:
            return None, "未找到模板"
        template_path = mapping[template_name]
        ds = load_ds(template_path)
        if len(ds) != len(lines):
            return None, "模板行数与编辑行数不一致"

        # 对每句生成或复用缓存
        segs_with_offsets: List[Tuple[AudioSegment, float]] = []
        model_path = ROOT / model_sel
        for idx, item in enumerate(ds):
            text = (lines[idx] or "").strip()
            if not text:
                continue
            h = param_hash(model_sel, speaker, key_shift, steps, text)
            cache_dir = OUTPUT_DIR / template_name / h
            cache_dir.mkdir(parents=True, exist_ok=True)
            wav_out = cache_dir / f"line_{idx+1}.wav"
            if not wav_out.exists():
                engine.synth_line(
                    model_path=model_path,
                    template_path=template_path,
                    line_index=idx,
                    text=text,
                    speaker=speaker or None,
                    key_shift=int(key_shift),
                    steps=int(steps),
                    out_wav=wav_out,
                )
            segs_with_offsets.append((audiosegment_from_file(wav_out), float(item.get("offset", 0.0))))

        # 时间线拼接为全曲人声
        vocal_full = concat_with_offsets(segs_with_offsets)
        final_dir = OUTPUT_DIR / template_name / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        ts_tag = time.strftime("%Y%m%d_%H%M%S")
        vocal_path = final_dir / f"{template_name}_vocal_{ts_tag}.wav"
        export_wav(vocal_full, vocal_path)

        # 混音版本（如有 BGM 且开启开关）
        mixed_path = None
        bgm_p = bgm_path_for(template_path)
        if (bgm_volume and bgm_volume > 0.0) and bgm_p and bgm_p.exists():
            bgm_audio = audiosegment_from_file(bgm_p)
            mixed = mix_full_song(vocal_full, bgm_audio, bgm_volume)
            mixed_path = final_dir / f"{template_name}_mixed_{ts_tag}.wav"
            export_wav(mixed, mixed_path)

        return str(vocal_path), (str(mixed_path) if mixed_path else None)

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def save_uploaded_template(file_obj) -> str:
    """
    将用户上传的 ds 模板保存到 templates/user，并返回模板名。
    """
    try:
        if file_obj is None:
            raise ValueError("未选择文件")
        src = Path(file_obj.name)
        if src.suffix.lower() != ".ds":
            raise ValueError("仅支持 .ds 模板文件")
        USER_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        dst = USER_TEMPLATES_DIR / src.name
        # 读取并简单校验
        data = json.loads(Path(file_obj.name).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("ds 文件格式错误：顶层必须是 list")
        dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return src.stem
    except Exception as e:
        return f"ERROR::{type(e).__name__}: {e}"


def build_ui():
    ensure_dirs()
    model_choices = list_model_choices()
    template_names = get_template_choices_and_bgm_visible()

    css = """
    /* 全局：启用页面整体滚动，移除左右分栏独立滚动 */
    #main-row { gap: 12px; }
    #left-panel, #right-panel {
        padding: 12px;
    }
    #left-panel { border-right: 1px solid #eee; }

    /* 响应式：窄屏下上下布局，宽屏左右布局 */
    @media (max-width: 900px) {
        #main-row { flex-direction: column !important; }
        #left-panel { border-right: none; border-bottom: 1px solid #eee; }
    }

    /* 紧凑按钮样式 */
    .compact-btn button { padding: 4px 10px !important; min-height: 30px !important; height: 30px !important; }
    .compact-row { gap: 8px !important; }
    """

    with gr.Blocks(title="DiffSinger WebUI", theme=gr.themes.Soft(), css=css, head='<meta name="description" content="项目地址 https://github.com/bingcheng1998/diffsinger-webui">') as demo:
        with gr.Row(elem_id="main-row"):
            # 左栏：控制/预览（固定）
            with gr.Column(elem_id="left-panel", scale=1, min_width=360):
                # 左栏标题与模型/模板选择、上传/下载
                gr.Markdown("## DiffSinger WebUI")
                gr.Markdown("项目地址： [https://github.com/bingcheng1998/diffsinger-webui](https://github.com/bingcheng1998/diffsinger-webui)")
                model_sel = gr.Dropdown(choices=model_choices, label="模型选择", value=(model_choices[0] if model_choices else None))
                template_sel = gr.Dropdown(choices=template_names, label="模板选择", value=(template_names[0] if template_names else None))
                with gr.Row(elem_classes=["compact-row"]):
                    upload = gr.UploadButton("上传ds模板", file_types=[".ds"], elem_classes=["compact-btn"])
                    download_btn = gr.DownloadButton(label="下载当前ds状态", elem_classes=["compact-btn"])

                with gr.Row():
                    bgm_volume = gr.Slider(0.0, 2.0, value=0.3, step=0.01, label="BGM音量", visible=False)
                    key_shift = gr.Slider(-12, 12, value=0, step=1, label="音高偏移")
                    steps = gr.Slider(1, 50, value=4, step=1, label="渲染步数")
                speaker = gr.Dropdown(label="演唱者", choices=[], value=None, interactive=True)

                # 单句预览与错误提示
                per_line_audio = gr.Audio(label="单句预览", autoplay=True, interactive=False)
                per_line_error = gr.Markdown("", visible=False)

                # 生成控制与输出
                gen_btn = gr.Button("生成整首")
                progress_md = gr.Markdown("", visible=True)
                full_vocal = gr.Audio(label="整首（人声）", autoplay=False)
                full_mixed = gr.Audio(label="整首（混音）", autoplay=False, visible=False)

            # 右栏：模板与歌词编辑
            with gr.Column(elem_id="right-panel", scale=2):
                # 状态与歌词编辑容器（右栏仅歌词编辑）
                lines_state = gr.State([])
                offsets_state = gr.State([])
                # 生成控制状态
                stop_flag = gr.State(False)
                generating_flag = gr.State(False)
                dyn = gr.Column()

                # 预创建文本框和错误提示，依据当前模板设置初始可见性和值
                textboxes = []
                error_markdowns = []
                original_lines_state = gr.State([])  # 存储原始歌词用于校验
                init_lines = []
                init_offsets = []
                if template_sel.value:
                    try:
                        _, init_lines, init_offsets = on_select_template(template_sel.value)
                    except Exception:
                        init_lines, init_offsets = [], []
                with dyn:
                    for i in range(MAX_LINES):
                        visible = i < len(init_lines)
                        val = init_lines[i] if visible else ""
                        tb = gr.Textbox(value=val, label=f"第 {i+1} 句", lines=1, max_lines=1, visible=visible)
                        textboxes.append(tb)
                        # 为每个文本框添加对应的错误提示
                        error_md = gr.Markdown("", visible=False)
                        error_markdowns.append(error_md)
                if template_sel.value:
                    lines_state.value = init_lines
                    offsets_state.value = init_offsets
                    original_lines_state.value = init_lines.copy()

        # 事件：选择模板时，更新 BGM 开关、整首混音可见性与文本框内容
        def on_template_change(template_name):
            bgm_update, lines, offsets = on_select_template(template_name)
            has_bgm = bool(bgm_update.get("visible", False)) if isinstance(bgm_update, dict) else False
            tb_updates = []
            error_updates = []
            n = len(lines)
            for i, tb in enumerate(textboxes):
                if i < n:
                    tb_updates.append(gr.update(value=lines[i], visible=True))
                    error_updates.append(gr.update(value="", visible=False))
                else:
                    tb_updates.append(gr.update(value="", visible=False))
                    error_updates.append(gr.update(value="", visible=False))
            # 返回：BGM、模板下拉、状态、错误清空、整首混音可见性（按是否存在BGM），原始歌词状态，以及所有文本框和错误提示更新
            return (
                bgm_update,
                gr.update(choices=get_template_choices_and_bgm_visible(), value=template_name),
                lines,
                offsets,
                gr.update(value="", visible=False),
                gr.update(visible=has_bgm),
                lines.copy(),
                *tb_updates,
                *error_updates,
            )

        # 模型切换：动态更新 speaker 下拉项
        def on_model_change(model_path_rel):
            try:
                if not model_path_rel:
                    return gr.update(choices=[], value=None, interactive=False)
                model_path = ROOT / model_path_rel
                choices = []
                if getattr(engine, "available", False) and model_path.exists():
                    predictor = engine._get_predictor(model_path)
                    av = getattr(predictor, "available_speakers", None)
                    if isinstance(av, (list, tuple)):
                        choices = list(av)
                if choices:
                    return gr.update(choices=choices, value=choices[0], interactive=True)
                else:
                    return gr.update(choices=[], value=None, interactive=False)
            except Exception:
                return gr.update(choices=[], value=None, interactive=False)

        model_sel.change(
            fn=on_model_change,
            inputs=[model_sel],
            outputs=[speaker],
        )

        # BGM 音量倍率变化：控制“整首（混音）”可见性（需当前模板存在 BGM 且倍率>0）
        def on_bgm_volume_change(bgm_vol, template_name):
            mapping = find_templates()
            has_bgm = False
            if template_name in mapping:
                has_bgm = bgm_path_for(mapping[template_name]) is not None
            return gr.update(visible=bool(bgm_vol and bgm_vol > 0 and has_bgm))

        bgm_volume.change(
            fn=on_bgm_volume_change,
            inputs=[bgm_volume, template_sel],
            outputs=[full_mixed],
        )

        # 模板切换：批量更新预创建文本框 + 初始化整首混音可见性
        template_sel.change(
            fn=on_template_change,
            inputs=[template_sel],
            outputs=[bgm_volume, template_sel, lines_state, offsets_state, per_line_error, full_mixed, original_lines_state, *textboxes, *error_markdowns],
        )

        # 上传模板：将用户 .ds 保存到 templates/user，并刷新模板下拉
        def on_upload_ds(file_obj):
            try:
                if not file_obj:
                    raise gr.Error("未选择文件")
                # gr.UploadButton 返回字典/路径，兼容不同返回
                import shutil
                from pathlib import Path as _P
                src = _P(file_obj.name) if hasattr(file_obj, "name") else _P(str(file_obj))
                if src.suffix.lower() != ".ds":
                    raise gr.Error("仅支持 .ds 文件")
                dst_dir = USER_TEMPLATES_DIR
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
                shutil.copyfile(src, dst)
                # 刷新模板列表，并选中新上传的模板名（无扩展）
                new_choices = get_template_choices_and_bgm_visible()
                base = src.stem
                # 若同名覆盖，模板名以去扩展后的相对名为准
                # 确认在 choices 中（user 覆盖 public）
                if base not in new_choices and (dst_dir / src.name).exists():
                    # 有些实现是用完整相对路径名，这里退化为重新计算 choices
                    pass
                return gr.update(choices=new_choices, value=base)
            except Exception as e:
                raise gr.Error(f"上传失败: {e}")

        upload.upload(
            fn=on_upload_ds,
            inputs=[upload],
            outputs=[template_sel],
        )

        # 初始构建已通过预创建完成

        # 文本提交事件：逐句渲染（为预创建文本框绑定）
        for idx, tb in enumerate(textboxes):
            def make_submit(i):
                def _submit(new_text, lines_list, original_lines_list, model_sel_v, template_sel_v, speaker_v, key_shift_v, steps_v, bgm_volume_v):
                    # 仅处理当前可见范围内的行
                    if not isinstance(lines_list, list) or i >= max(len(lines_list), 0):
                        return gr.update(), gr.update(), gr.update(), lines_list
                    
                    # 校验歌词格式
                    original_text = original_lines_list[i] if i < len(original_lines_list) else ""
                    is_valid, rendered_original = validate_lyric_format(new_text, original_text)
                    
                    # 更新错误提示
                    if not is_valid:
                        error_msg = f"字数与原始文本不符：{rendered_original}"
                        error_update = gr.update(value=error_msg, visible=True)
                    else:
                        error_update = gr.update(value="", visible=False)
                    
                    # 渲染音频
                    audio_path, err = render_single_line(model_sel_v, template_sel_v, i, new_text, speaker_v, key_shift_v, steps_v, bgm_volume_v)
                    if i < len(lines_list):
                        lines_list[i] = new_text
                    if err:
                        return gr.update(value=None), gr.update(value=f"❌ {err}", visible=True), error_update, lines_list
                    return gr.update(value=audio_path), gr.update(value="", visible=False), error_update, lines_list
                return _submit
            
            tb.submit(
                fn=make_submit(idx),
                inputs=[tb, lines_state, original_lines_state, model_sel, template_sel, speaker, key_shift, steps, bgm_volume],
                outputs=[per_line_audio, per_line_error, error_markdowns[idx], lines_state],
            )

        # 生成整首（支持进度与中断）
        def on_gen_or_stop(model_sel_v, template_sel_v, speaker_v, key_shift_v, steps_v, bgm_volume_v, lines, stop, generating, progress=gr.Progress(track_tqdm=True)):
            # 若正在生成，本次点击作为“停止”信号，仅更新按钮与提示
            if generating:
                stop = True
                return gr.update(), gr.update(), gr.update(value="生成整首"), gr.update(value="已请求停止，稍候..."), stop, generating

            # 启动生成：切换按钮，清空输出，重置停止标志
            stop = False
            generating = True
            yield gr.update(value=None), gr.update(value=None), gr.update(value="停止生成整首"), gr.update(value="开始生成..."), stop, generating

            try:
                mapping = find_templates()
                if not model_sel_v:
                    raise gr.Error("请先选择模型")
                if template_sel_v not in mapping:
                    raise gr.Error("未找到模板")
                template_path = mapping[template_sel_v]
                ds = load_ds(template_path)
                if len(ds) != len(lines or []):
                    raise gr.Error("模板行数与编辑行数不一致")

                model_path = ROOT / model_sel_v
                segs_with_offsets = []
                total = len(ds)
                for idx, item in enumerate(ds):
                    if stop:
                        yield gr.update(), gr.update(), gr.update(value="生成整首"), gr.update(value=f"已中断，完成 {idx}/{total} 行。"), stop, False
                        return
                    text = (lines[idx] or "").strip()
                    if not text:
                        # 更新进度显示但不合成
                        progress((idx + 1) / total, desc=f"跳过空白句 {idx+1}/{total}")
                        yield gr.update(), gr.update(), gr.update(value="停止生成整首"), gr.update(value=f"跳过空白句 {idx+1}/{total}"), stop, True
                        continue

                    progress((idx + 1) / total, desc=f"渲染第 {idx+1}/{total} 句")
                    yield gr.update(), gr.update(), gr.update(value="停止生成整首"), gr.update(value=f"渲染第 {idx+1}/{total} 句..."), stop, True

                    h = param_hash(model_sel_v, speaker_v, key_shift_v, steps_v, text)
                    cache_dir = OUTPUT_DIR / template_sel_v / h
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    wav_out = cache_dir / f"line_{idx+1}.wav"
                    if not wav_out.exists():
                        engine.synth_line(
                            model_path=model_path,
                            template_path=template_path,
                            line_index=idx,
                            text=text,
                            speaker=speaker_v or None,
                            key_shift=int(key_shift_v),
                            steps=int(steps_v),
                            out_wav=wav_out,
                        )
                    segs_with_offsets.append((audiosegment_from_file(wav_out), float(item.get("offset", 0.0))))

                # 拼接输出
                vocal_full = concat_with_offsets(segs_with_offsets)
                final_dir = OUTPUT_DIR / template_sel_v / "final"
                final_dir.mkdir(parents=True, exist_ok=True)
                ts_tag = time.strftime("%Y%m%d_%H%M%S")
                vocal_path = final_dir / f"{template_sel_v}_vocal_{ts_tag}.wav"
                export_wav(vocal_full, vocal_path)

                mixed_path = None
                bgm_p = bgm_path_for(template_path)
                if (bgm_volume_v and bgm_volume_v > 0.0) and bgm_p and bgm_p.exists():
                    mixed = mix_full_song(vocal_full, audiosegment_from_file(bgm_p), bgm_volume_v)
                    mixed_path = final_dir / f"{template_sel_v}_mixed_{ts_tag}.wav"
                    export_wav(mixed, mixed_path)

                # 完成
                yield gr.update(value=str(vocal_path)), gr.update(value=(str(mixed_path) if mixed_path else None)), gr.update(value="生成整首"), gr.update(value="已完成"), False, False
            except Exception as e:
                yield gr.update(value=None), gr.update(value=None), gr.update(value="生成整首"), gr.update(value=f"❌ 失败：{type(e).__name__}: {e}"), False, False

        gen_btn.click(
            fn=on_gen_or_stop,
            inputs=[model_sel, template_sel, speaker, key_shift, steps, bgm_volume, lines_state, stop_flag, generating_flag],
            outputs=[full_vocal, full_mixed, gen_btn, progress_md, stop_flag, generating_flag],
        )

        # 下载当前编辑后的 ds
        def build_current_ds(template_sel_v, lines, offsets):
            mapping = find_templates()
            if not template_sel_v or template_sel_v not in mapping:
                raise gr.Error("未选择有效模板")
            tpl = mapping[template_sel_v]
            ds = load_ds(tpl)
            # 覆盖 text，并基于最新文本重算 ph_seq / ph_num
            for i in range(min(len(ds), len(lines or []))):
                new_text = lines[i] if lines and i < len(lines) else ds[i].get("text", "")
                ds[i].replace(new_text)
            # 输出到 output/pred_all/<template>/edits
            ts_tag = time.strftime("%Y%m%d_%H%M%S")
            out_dir = OUTPUT_DIR / template_sel_v / "edits"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{template_sel_v}_edited_{ts_tag}.ds"
            out_path.write_text(json.dumps(ds, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(out_path)

        download_btn.click(
            fn=build_current_ds,
            inputs=[template_sel, lines_state, offsets_state],
            outputs=[download_btn],
        )

        # 上传模板：保存后通过更新模板下拉触发重建（复用模板切换事件）
        def on_upload(file_obj, current_template):
            name = save_uploaded_template(file_obj)
            if name.startswith("ERROR::"):
                return gr.update(choices=get_template_choices_and_bgm_visible(), value=current_template), gr.update(value=f"❌ {name[7:]}", visible=True)
            # 成功：将模板选择切换为新模板，触发 on_template_change 自动重建文本框
            return gr.update(choices=get_template_choices_and_bgm_visible(), value=name), gr.update(value="", visible=False)

        upload.upload(
            fn=on_upload,
            inputs=[upload, template_sel],
            outputs=[template_sel, per_line_error],
        )

        # 进入页面时，自动刷新 speaker 与模板/BGM/歌词区
        def on_app_load(model_path_rel, template_name):
            spk_upd = on_model_change(model_path_rel)
            tpl_upds = on_template_change(template_name)
            return (spk_upd, *tpl_upds)

        demo.load(
            fn=on_app_load,
            inputs=[model_sel, template_sel],
            outputs=[speaker, bgm_volume, template_sel, lines_state, offsets_state, per_line_error, full_mixed, original_lines_state, *textboxes, *error_markdowns],
        )

    return demo


def main():
    demo = build_ui()
    demo.launch(show_error=True)


if __name__ == "__main__":
    main()