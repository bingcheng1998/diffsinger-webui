import argparse
import gradio as gr
import json
import os
import sys
import hashlib
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
from pydub import AudioSegment
# 引入与 sample.py 一致的组件
try:
    from voice_bank import PredAll
    from voice_bank.commons.ds_reader import DSReader
    from voice_bank.commons.phome_num_counter import Phome
    from pypinyin import pinyin, Style
except Exception:
    PredAll = None
    DSReader = None
    Phome = None
    Style = None

def get_opencpop_dict(path: str = str(Path("dictionaries") / "opencpop-extension.txt")) -> Dict[str, str]:
    result = {"AP": "AP", "SP": "SP"}
    p = Path(path)
    if not p.exists():
        return result
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                k, v = line.split("\t", 1)
                result[k.strip()] = v.strip()
    return result

def get_phonemes(text: str, opencpop_dict: Dict[str, str]) -> List[str]:
    if Style is None:
        # 无 pypinyin 时，退化为逐字符
        return [opencpop_dict.get(ch, ch) for ch in list(text)]
    pys = [x[0] for x in pinyin(text, style=Style.NORMAL)]
    result: List[str] = []
    for py in pys:
        py = py.strip()
        if not py:
            continue
        result.append(opencpop_dict.get(py, py))
    return " ".join(result).split()

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


def load_ds(template_path: Path) -> List[Dict]:
    # ds: 一个 list，每个元素为 dict，至少包含 text；可包含 offset（秒）
    with open(template_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("ds 模板需要是一个 list")
    # 标准化
    norm = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"ds 第 {i+1} 个元素不是 dict")
        text = item.get("text", "")
        if not isinstance(text, str):
            raise ValueError(f"ds 第 {i+1} 个元素的 text 不是字符串")
        offset = item.get("offset", 0.0)
        try:
            offset = float(offset)
        except Exception:
            offset = 0.0
        norm.append({"text": text, "offset": offset, **item})
    return norm


def audiosegment_from_file(path: Path) -> AudioSegment:
    return AudioSegment.from_file(str(path))


def export_wav(seg: AudioSegment, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    seg.export(str(path), format="wav")


def overlay_bgm_snippet(vocal_wav: Path, bgm_audio: AudioSegment, offset_sec: float, gain_db: float = 0.0) -> AudioSegment:
    vocal = audiosegment_from_file(vocal_wav)
    if gain_db != 0.0:
        vocal = vocal + gain_db
    start_ms = max(int(offset_sec * 1000), 0)
    # 确保 BGM 足够长，不够则静音补齐
    if len(bgm_audio) < start_ms + len(vocal):
        pad_ms = start_ms + len(vocal) - len(bgm_audio)
        bgm_audio = bgm_audio + AudioSegment.silent(duration=pad_ms)
    mixed = bgm_audio.overlay(vocal, position=start_ms)
    # 裁剪到混音范围以便预览：从 start_ms 到 start_ms+len(vocal)
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


def mix_full_song(vocal: AudioSegment, bgm: AudioSegment) -> AudioSegment:
    # 保证两者同长度
    if len(bgm) < len(vocal):
        bgm = bgm + AudioSegment.silent(duration=(len(vocal) - len(bgm)))
    else:
        vocal = vocal + AudioSegment.silent(duration=(len(bgm) - len(vocal)))
    return bgm.overlay(vocal)


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
        self.predictors: Dict[str, object] = {}  # model_path -> PredAll 实例
        self.opencpop = get_opencpop_dict()

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
            ds_list = DSReader(str(template_path)).read_ds()
            if not (0 <= line_index < len(ds_list)):
                raise IndexError("行索引越界")
            ds = ds_list[line_index]
            old_text = ds.get("text", "")
            ds["text"] = text
            # 若文本变化或缺少音素信息，则基于新文本重算音素
            if (text != old_text) or (not ds.get("ph_seq")) or (not ds.get("ph_num")):
                phonemes = get_phonemes(text, self.opencpop)
                ds["ph_seq"] = " ".join(phonemes)
                ds["ph_num"] = " ".join(map(str, Phome(phonemes).get_ph_num())) if Phome else ""

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
        return gr.update(visible=False, value=False), [], []
    p = mapping[template_name]
    bgm = bgm_path_for(p)
    ds = load_ds(p)
    lines = [item.get("text", "") for item in ds]
    offsets = [float(item.get("offset", 0.0)) for item in ds]
    bgm_update = gr.update(visible=(bgm is not None), value=(bgm is not None))
    return bgm_update, lines, offsets


def render_single_line(
    model_sel: str,
    template_name: str,
    line_index: int,
    new_text: str,
    speaker: str,
    key_shift: int,
    steps: int,
    use_bgm: bool,
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
        if use_bgm and mapping_bgm and mapping_bgm.exists():
            bgm_audio = audiosegment_from_file(mapping_bgm)
            offset = float(ds[line_index].get("offset", 0.0))
            mixed_seg = overlay_bgm_snippet(wav_out, bgm_audio, offset)
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
    use_bgm: bool,
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
        if use_bgm and bgm_p and bgm_p.exists():
            bgm_audio = audiosegment_from_file(bgm_p)
            mixed = mix_full_song(vocal_full, bgm_audio)
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

    with gr.Blocks(title="DiffSinger WebUI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## DiffSinger WebUI")
        with gr.Row():
            model_sel = gr.Dropdown(choices=model_choices, label="模型选择", value=(model_choices[0] if model_choices else None))
            template_sel = gr.Dropdown(choices=template_names, label="模板选择", value=(template_names[0] if template_names else None))

        with gr.Row():
            bgm_switch = gr.Checkbox(label="BGM开关", value=False, visible=False)
            key_shift = gr.Slider(-12, 12, value=0, step=1, label="key_shift")
            steps = gr.Slider(1, 100, value=20, step=1, label="acoustic_steps")
            speaker = gr.Dropdown(label="speaker", choices=[], value=None, interactive=True)

        # 载入模板后动态生成每句文本框（改为预创建+批量update）
        lines_state = gr.State([])
        offsets_state = gr.State([])
        textboxes = []

        # 占位容器
        per_line_audio = gr.Audio(label="单句预览", autoplay=True, interactive=False)
        per_line_error = gr.Markdown("", visible=False)

        def rebuild_textboxes(lines):
            comps = []
            for idx, txt in enumerate(lines):
                comps.append(gr.Textbox(value=txt, label=f"第 {idx+1} 句", lines=1, max_lines=1))
            return comps

        # 动态区域
        dyn = gr.Column()
        # 预创建文本框，依据当前模板设置初始可见性和值
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
        if template_sel.value:
            lines_state.value = init_lines
            offsets_state.value = init_offsets
        gen_btn = gr.Button("生成整首")
        with gr.Row():
            full_vocal = gr.Audio(label="整首（人声）", autoplay=False)
            full_mixed = gr.Audio(label="整首（混音）", autoplay=False)

        with gr.Row():
            upload = gr.File(label="上传 ds 模板（同名覆盖公开模板）", file_types=[".ds"], file_count="single")
        with gr.Row():
            download_btn = gr.DownloadButton(label="下载当前编辑状态(.ds)")

        # 事件：选择模板时，更新 BGM 开关显示与文本框内容
        def on_template_change(template_name):
            bgm_update, lines, offsets = on_select_template(template_name)
            tb_updates = []
            n = len(lines)
            for i, tb in enumerate(textboxes):
                if i < n:
                    tb_updates.append(gr.update(value=lines[i], visible=True, label=f"第 {i+1} 句"))
                else:
                    tb_updates.append(gr.update(value="", visible=False, label=f"第 {i+1} 句"))
            # 返回：BGM、模板下拉、状态、错误清空、以及所有文本框更新
            return (
                bgm_update,
                gr.update(choices=get_template_choices_and_bgm_visible(), value=template_name),
                lines,
                offsets,
                gr.update(value="", visible=False),
                *tb_updates,
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

        # 模板切换：批量更新预创建文本框
        template_sel.change(
            fn=on_template_change,
            inputs=[template_sel],
            outputs=[bgm_switch, template_sel, lines_state, offsets_state, per_line_error, *textboxes],
        )

        # 初始构建已通过预创建完成

        # 文本提交事件：逐句渲染（为预创建文本框绑定）
        for idx, tb in enumerate(textboxes):
            def make_submit(i):
                def _submit(new_text, lines_list, model_sel_v, template_sel_v, speaker_v, key_shift_v, steps_v, use_bgm_v):
                    # 仅处理当前可见范围内的行
                    if not isinstance(lines_list, list) or i >= max(len(lines_list), 0):
                        return gr.update(), gr.update(), lines_list
                    audio_path, err = render_single_line(model_sel_v, template_sel_v, i, new_text, speaker_v, key_shift_v, steps_v, use_bgm_v)
                    if i < len(lines_list):
                        lines_list[i] = new_text
                    if err:
                        return gr.update(value=None), gr.update(value=f"❌ {err}", visible=True), lines_list
                    return gr.update(value=audio_path), gr.update(value="", visible=False), lines_list
                return _submit
            tb.submit(
                fn=make_submit(idx),
                inputs=[tb, lines_state, model_sel, template_sel, speaker, key_shift, steps, bgm_switch],
                outputs=[per_line_audio, per_line_error, lines_state],
            )

        # 生成整首
        def on_gen(model_sel_v, template_sel_v, speaker_v, key_shift_v, steps_v, use_bgm_v, lines):
            vocal, mixed = generate_full_song(model_sel_v, template_sel_v, lines, speaker_v, key_shift_v, steps_v, use_bgm_v)
            if isinstance(mixed, str) and mixed.startswith("ERROR::"):
                return gr.update(value=None), gr.update(value=None)
            if mixed is None:
                return gr.update(value=vocal), gr.update(value=None)
            return gr.update(value=vocal), gr.update(value=mixed)

        gen_btn.click(
            fn=on_gen,
            inputs=[model_sel, template_sel, speaker, key_shift, steps, bgm_switch, lines_state],
            outputs=[full_vocal, full_mixed],
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
                ds[i]["text"] = new_text
                phonemes = get_phonemes(new_text, get_opencpop_dict())
                ds[i]["ph_seq"] = " ".join(phonemes)
                ds[i]["ph_num"] = " ".join(map(str, Phome(phonemes).get_ph_num())) if Phome else ds[i].get("ph_num", "")
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

        upload.change(
            fn=on_upload,
            inputs=[upload, template_sel],
            outputs=[template_sel, per_line_error],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()