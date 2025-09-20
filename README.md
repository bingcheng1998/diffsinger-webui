# DiffSinger Gradio WebUI

一个基于 Python 的 DiffSinger WebUI，支持模板驱动逐句渲染、整曲合成与 BGM 混音。

## 环境要求
- Python 3.8+
- torch==1.13.1
- 其余依赖见 `requirements.txt`

## 安装
```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install -r requirements.txt
```

请确保您能成功安装与当前系统兼容的 `torch==1.13.1` 以及 `diffsinger-utau`。

## 目录结构
- `models/`：放置 DiffSinger 模型（详见 `models/README.md`）
- `templates/public/`：公开 ds 模板（可共建）
- `templates/user/`：用户上传 ds 模板（同名覆盖公开模板）
- `output/pred_all/`：缓存与最终输出

BGM：将与模板同名的音频（如 `song.ds` 与 `song.mp3`）放在同一目录可启用 BGM 开关。

## 启动
```bash
python app.py --host 0.0.0.0 --port 7860
```

首次使用可上传一个 ds 模板，或参考 diffsinger-utau 的 sample.ds。