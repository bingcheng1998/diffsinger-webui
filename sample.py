from calendar import c
from pathlib import Path
from random import sample
from numpy.random import f
from diffsinger_utau.voice_bank import PredAll
from diffsinger_utau.voice_bank.commons.ds_reader import DSReader
from diffsinger_utau.voice_bank.commons.phome_num_counter import Phome
from diffsinger_utau.samples import get_sample_path, list_samples
from pypinyin import pinyin, Style
import json
from typing import Optional

# 初始化预测器
voice_bank = Path("/Users/bc/Developer/diffsinger_utau/artifacts/JiangKe_DiffSinger_CE_25.06")
predictor = PredAll(voice_bank)

# 获取可能存在的声库、作者、头像和背景图
character = predictor.voice_bank_reader.character
author: Optional[str] = getattr(character, "author", None) # 声库作者
name: Optional[str] = getattr(character, "author", None) # 声库名称
image: Optional[str] = getattr(character, "image", None) # 声库头像
portrait: Optional[str] = getattr(character, "portrait", None) # 声库背景图
portrait_opacity: Optional[float] = getattr(character, "portrait_opacity", None) # 声库背景图透明度

# 获取所有说话人列表
if predictor.available_speakers:
    print(f"可用说话人: {predictor.available_speakers}")
else:
    print("未找到可用说话人")

# 读取 DS 文件，取第一句测试
sample = sorted(list_samples())
print(f'样本: {sample}')
ds_path = get_sample_path(sample[0])
ds = DSReader(ds_path).read_ds()[0]

# 修改歌词

print(f'模板歌词: {ds["text"]}')
new_text = "AP 我想 SP 所有的一切 SP 都如意 SP"
ds["text"] = new_text
print(f'修改后歌词: {ds["text"]}')

ds.replace(new_text)

# 执行完整推理
results = predictor.predict_full_pipeline(
    ds=ds,
    lang="zh",
    speaker=predictor.available_speakers[0],  # 随机选择说话人
    key_shift=0,
    pitch_steps=10,
    variance_steps=10,
    acoustic_steps=50,
    gender=0.0,
    output_dir="output/pred_all",
    save_intermediate=True,
)

print(f"生成音频: {results['audio_path']}")