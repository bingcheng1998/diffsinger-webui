# 模型目录说明

## 下载声库

什么是声库？声库可以理解为歌唱者的模型，有着各自的音色等特性。

社区提供了[DiffSinger自制声库分享](https://docs.qq.com/sheet/DQXNDY0pPaEpOc3JN)，如果你不确定下载哪个，推荐从[zhibin club](https://www.zhibin.club/)下载[姜柯JiangKe](https://pan.quark.cn/s/254f030af8cb#/list/share/0929019064004907b7b95212c03066ed)声库开始尝试。

下载声库后，需要解压，解压缩到 `models/`下。

## 模型目录

将 DiffSinger 模型文件放在`models/`目录下。组织形式示例：
1. 以单个目录表示一个模型（推荐）：例如
   ```
   models/
     YourSingerModel/
       dsdur/
       dspitch/
       dsvariance/
       dsvocoder/
       variance_assets/
   ```
   选择时请在 WebUI 中选中 `YourSingerModel` 这个目录名。



注意：
- 本项目要求 Python 3.8+ 与 torch==1.13.1。
- 请确保模型结构与您安装的 diffsinger-utau 版本兼容。