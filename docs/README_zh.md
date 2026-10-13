# 中文复现指南

本仓库只包括第一篇论文的核心内容：PaLR-DF、任务加权谱诊断、秩干预与比较结果。论文标题为 **Long-Horizon Satellite Clock Bias Prediction: From Mapping Redundancy to Low-Rank Decoding**。

## 先验证表格和图片

按首页安装 Python 3.11 环境和依赖，在仓库根目录运行：

```bash
python -m unittest discover -s tests -v
palr-df verify-results --output runs/tables
palr-df figures --output runs/figures
```

这部分不需要下载权重。程序核对文件哈希、完整星集、种子、窗口数量，然后重算论文的比较表、三种子／五种子秩干预表和谱中位数。`verification.json` 中 `passed` 应为 `true`。图片会同时生成 PNG 和 PDF，字体采用跨平台配置。

## 再验证程序、数据和权重

将同版本的 `PaLR-DF-v0.1.0-assets.zip` 解压到仓库根目录，得到 `artifacts/`：

```bash
palr-df verify-assets --assets artifacts
palr-df evaluate --assets artifacts --horizon 12h --seed 2024 --satellites C29 --output runs/replay_C29
palr-df spectrum --assets artifacts --family v2 --horizon 12h --seed 2024 --satellites C29 --output runs/spectrum_C29
```

`evaluate` 从原始归档权重生成预测，保存逐窗口误差、预测数组、论文参考值和数值差异。`spectrum` 从训练窗口重新提取表示，计算普通、上下文加权、任务加权谱，并与归档有效秩比较。CPU 即可完成，默认 4 线程；网络前向使用 float32，钟差重建与评价使用 float64。

数据包共 560 个权重：PaLR-DF 与 V2 各 280 个，对应两种时长、28 星和五种子。`--satellites all` 回放某个时长／种子的完整历史星集；全矩阵脚本见英文 [复现指南](reproduction.md)。

## 训练入口

```bash
palr-df train --config configs/palr_12h.json --data artifacts/data/code_clock.npz --satellite C29 --seed 2024 --epochs 1 --output runs/smoke_C29
```

去掉 `--epochs 1` 即采用完整配置训练；加入 `--rank 6` 等参数可改变残差基秩。短运行明确标记为程序检查，不能代替论文结果。训练保存配置、过程记录、最佳权重和数据哈希；程序拒绝覆盖已有非空输出目录。

论文的原始 28 星结果完整保留，包括 C41。任何后续新星集、新时期实验都应单独命名和报告，不能混入这些归档数据。具体训练／选择／报告区间、基线实现口径见 [实验协议](protocol.md)。

## 文件说明

- `README.md`：英文展示首页和精确复现命令。
- `src/palr_df/`：独立方法代码及精选比较适配器。
- `results/frozen/`：2,520 条比较记录、1,400 条秩干预记录、560 条谱记录及来源哈希。
- `docs/data.md`：CODE 数据来源、单位、GPS 时间系统及无插补的转换规则。
- `results/validation.json`：本发布包的实际检查结果。

代码按 MIT 发布，CODE/IGS 数据遵循提供方条款并保留引用要求。请同时引用论文和 CODE/IGS 数据来源。
