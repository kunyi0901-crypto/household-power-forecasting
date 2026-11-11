# 家庭小时功率预测

比较日周期基线、Ridge回归和两层GRU，使用过去24小时平均有功功率预测下一小时平均功率（kW）。这是单家庭时间序列预测实验。

## 文件

- experiments/run_forecast.py：读取与处理数据、构造预测窗口、拟合基线与GRU、保存模型和评价结果。
- experiments/summarize.py：独立重算测试误差，并生成训练曲线和预测对比图。
- requirements-course.txt：直接依赖版本。

## 环境与安装

实际运行环境为Python 3.11.7、PyTorch 2.5.1+cpu，使用CPU训练。在仓库根目录执行以下Windows PowerShell命令：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-course.txt
```

依赖清单中的torch==2.5.1未指定计算后端，不同平台的安装包可能不同。本项目未验证所有平台；不同计算环境可能产生数值差异。

## 数据准备

数据：UCI Individual Household Electric Power Consumption。

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

下载原始ZIP，保存为 data/raw/household_power_consumption.zip，不必解压。原始ZIP内部需包含household_power_consumption.txt。数据不随仓库分发。

引用：Hebrail G, Berard A. (2006). Individual Household Electric Power Consumption. UCI Machine Learning Repository. DOI: 10.24432/C58K54。数据许可CC BY 4.0。

## 运行

```powershell
.\.venv\Scripts\python.exe experiments/run_forecast.py
.\.venv\Scripts\python.exe experiments/summarize.py
```

先训练再汇总。结果自动写入experiment_outputs目录；重复运行会覆盖该目录中的同名结果文件。

## 数据与实验设计

分钟功率按小时取均值，仅保留60个有效分钟的小时，其余标记为缺失并保留完整时间网格。每个样本含24小时输入和1小时标签，任何位置缺失则排除该样本。按完整时间网格的70%/15%/15%划分训练、验证和测试，按目标时点归属分段。有效窗口数分别为22,818、4,826、4,805。

标准化仅拟合训练数据。Ridge正则化参数在0.01、0.1、1、10、100、1000中按验证误差选择。GRU采用两层、隐藏维度32、Adam学习率0.001、批量256、最多40轮、早停耐心6轮，固定种子42、2024、2026。恢复验证误差最低的权重后才计算测试误差。所有模型使用相同有效测试样本，评价单位为kW。

## 主要结果

| 模型 | 测试MAE（kW） | 测试RMSE（kW） |
| --- | ---: | ---: |
| 日周期基线 | 0.5078 | 0.7584 |
| Ridge | 0.3603 | 0.5061 |
| GRU三个种子平均 | 0.3485 | 0.4940 |

GRU相对Ridge的MAE下降约3.27%，RMSE下降约2.40%。这里平均的是三个模型的指标，不是集成预测。结论限于单家庭、固定划分和一小时预测；未进行统计显著性检验。

## 输出文件

- config.json、sample_audit.csv：配置与样本筛选数量。
- ridge_validation.csv、selected_model.json：模型选择记录。
- training_log.csv、metrics.csv：训练曲线数据及评价指标。
- validation_predictions.csv、test_predictions.csv：逐时真实值与预测值。
- gru_*.pt：最佳检查点与标准化参数。
- hourly_power.csv：小时数据。
- learning_curves.png、forecast_week.png：图形；预测图固定展示2010年5月1日至7日、GRU种子42。

## 方法来源与辅助说明

GRU：Cho et al. (2014), https://aclanthology.org/D14-1179/ 。Ridge：Hoerl & Kennard (1970), DOI: 10.1080/00401706.1970.10488634。

代码编写与说明整理使用了AI辅助，作者已亲自重跑并核查实验。详细学习及重跑记录在本地保留，不纳入公开仓库。
