## 程序说明介绍

### 00_download_chembl.py
从ChEMBL的API中链接，寻找standard_type = MIC且target_organism = Escherichia coli的符合要求的数据，为后续的回归训练准备数据。

### chembl_ecoli_MIC_raw.csv
是下载的chembl上所有符合standard_type = MIC且target_organism = Escherichia coli要求的raw数据，共计Shape:(90099, 47)

### 数据清洗
01_stat_chembl.py是查看数据都有哪些类型，02_dataclean.py负责筛选和清晰，从 ChEMBL 原始记录 `chembl_ecoli_MIC_raw.csv` 中进行保守筛选，并按最终保留的记录数生成`chembl_ecoli_MIC_clean_<记录数>.csv`。

**清洗流程图**

```text
90,099 raw ChEMBL activities
              │
              ▼
必需字段完整：SMILES、MIC、菌株、assay、文献及分子标识符
              │
              ▼
MIC 可转为有限数值，并且 standard_value > 0
              │
              ▼
standard_type == "MIC"
standard_units == "ug.mL-1"
standard_relation == relation == "="
              │
              ▼
standard_flag == 1
data_validity_comment 为空
potential_duplicate == 0
              │
              ▼
target_pref_name == target_organism == "Escherichia coli"
target_tax_id == 562
assay_description 包含 "ATCC 25922"
              │
              ▼
排除协同、联合用药及存在其他化合物的实验
              │
              ▼
明确为 broth dilution / microdilution / CLSI-NCCLS M07
排除 agar、disk/disc、diffusion、E-test 等方法
              │
              ▼
SMILES 为单片段结构（不含 "."）
              │
              ▼
RDKit 能够解析并完成结构 sanitization
              │
              ▼
检查 canonical SMILES 是否重复
           ┌──┴──┐
           │     │
          有     无
           │     │
           ▼     │
   整个重复组全部删除
           └──┬──┘
              ▼
检查 parent molecule ChEMBL ID 是否重复
           ┌──┴──┐
           │     │
          有     无
           │     │
           ▼     │
   整个重复组全部删除
           └──┬──┘
              ▼
生成回归标签：log2_mic = log2(standard_value)
              │
              ▼
最终一致性检查
         ┌────┴────┐
         │         │
       未通过      通过
         │         │
         ▼         ▼
    抛出错误      保存 CSV
    不保存数据    clean_<最终记录数>.csv
```

最终数据集仅包含单一菌种及菌株 *Escherichia coli* ATCC 25922 的单药 MIC 记录。活性指标统一为 MIC，单位统一为 μg/mL（ChEMBL 标准单位 `ug.mL-1`），原始关系字段 `relation` 和标准化关系字段 `standard_relation` 均为精确值 `=`，MIC 必须为有限且大于 0 的数值。

数据质量方面，仅保留 `standard_flag = 1`、无 `data_validity_comment` 且未被 ChEMBL 标记为潜在重复（`potential_duplicate = 0`）的记录；排除协同用药、联合用药及在其他化合物存在条件下测量的记录；实验描述必须明确属于 broth dilution、microdilution 或 CLSI/NCCLS M07 broth-dilution 方法，并排除 agar、disk/disc、diffusion 和 E-test 等非肉汤稀释实验。

分子结构方面，仅保留不含 `.` 的单片段 canonical SMILES，并要求结构能够通过 RDKit 解析。为避免同一化学实体对应多个标签，凡 canonical SMILES 重复或 parent molecule ChEMBL ID 重复的组均整体删除，而不是从组内任意选择一条记录。因此，最终数据中的每个 canonical SMILES 和每个 parent molecule 均只出现一次。

脚本在保留原始 ChEMBL 字段的基础上，新增连续回归标签：

```text
log2_mic = log2(standard_value)
```

其中 `standard_value` 是以 μg/mL 表示的原始 MIC，`log2_mic` 用作 D-MPNN 回归模型的训练目标；原始 MIC 仍被保留，以便审计、结果解释及将模型预测转换回 μg/mL。
