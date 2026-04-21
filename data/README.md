# 数据目录结构

```
data/
├── raw/                    # 原始数据
│   ├── market/            # 市场数据
│   │   ├── daily/         # 日线数据
│   │   │   ├── stock_list.csv
│   │   │   └── {symbol}/  # 按股票代码分目录
│   │   │       └── YYYYMMDD_YYYYMMDD.csv
│   │   ├── minute/        # 分钟数据
│   │   │   └── {symbol}/
│   │   │       └── YYYYMMDD_[1,5,15,30,60]min.csv
│   │   ├── industry/      # 行业数据
│   │   │   └── {industry_code}_YYYYMMDD_YYYYMMDD.csv
│   │   └── realtime/      # 实时数据
│   │
│   ├── fundamental/       # 基本面数据
│   │   └── daily/
│   │       └── {symbol}/
│   │           ├── 资产负债表.csv
│   │           ├── 利润表.csv
│   │           └── 现金流量表.csv
│   │
│   ├── sentiment/         # 市场情绪数据
│   │   └── daily/
│   │       └── {symbol}/
│   │           ├── north_money_YYYYMMDD_YYYYMMDD.csv
│   │           ├── margin_YYYYMMDD_YYYYMMDD.csv
│   │           └── top_traders_YYYYMMDD_YYYYMMDD.csv
│   │
│   └── macro/            # 宏观经济数据
│       └── daily/
│           ├── gdp.csv
│           ├── cpi.csv
│           ├── retail_sales.csv
│           └── industrial_production.csv
│
└── processed/            # 处理后的数据
    ├── market/          # 处理后的市场数据
    │   ├── daily/      # 日频数据
    │   │   └── {symbol}/
    │   │       └── features_YYYYMMDD.csv
    │   └── minute/     # 分钟级数据
    │       └── {symbol}/
    │           └── features_YYYYMMDD_[1,5,15,30,60]min.csv
    │
    ├── fundamental/     # 处理后的基本面数据
    │   └── daily/
    │       └── {symbol}/
    │           └── features_YYYYMMDD.csv
    │
    ├── sentiment/       # 处理后的情绪数据
    │   └── daily/
    │       └── {symbol}/
    │           └── features_YYYYMMDD.csv
    │
    └── macro/          # 处理后的宏观数据
        └── daily/
            └── features_YYYYMMDD.csv

## 说明

1. 原始数据（raw/）：
   - 直接从数据源获取的未经处理的数据
   - 按数据类型和时间频率分类存储
   - 文件名包含数据的时间范围

2. 处理后数据（processed/）：
   - 经过清洗、特征工程后的数据
   - 保持与原始数据相同的组织结构
   - 文件名统一使用 features_YYYYMMDD 格式

3. 命名规范：
   - 股票代码目录：使用6位数字代码
   - 日期格式：YYYYMMDD
   - 文件分隔符：下划线（_）

4. 数据更新：
   - 日频数据：每个交易日收盘后更新
   - 分钟数据：实时更新
   - 财务数据：按季度更新
   - 宏观数据：按发布频率更新 