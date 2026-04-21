# 多资产智能投资组合优化系统（LSTM + RL + 多策略）

## 项目简介

本项目集成了多因子特征工程、LSTM深度学习、强化学习（DDPG、PPO等）、经典投资组合优化等多种方法，实现多资产投资组合的智能优化与自动化对比分析。系统支持多种优化策略，自动生成可视化报告，适用于量化投资研究与实盘策略开发。

---

## 目录结构

```
├── src/                    # 核心源码
│   ├── rl/                 # 强化学习模块（PPO/DDPG/多资产环境/对比实验）
│   ├── model/              # LSTM及深度模型
│   ├── portfolio/          # 投资组合优化方法
│   ├── eval/               # 评估与对比分析
│   ├── data/               # 数据采集、处理与特征工程
│   ├── database/           # 数据库初始化与管理
│   └── user_demo.py        # 用户主流程与多策略对比入口
├── data/                   # 数据文件（原始/处理后）
├── model_ckpt/             # 训练好的模型权重
├── output/                 # 自动生成的对比图片与报告
├── requirements.txt        # 依赖包列表
└── README.md               # 项目说明文档
```

---

## 快速上手

1. **环境安装**
   ```bash
   pip install -r requirements.txt
   ```

<!-- 2. **数据采集**
   ```bash
   python -m src.collect_data
   ```
   - 自动采集沪深主流股票的行情、基本面、情绪、宏观等数据，保存在`data/`目录。

3. **数据预处理与特征工程**
   ```bash
   python -m src.data.process_data
   ```
   - 数据清洗、特征工程、标准化，结果保存在`data/processed/`。 -->

4. **LSTM模型训练**
   - 单资产训练：
     ```bash
     python -m src.model.lstm_train
     ```
   - 多资产训练：
     ```bash
     python -m src.model.lstm_train_multi_asset
     ```

5. **强化学习智能体训练**
   - PPO多资产训练与reward对比实验：
     ```bash
     python -m src.rl.train_multi_asset_ppo
     python -m src.rl.ppo_reward_compare
     ```

6. **多策略对比与报告生成**
   ```bash
   python -m src.user_demo
   ```
   - 自动对比RL、LSTM+RL、等权、最小方差、最大夏普等多种策略，结果与报告保存在`output/`目录。

---

## 支持的优化策略

- RL智能体（DDPG、PPO，支持多reward对比）
- LSTM+RL协同推理
- 等权重分配
- 最小方差组合
- 最大夏普比组合

---

## 主要特性

- 多因子特征工程与深度LSTM特征提取
- 多资产RL环境，支持PPO/DDPG等主流算法
- reward函数灵活切换与对比（Sharpe、Sortino、Calmar等）
- 多策略协同推理与自动化对比
- 自动化可视化与报告生成
- 数据采集、预处理、训练、评估全流程自动化
- 支持数据库管理与环境变量配置
- 代码结构清晰，便于扩展新算法和新特征

---

## 依赖环境

详见`requirements.txt`，已覆盖所有主流深度学习、RL、数据处理、可视化、数据库等依赖。

---

## 版本维护建议

- 当前版本：v1.1.0
- 后续建议：
  - 增加Web可视化界面与交互式分析
  - 支持更多市场特征与数据源
  - 集成SAC等更先进RL算法
  - 自动化数据清理与异常检测
  - 完善单元测试与持续集成
  - 丰富文档与API说明

---
