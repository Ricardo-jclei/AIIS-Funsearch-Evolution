import subprocess
import sys
import os

# =============================
# 多资产智能投资组合优化系统 自动化全流程演示脚本
# =============================

def run_step(cmd, desc):
    print(f"\n==============================")
    print(f"【流程】{desc}")
    print(f"【命令】{cmd}")
    print(f"==============================")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"[错误] 步骤失败: {desc}")
        sys.exit(1)
    print(f"[完成] {desc}\n")

if __name__ == "__main__":
    print("\n==============================")
    print("多资产智能投资组合优化系统——自动化全流程演示")
    print("==============================\n")

    # 1. 数据采集
    run_step(
        "python -m src.collect_data",
        "1. 数据采集：自动获取股票行情、基本面、情绪、宏观等数据，保存到 data/ 目录"
    )

    # 2. 数据预处理与特征工程
    run_step(
        "python -m src.data.process_data",
        "2. 数据预处理与特征工程：清洗、融合、标准化，生成多因子特征，保存到 data/processed/"
    )

    # 3. LSTM模型训练（多资产）
    run_step(
        "python -m src.model.lstm_train_multi_asset",
        "3. LSTM模型训练（多资产）：深度特征提取，权重保存到 model_ckpt/"
    )

    # 4. 强化学习智能体训练（DDPG，多资产）
    run_step(
        "python -m src.rl.train_multi_asset_ddpg",
        "4. 强化学习训练（DDPG，多资产）：RL智能体训练，权重保存到 model_ckpt/"
    )

    # 5. 强化学习智能体训练（PPO，多资产）
    run_step(
        "python -m src.rl.train_multi_asset_ppo",
        "5. 强化学习训练（PPO，多资产）：RL智能体训练，权重保存到 model_ckpt/"
    )

    # 6. 多策略对比与自动报告生成
    run_step(
        "python -m src.user_demo",
        "6. 多策略对比与自动报告生成：自动对比RL、LSTM+RL、等权、最小方差、最大夏普等策略，生成可视化报告，输出到 output/ 目录"
    )

    print("\n==============================")
    print("🎉 全流程演示完成！请查看 output/ 目录下的报告和图片。\n")
    print("如需单步调试或详细分析，可分别运行各模块脚本。\n") 