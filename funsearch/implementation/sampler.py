# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import openai

from funsearch.implementation import evaluator
from funsearch.implementation import programs_database


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt

    # ====================== 【填写你的API配置】 ======================
    self.API_KEY = "sk-hbob2QgBFpclKVU8yrjXHmwIIN36o8cu1k1ccUGUXF1jqam0"
    self.HOST_URL = "https://api.bltcy.ai"
    # ===============================================================

    print("Connecting to OpenAI API service...")
    self.client = openai.OpenAI(
        base_url=f"{self.HOST_URL}/v1",
        api_key=self.API_KEY,
        timeout=120,
    )
    print("API connection initialized!")

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt` via OpenAI API."""
    try:
      response = self.client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[
              {"role": "system", "content": "You are an expert investment strategy developer. Generate innovative, diverse, and effective investment strategies. Use market_state[:, :5] (first 5 features) for 5 stocks. Return only pure function body code, no explanation. Include:\n1. Market trend analysis using recent price movements\n2. Volatility-based weighting\n3. Mean-reversion strategies\n4. Momentum-based approaches\n5. Risk control mechanisms\n6. Diverse weighting schemes\n7. Proper handling of portfolio=None case\n8. Return shape must be (5,)\n9. Weights must be non-negative and sum to 1\n10. Use only numpy operations\n\nCRITICAL NUMERICAL STABILITY REQUIREMENTS:\n- AVOID np.exp() with large values - use np.clip(x, -10, 10) before exp: np.exp(np.clip(x, -10, 10))\n- ALL divisions must have small constant protection: / (value + 1e-8)\n- Use np.clip(weights, 0, 1) to ensure non-negative weights\n- Avoid std() of zero - always add: / (np.std(...) + 1e-8)\n- Test intermediate values for inf/nan\n\nIMPORTANT: Generate diverse strategies, not just equal weight. Avoid simple mean strategies."},
              {"role": "user", "content": f"Complete this investment strategy function. Use market_state[:, :5] for 5 stocks. Generate diverse, innovative strategies. Examples of good strategies:\n\nExample 1: Trend-following with volatility adjustment (STABLE)\nif portfolio is None:\n    portfolio = market_state[-1, :5] if market_state is not None else np.ones(5)\ntrend = market_state[-1, :5] - market_state[0, :5]\nvolatility = np.std(market_state[:, :5], axis=0)\nweights = np.exp(np.clip(trend, -10, 10)) / (volatility + 1e-8)\nweights = np.clip(weights, 0, 1)\nweights = weights / (np.sum(weights) + 1e-8)\nreturn weights\n\nExample 2: Mean-reversion with dynamic adjustment (STABLE)\nif portfolio is None:\n    portfolio = market_state[-1, :5] if market_state is not None else np.ones(5)\nmean_prices = np.mean(market_state[:, :5], axis=0) + 1e-8\ncurrent_prices = market_state[-1, :5] + 1e-8\nweights = mean_prices / current_prices\nweights = np.clip(weights, 0, 1)\nweights = weights / (np.sum(weights) + 1e-8)\nreturn weights\n\nExample 3: Momentum-based with risk parity (STABLE)\nif portfolio is None:\n    portfolio = market_state[-1, :5] if market_state is not None else np.ones(5)\nmomentum = market_state[-5:, :5].mean(axis=0) - market_state[:5, :5].mean(axis=0)\nvolatility = np.std(market_state[:, :5], axis=0) + 1e-8\nweights = momentum / volatility\nweights = np.clip(weights, 0, 1)\nweights = weights / (np.sum(weights) + 1e-8)\nreturn weights\n\nNow complete:\n{prompt}"}
          ],
          temperature=0.9,  # 增加温度以提高多样性
          max_tokens=500     # 增加token以允许更复杂的策略
      )
      generated_code = response.choices[0].message.content.strip()
      # 移除stock_list引用，因为这个变量不存在
      generated_code = generated_code.replace('stock_list', '5')
      # 确保使用market_state[:, :5]
      if 'market_state' in generated_code and 'market_state[:, :5]' not in generated_code:
          # 简单替换，避免复杂的正则表达式
          generated_code = generated_code.replace('market_state', 'market_state[:, :5]')

    except Exception as e:
      print(f"[API Error] {str(e)} → Using fallback strategy")
      generated_code = ""

    # -------------------------- Code Fix Logic --------------------------
    # 移除代码块标记
    if '```python' in generated_code:
        generated_code = generated_code.replace('```python', '')
    if '```' in generated_code:
        generated_code = generated_code.replace('```', '')
    
    # 移除完整的函数定义，只保留函数体
    # LLM可能生成完整的函数定义如 "def investment_strategy_v1(...)" 或 "def investment_strategy(...)"
    # 我们需要移除这些，只保留函数体
    import re
    
    # 去除首尾空白
    generated_code = generated_code.strip()
    
    # 首先检查是否包含完整的函数定义
    if generated_code.startswith('def investment_strategy'):
        # 找到函数定义结束的位置（第一个换行后的内容）
        lines = generated_code.split('\n', 1)
        if len(lines) > 1:
            # 移除第一行（函数定义）
            generated_code = lines[1]
        else:
            # 如果只有一行，那就直接使用这一行作为body
            # 尝试找到冒号后的位置
            colon_idx = generated_code.find(':')
            if colon_idx != -1:
                generated_code = generated_code[colon_idx+1:]
    
    if 'investment_strategy_with_savgol' in generated_code:
        generated_code = generated_code.replace('investment_strategy_with_savgol', 'investment_strategy')
    
    if 'return' not in generated_code:
        generated_code += '\n    return np.ones(len(portfolio)) / len(portfolio)'
    
    if 'from scipy.signal import savgol_filter' in generated_code:
        generated_code = generated_code.replace('from scipy.signal import savgol_filter', '')
        generated_code = generated_code.replace('savgol_filter', 'np.mean')
    
    if 'sharpe_ratio' in generated_code and 'return sharpe_ratio' in generated_code:
        generated_code = generated_code.replace('return sharpe_ratio', 'return np.ones(len(portfolio)) / len(portfolio)')
    
    if 'portfolio is None' not in generated_code and 'portfolio == None' not in generated_code:
        lines = generated_code.split('\n')
        if len(lines) > 0:
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('#'):
                    portfolio_check = '    # Handle None portfolio\n    if portfolio is None:\n        portfolio = market_state[-1, :5] if market_state is not None else np.ones(5)\n'
                    lines.insert(i, portfolio_check)
                    break
            generated_code = '\n'.join(lines)
    
    # 添加数值稳定性处理
    # 修复clip函数参数错误
    import re
    
    # 修复嵌套的clip调用
    generated_code = re.sub(r'np\.clip\(np\.clip\(([^,]+),\s*(-?\d+),\s*(\d+),\s*(-?\d+),\s*(\d+)\)\)', r'np.clip(\1, \4, \5)', generated_code)
    
    # 修复基础权重计算
    generated_code = re.sub(r'base_weights\s*=\s*np\.ones\(len\(portfolio\)\)\s*/\(\(len\(portfolio\)\s*\+\s*1e-8\)\s*\+\s*1e-8\)', r'base_weights = np.ones(len(portfolio)) / len(portfolio)', generated_code)
    
    # 移除错误的除法修复（之前的修复导致了语法错误）
    # 不再使用会导致语法错误的除法修复
    
    # 增强数值稳定性：确保所有np.exp都有clip保护
    # 匹配 np.exp(xxx) 其中xxx不包含clip
    generated_code = re.sub(r'np\.exp\(([^)]+)\)', lambda m: 'np.exp(np.clip(' + m.group(1) + ', -10, 10))' if 'clip' not in m.group(1) else m.group(0), generated_code)
    
    # 移除多余的1e-8保护
    generated_code = re.sub(r'\+\s*1e-8\s*\+\s*1e-8', '+ 1e-8', generated_code)
    generated_code = re.sub(r'\+\s*1e-8\)\s*\+\s*1e-8\)', ') + 1e-8)', generated_code)
    
    if 'lstm_model(' in generated_code:
        lines = generated_code.split('\n')
        generated_code = '\n'.join([l for l in lines if 'lstm_model(' not in l])
    
    if 'pd.DataFrame' in generated_code or 'pandas' in generated_code:
        lines = generated_code.split('\n')
        generated_code = '\n'.join([l for l in lines if 'pd.DataFrame' not in l and 'pandas' not in l])
    
    if 'return np.ones(1)' in generated_code or 'return np.array([0.2])' in generated_code:
        generated_code = generated_code.replace('return np.ones(1)', 'return np.ones(len(portfolio)) / len(portfolio)')
        generated_code = generated_code.replace('return np.array([0.2])', 'return np.ones(len(portfolio)) / len(portfolio)')
    
    # 只在代码为空时使用回退策略
    if not generated_code or len(generated_code.strip()) < 10:
        import random
        fallback_choice = random.randint(1, 4)
        if fallback_choice == 1:
            generated_code = '    # Trend-following strategy\n    if market_state is not None:\n        trend = market_state[-1, :5] - market_state[0, :5]\n        weights = np.exp(trend)\n        weights = weights / np.sum(weights)\n        return weights\n    else:\n        return np.ones(5) / 5'
        elif fallback_choice == 2:
            generated_code = '    # Mean-reversion strategy\n    if market_state is not None:\n        mean_prices = np.mean(market_state[:, :5], axis=0)\n        current_prices = market_state[-1, :5]\n        weights = mean_prices / (current_prices + 1e-8)\n        weights = weights / np.sum(weights)\n        return weights\n    else:\n        return np.ones(5) / 5'
        elif fallback_choice == 3:
            generated_code = '    # Volatility-based strategy\n    if market_state is not None:\n        volatility = np.std(market_state[:, :5], axis=0)\n        weights = 1.0 / (volatility + 1e-8)\n        weights = weights / np.sum(weights)\n        return weights\n    else:\n        return np.ones(5) / 5'
        else:
            generated_code = '    # Momentum strategy\n    if market_state is not None and len(market_state) > 5:\n        momentum = market_state[-5:, :5].mean(axis=0) - market_state[:5, :5].mean(axis=0)\n        weights = np.maximum(0, momentum)\n        weights = weights / np.sum(weights)\n        return weights\n    else:\n        return np.ones(5) / 5'
    
    print(f"[LLM] Generated code:\n{generated_code}")
    return generated_code

  def draw_samples(self, prompt: str) -> Collection[str]:
    samples = [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]
    for i, sample in enumerate(samples):
        if "Equal weight" in sample:
            print(f"[Sampler] Sample {i}: Using fallback")
        else:
            print(f"[Sampler] Sample {i}: New code generated")
    return samples


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
      max_samples: int = None,
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt)
    self._max_samples = max_samples
    self._sample_count = 0

  def sample(self):
    while True:
      if self._max_samples is not None and self._sample_count >= self._max_samples:
        print(f"\nAuto-stop: Sample limit reached ({self._sample_count} samples)")
        break
      
      prompt = self._database.get_prompt()
      
      # 获取前一轮的最佳策略
      best_programs = []
      best_scores = []
      for island_id in range(len(self._database._best_program_per_island)):
          program = self._database._best_program_per_island[island_id]
          score = self._database._best_score_per_island[island_id]
          if program and score > -float('inf'):
              best_programs.append(program)
              best_scores.append(score)
      
      # 构建包含历史信息的提示
      if best_programs:
          # 按分数排序，取前3个最佳策略
          sorted_indices = sorted(range(len(best_scores)), key=lambda i: best_scores[i], reverse=True)
          top_programs = [best_programs[i] for i in sorted_indices[:3]]
          
          history_info = "\n# 历史最佳策略：\n"
          for i, program in enumerate(top_programs):
              history_info += f"\n# 策略 {i+1}:\n{program.body}\n"
          
          enhanced_prompt = prompt.code + history_info + "\n# 请基于历史最佳策略生成更优的版本："
      else:
          enhanced_prompt = prompt.code
      
      samples = self._llm.draw_samples(enhanced_prompt)
      self._sample_count += 1
      for sample in samples:
        chosen_evaluator = np.random.choice(self._evaluators)
        chosen_evaluator.analyse(
            sample, prompt.island_id, prompt.version_generated)