from .base_processor import BaseProcessor
import pandas as pd
from typing import Dict, List

class MacroProcessor(BaseProcessor):
    """宏观数据处理器类，处理宏观经济数据"""

    def __init__(self, config: Dict):
        super().__init__(config)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理宏观数据的主方法"""
        required = ["商品", "日期", "今值", "预测值", "前值"]
        missing = set(required) - set(data.columns)
        if missing:
            self.logger.warning(f"宏观数据缺少必需的列: {missing}，自动补齐为NaN")
            for col in missing:
                data[col] = pd.NA
        # 处理缺失值
        macro_cfg = self.config.get('macro', {}) if hasattr(self.config, 'get') else {}
        data = self.handle_missing_values(data, strategy=macro_cfg.get('missing_value_strategy', 'fill'))
        # 提取特征
        data = self.extract_features(data)
        return data

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """提取宏观经济特征"""
        macro_cfg = self.config.get('macro', {}) if hasattr(self.config, 'get') else {}
        features = macro_cfg.get('features', [])
        for feature in features:
            if feature == 'gdp_growth_rate' and 'gdp' in data.columns:
                data['gdp_growth_rate'] = data['gdp'] / data['gdp'].shift(1) - 1
            elif feature == 'inflation_rate' and 'cpi' in data.columns:
                data['inflation_rate'] = data['cpi'] / data['cpi'].shift(1) - 1
            # 添加更多特征提取逻辑
        return data 