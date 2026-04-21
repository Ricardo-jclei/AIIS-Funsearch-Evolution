from typing import Dict, List
import pandas as pd
import numpy as np
from .base_processor import BaseProcessor

class MarketProcessor(BaseProcessor):
    """市场数据处理类"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        # 读取日线和分钟线配置
        self.daily_cfg = config['data_types']['market']['daily']
        self.minute_cfg = config['data_types']['market']['minute']

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理市场数据，自动识别日线/分钟线"""
        # 判断是日线还是分钟线
        if set(self.daily_cfg['required_columns']).issubset(set(data.columns)):
            cfg = self.daily_cfg
            to_format = '%Y-%m-%d'
        elif set(self.minute_cfg['required_columns']).issubset(set(data.columns)):
            cfg = self.minute_cfg
            # 判断原始字段是否有秒
            sample = data[cfg['date_column']].astype(str).iloc[0]
            if len(sample) == 16:  # 2024-05-24 09:35
                to_format = '%Y-%m-%d %H:%M'
            elif len(sample) == 19:  # 2024-05-24 09:35:00
                to_format = '%Y-%m-%d %H:%M:%S'
            else:
                to_format = '%Y-%m-%d %H:%M'
        else:
            # 自动补齐缺失字段
            if len(set(self.daily_cfg['required_columns']) - set(data.columns)) < len(set(self.minute_cfg['required_columns']) - set(data.columns)):
                cfg = self.daily_cfg
                to_format = '%Y-%m-%d'
            else:
                cfg = self.minute_cfg
                to_format = '%Y-%m-%d %H:%M'
            missing = set(cfg['required_columns']) - set(data.columns)
            if missing:
                self.logger.warning(f"市场数据缺少必需的列: {missing}，自动补齐为NaN")
                for col in missing:
                    data[col] = np.nan
        # 转换日期格式
        data = self.convert_date_format(
            data, 
            cfg['date_column'],
            self.config.get('date_format', '%Y-%m-%d'),
            to_format
        )
        # 处理缺失值
        data = self.handle_missing_values(data, strategy='fill')
        # 计算技术指标
        data = self.add_technical_indicators(data, cfg)
        return data

    def add_technical_indicators(self, data: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
        """添加技术指标，适配中英文列名，支持日线/分钟线"""
        ma_windows = self.config.get('feature_engineering', {}).get('technical_indicators', {}).get('ma', [5, 10, 20, 30, 60])
        rsi_windows = self.config.get('feature_engineering', {}).get('technical_indicators', {}).get('rsi', [6, 12, 24])
        macd_params = self.config.get('feature_engineering', {}).get('technical_indicators', {}).get('macd', {'fast': 12, 'slow': 26, 'signal': 9})
        bb_params = self.config.get('feature_engineering', {}).get('technical_indicators', {}).get('bollinger_bands', {'window': 20, 'std': 2})
        close_col = cfg['price_columns'][1]  # "收盘"或"close"
        # 计算移动平均线
        for window in ma_windows:
            data[f'MA{window}'] = data[close_col].rolling(window=window).mean()
        # 计算RSI
        for window in rsi_windows:
            delta = data[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            data[f'RSI{window}'] = 100 - (100 / (1 + rs))
        # 计算MACD
        exp1 = data[close_col].ewm(span=macd_params['fast'], adjust=False).mean()
        exp2 = data[close_col].ewm(span=macd_params['slow'], adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=macd_params['signal'], adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        # 计算布林带
        bb_middle = data[close_col].rolling(window=bb_params['window']).mean()
        bb_std = data[close_col].rolling(window=bb_params['window']).std()
        data['BB_Upper'] = bb_middle + bb_params['std'] * bb_std
        data['BB_Lower'] = bb_middle - bb_params['std'] * bb_std
        return data 