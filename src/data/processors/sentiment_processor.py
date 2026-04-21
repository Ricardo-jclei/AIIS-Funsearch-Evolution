from typing import Dict, List
import pandas as pd
import numpy as np
from .base_processor import BaseProcessor

class SentimentProcessor(BaseProcessor):
    """情绪数据处理类"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        # 只处理fund_flow数据，字段与原始表头一致
        self.required_columns = [
            '日期', '收盘价', '涨跌幅',
            '主力净流入-净额', '主力净流入-净占比',
            '超大单净流入-净额', '超大单净流入-净占比',
            '大单净流入-净额', '大单净流入-净占比',
            '中单净流入-净额', '中单净流入-净占比',
            '小单净流入-净额', '小单净流入-净占比'
        ]
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理情绪数据，仅处理fund_flow"""
        # 只保留fund_flow相关字段
        data = data[[col for col in self.required_columns if col in data.columns]].copy()
        # 检查缺失字段
        missing = set(self.required_columns) - set(data.columns)
        if missing:
            self.logger.warning(f"情绪数据缺少必需的列: {missing}，自动补齐为NaN")
            for col in missing:
                data[col] = np.nan
        # 转换日期格式
        data = self.convert_date_format(
            data, 
            '日期',
            self.config.get('date_format', '%Y-%m-%d'),
            '%Y-%m-%d'
        )
        # 处理缺失值
        data = self.handle_missing_values(data, strategy='fill')
        # 计算情绪指标
        data = self.calculate_sentiment_indicators(data)
        return data
        
    def calculate_sentiment_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算情绪指标"""
        # 计算总净流入
        data['总净流入'] = (
            data['主力净流入-净额'] + 
            data['超大单净流入-净额'] + 
            data['大单净流入-净额'] + 
            data['中单净流入-净额'] + 
            data['小单净流入-净额']
        )
        
        # 计算资金集中度
        data['资金集中度'] = (
            (data['主力净流入-净额'] + data['超大单净流入-净额']) / 
            (data['总净流入'].abs() + 1e-6)  # 避免除以0
        )
        
        # 计算主力控盘度
        data['主力控盘度'] = (
            (data['主力净流入-净额'] + data['超大单净流入-净额'] + data['大单净流入-净额']) / 
            (data['总净流入'].abs() + 1e-6)
        )
        
        # 计算资金流向趋势（5日移动平均）
        data['资金流向趋势'] = data['总净流入'].rolling(window=5).mean()
        
        # 计算标准化后的资金流向
        data['标准化资金流向'] = (
            (data['总净流入'] - data['总净流入'].mean()) / 
            (data['总净流入'].std() + 1e-6)
        )
        
        return data 