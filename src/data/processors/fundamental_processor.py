from typing import Dict
import pandas as pd
import numpy as np
from .base_processor import BaseProcessor

class FundamentalProcessor(BaseProcessor):
    """基本面数据处理类，自动适配表头"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        # 不再强制required_columns，自动适配
        self.required_columns = None
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理基本面数据，自动适配表头"""
        # 去除列名空格
        data.columns = data.columns.str.strip()
        # 自动查找日期字段
        date_col = None
        for candidate in ['报告日', '报表期', '日期', 'Period', 'Date']:
            if candidate in data.columns:
                date_col = candidate
                break
        if date_col is None:
            self.logger.warning("基本面数据未找到日期字段，无法处理！")
            return data
        # 转换日期格式
        data = self.convert_date_format(
            data, 
            date_col,
            self.config.get('date_format', '%Y-%m-%d'),
            '%Y-%m-%d'
        )
        # 处理缺失值
        data = self.handle_missing_values(data, strategy='fill')
        # 计算财务比率（仅对存在的字段计算）
        data = self.calculate_financial_ratios(data)
        return data
        
    def calculate_financial_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算财务比率，自动判断字段是否存在"""
        # 盈利能力指标
        if all(col in data.columns for col in ['净利润', '资产总计', '归属于母公司股东权益合计']):
            data['ROE'] = data['净利润'] / data['归属于母公司股东权益合计']
            data['ROA'] = data['净利润'] / data['资产总计']
        # 成长性指标
        if all(col in data.columns for col in ['营业收入', '净利润']):
            data['营收增长率'] = data['营业收入'].pct_change()
            data['净利润增长率'] = data['净利润'].pct_change()
        # 杠杆指标
        if all(col in data.columns for col in ['负债合计', '资产总计', '归属于母公司股东权益合计']):
            data['资产负债率'] = data['负债合计'] / data['资产总计']
            data['权益乘数'] = data['资产总计'] / data['归属于母公司股东权益合计']
        # 效率指标
        if all(col in data.columns for col in ['营业收入', '资产总计']):
            data['总资产周转率'] = data['营业收入'] / data['资产总计']
        # 现金流指标
        if all(col in data.columns for col in ['经营活动产生的现金流量净额', '净利润']):
            data['经营现金流/净利润'] = data['经营活动产生的现金流量净额'] / data['净利润']
        return data 