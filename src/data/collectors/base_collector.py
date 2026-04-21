from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
import logging

class BaseDataCollector(ABC):
    """数据收集器基类"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def fetch_stock_data(self, stock_code: str, 
                        start_date: str, 
                        end_date: str) -> pd.DataFrame:
        """获取股票数据"""
        pass
    
    @abstractmethod
    def fetch_index_data(self, index_code: str,
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
        """获取指数数据"""
        pass
    
    @abstractmethod
    def fetch_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        pass
    
    def validate_dates(self, start_date: str, end_date: str) -> bool:
        """验证日期格式"""
        try:
            start = datetime.strptime(start_date, '%Y%m%d')
            end = datetime.strptime(end_date, '%Y%m%d')
            assert start <= end
            return True
        except Exception as e:
            self.logger.error(f"日期验证失败: {str(e)}")
            raise ValueError("日期格式错误或起始日期大于结束日期")
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        column_mapping = {
            'trade_date': 'date',
            'vol': 'volume',
            'ts_code': 'code',
            'change': 'price_change',
            'pct_chg': 'price_change_pct'
        }
        
        return df.rename(columns=column_mapping)
    
    def handle_error(self, error: Exception, context: str):
        """统一错误处理"""
        error_msg = f"{context}: {str(error)}"
        self.logger.error(error_msg)
        raise Exception(error_msg) 