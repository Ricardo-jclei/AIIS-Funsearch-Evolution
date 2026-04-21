from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class BaseProcessor(ABC):
    """基础数据处理器类，定义数据处理的基本接口"""
    
    def __init__(self, config: Dict):
        """
        初始化数据处理器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers = {}  # 存储不同特征的标准化器
        
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理数据的主方法
        
        Args:
            data: 输入数据，可以是DataFrame或numpy数组
            
        Returns:
            处理后的数据
        """
        pass
    
    def validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """验证数据是否包含必需的列"""
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            self.logger.error(f"缺少必需的列: {missing_columns}")
            return False
        return True
    
    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """处理缺失值"""
        if strategy == 'drop':
            return data.dropna()
        elif strategy == 'fill':
            return data.ffill().bfill()
        else:
            return data.fillna(0)
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        df = data.copy()
        
        # 移动平均线
        for window in [5, 10, 20, 30, 60]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            
        # 相对强弱指标 (RSI)
        df['price_diff'] = df['close'].diff()
        df['gain'] = df['price_diff'].clip(lower=0)
        df['loss'] = -df['price_diff'].clip(upper=0)
        
        for window in [6, 12, 24]:
            avg_gain = df['gain'].rolling(window=window).mean()
            avg_loss = df['loss'].rolling(window=window).mean()
            rs = avg_gain / avg_loss
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['signal_line']
        
        # 布林带
        for window in [20]:
            df[f'bb_middle_{window}'] = df['close'].rolling(window=window).mean()
            df[f'bb_upper_{window}'] = (df[f'bb_middle_{window}'] + 
                                      2 * df['close'].rolling(window=window).std())
            df[f'bb_lower_{window}'] = (df[f'bb_middle_{window}'] - 
                                      2 * df['close'].rolling(window=window).std())
            
        # 成交量相关指标
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        
        # 删除中间计算列
        df = df.drop(['price_diff', 'gain', 'loss'], axis=1)
        
        return df
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算收益率"""
        df = data.copy()
        
        # 日收益率
        df['daily_return'] = df['close'].pct_change()
        
        # 累积收益率
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        
        # 波动率（20日滚动标准差）
        df['volatility'] = df['daily_return'].rolling(window=20).std()
        
        return df 

    def remove_outliers(self, df: pd.DataFrame, 
                       columns: List[str], 
                       n_std: float = 3) -> pd.DataFrame:
        """移除异常值
        
        Args:
            df: 输入数据框
            columns: 需要处理的列
            n_std: 标准差的倍数，超过均值±n_std倍标准差的视为异常值
        """
        df_clean = df.copy()
        for col in columns:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            df_clean[col] = df_clean[col].clip(
                lower=mean - n_std * std,
                upper=mean + n_std * std
            )
        return df_clean
    
    def fill_missing_values(self, df: pd.DataFrame, 
                          methods: Dict[str, str]) -> pd.DataFrame:
        """填充缺失值
        
        Args:
            df: 输入数据框
            methods: 填充方法字典，键为列名，值为填充方法
                    支持：'ffill'（前向填充）, 'bfill'（后向填充）, 
                    'mean'（均值）, 'median'（中位数）, 'zero'（零）
        """
        df_filled = df.copy()
        for col, method in methods.items():
            if method == 'ffill':
                df_filled[col] = df_filled[col].fillna(method='ffill')
            elif method == 'bfill':
                df_filled[col] = df_filled[col].fillna(method='bfill')
            elif method == 'mean':
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
            elif method == 'median':
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            elif method == 'zero':
                df_filled[col] = df_filled[col].fillna(0)
        return df_filled
    
    def normalize_data(self, df: pd.DataFrame, 
                      columns: List[str],
                      method: str = 'standard',
                      fit: bool = True) -> pd.DataFrame:
        """数据标准化
        
        Args:
            df: 输入数据框
            columns: 需要标准化的列
            method: 标准化方法，'standard'或'minmax'
            fit: 是否需要拟合scaler（训练集设为True，测试集设为False）
        """
        df_norm = df.copy()
        
        for col in columns:
            if fit:
                if method == 'standard':
                    scaler = StandardScaler()
                else:  # minmax
                    scaler = MinMaxScaler()
                df_norm[col] = scaler.fit_transform(
                    df_norm[col].values.reshape(-1, 1)
                ).flatten()
                self.scalers[col] = scaler
            else:
                if col in self.scalers:
                    df_norm[col] = self.scalers[col].transform(
                        df_norm[col].values.reshape(-1, 1)
                    ).flatten()
                else:
                    self.logger.warning(f"列 {col} 没有对应的scaler")
        
        return df_norm
    
    def calculate_returns(self, df: pd.DataFrame,
                         price_col: str = 'close',
                         periods: List[int] = [1, 5, 20]) -> pd.DataFrame:
        """计算收益率
        
        Args:
            df: 输入数据框
            price_col: 价格列名
            periods: 收益率周期列表
        """
        df_returns = df.copy()
        
        # 计算简单收益率
        for period in periods:
            col_name = f'return_{period}d'
            df_returns[col_name] = (
                df_returns[price_col].pct_change(period)
            )
        
        # 计算对数收益率
        for period in periods:
            col_name = f'log_return_{period}d'
            df_returns[col_name] = (
                np.log(df_returns[price_col]) - 
                np.log(df_returns[price_col].shift(period))
            )
        
        return df_returns
    
    def add_time_features(self, df: pd.DataFrame, 
                         date_col: str = 'date') -> pd.DataFrame:
        """添加时间特征
        
        Args:
            df: 输入数据框
            date_col: 日期列名
        """
        df_time = df.copy()
        
        # 确保日期列的类型是datetime
        df_time[date_col] = pd.to_datetime(df_time[date_col])
        
        # 提取时间特征
        df_time['year'] = df_time[date_col].dt.year
        df_time['month'] = df_time[date_col].dt.month
        df_time['day'] = df_time[date_col].dt.day
        df_time['weekday'] = df_time[date_col].dt.weekday
        df_time['quarter'] = df_time[date_col].dt.quarter
        
        # 是否月初/月末
        df_time['is_month_start'] = df_time[date_col].dt.is_month_start.astype(int)
        df_time['is_month_end'] = df_time[date_col].dt.is_month_end.astype(int)
        
        # 是否季初/季末
        df_time['is_quarter_start'] = df_time[date_col].dt.is_quarter_start.astype(int)
        df_time['is_quarter_end'] = df_time[date_col].dt.is_quarter_end.astype(int)
        
        return df_time
    
    def process_stock_data(self, df: pd.DataFrame, 
                          is_training: bool = True) -> pd.DataFrame:
        """处理股票数据的完整流程
        
        Args:
            df: 输入的股票数据
            is_training: 是否为训练数据
        """
        try:
            # 1. 填充缺失值
            fill_methods = {
                'open': 'ffill',
                'high': 'ffill',
                'low': 'ffill',
                'close': 'ffill',
                'volume': 'zero',
                'amount': 'zero'
            }
            df_processed = self.fill_missing_values(df, fill_methods)
            
            # 2. 移除异常值
            price_columns = ['open', 'high', 'low', 'close']
            df_processed = self.remove_outliers(
                df_processed, price_columns, n_std=3
            )
            
            # 3. 计算收益率
            df_processed = self.calculate_returns(
                df_processed, 'close', [1, 5, 10, 20, 60]
            )
            
            # 4. 添加时间特征
            df_processed = self.add_time_features(df_processed)
            
            # 5. 标准化数据
            normalize_columns = [
                'open', 'high', 'low', 'close', 
                'volume', 'amount'
            ]
            df_processed = self.normalize_data(
                df_processed, 
                normalize_columns,
                method='standard',
                fit=is_training
            )
            
            return df_processed
            
        except Exception as e:
            self.logger.error(f"处理数据时出错: {str(e)}")
            raise
            
    def prepare_sequence_data(self, df: pd.DataFrame,
                            sequence_length: int,
                            target_col: str = 'return_1d',
                            feature_cols: Optional[List[str]] = None) -> tuple:
        """准备序列数据，用于LSTM模型
        
        Args:
            df: 输入数据框
            sequence_length: 序列长度
            target_col: 目标列名
            feature_cols: 特征列名列表
        
        Returns:
            (X, y): X为特征序列，y为目标值
        """
        if feature_cols is None:
            feature_cols = [
                'open', 'high', 'low', 'close', 
                'volume', 'amount',
                'return_1d', 'return_5d', 'return_20d'
            ]
        
        # 准备特征数据
        data = df[feature_cols].values
        X, y = [], []
        
        for i in range(len(df) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(df[target_col].iloc[i + sequence_length])
        
        return np.array(X), np.array(y)

    def save_processed_data(self, data: Union[pd.DataFrame, np.ndarray], 
                          filepath: str) -> None:
        """
        保存处理后的数据
        
        Args:
            data: 处理后的数据
            filepath: 保存路径
        """
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        elif isinstance(data, np.ndarray):
            np.save(filepath, data) 

    def convert_date_format(self, data: pd.DataFrame, date_column: str, from_format: str, to_format: str) -> pd.DataFrame:
        """增强：自动推断常见日期格式，优先支持8位数字（如20250331）"""
        if date_column in data.columns:
            try:
                # 先尝试原始from_format
                data[date_column] = pd.to_datetime(data[date_column], format=from_format, errors='raise').dt.strftime(to_format)
            except Exception:
                # 尝试常见格式
                tried = False
                for fmt in ['%Y%m%d', '%Y/%m/%d', '%Y.%m.%d', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M']:
                    try:
                        data[date_column] = pd.to_datetime(data[date_column], format=fmt, errors='raise').dt.strftime(to_format)
                        tried = True
                        break
                    except Exception:
                        continue
                if not tried:
                    # 针对8位数字字符串或整数
                    mask = data[date_column].astype(str).str.match(r'^\d{8}$')
                    if mask.any():
                        try:
                            data.loc[mask, date_column] = pd.to_datetime(data.loc[mask, date_column], format='%Y%m%d', errors='raise').dt.strftime(to_format)
                        except Exception:
                            pass
                    # 最后宽容推断
                    data[date_column] = pd.to_datetime(data[date_column], errors='coerce').dt.strftime(to_format)
        return data 