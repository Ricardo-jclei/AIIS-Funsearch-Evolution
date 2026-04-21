import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import List, Optional, Dict, Any
import logging
from .base_collector import BaseDataCollector

class MarketDataCollector(BaseDataCollector):
    """市场数据收集器 - 服务于预测-决策双引擎系统"""
    
    def __init__(self, data_dir: str = "data", max_stocks: int = 50):
        """
        初始化收集器
        
        Args:
            data_dir: 数据存储根目录
            max_stocks: 最大收集的股票数量
        """
        super().__init__({"data_dir": data_dir, "max_stocks": max_stocks})
        self.data_dir = data_dir
        self.max_stocks = max_stocks
        self.logger = logging.getLogger(__name__)
        
    def fetch_stock_data(self, stock_code: str, 
                        start_date: str, 
                        end_date: str) -> pd.DataFrame:
        """实现基类的抽象方法：获取股票数据"""
        return self.collect_daily_data(symbol=stock_code,
                                     start_date=start_date,
                                     end_date=end_date)
    
    def fetch_index_data(self, index_code: str,
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
        """实现基类的抽象方法：获取指数数据"""
        return ak.stock_zh_index_daily(symbol=index_code)
    
    def fetch_stock_list(self) -> pd.DataFrame:
        """实现基类的抽象方法：获取股票列表"""
        return self.collect_stock_list()
    
    def collect_stock_list(self) -> pd.DataFrame:
        """收集股票列表，选取沪深300中权重最大的前N只股票"""
        try:
            # 获取沪深300成分股
            hs300 = ak.index_stock_cons_weight_csindex(symbol="000300")
            
            # 打印列名以进行调试
            self.logger.info(f"获取到的数据列名: {hs300.columns.tolist()}")
            
            # 按权重排序并选取前N只（假设权重列名为'权重'）
            stock_list = hs300.sort_values(by='权重', ascending=False).head(self.max_stocks)
            
            # 重命名列以统一格式
            stock_list = stock_list.rename(columns={
                '成分券代码': 'code',
                '成分券名称': 'name',
                '权重': 'weight'
            })
            
            # 保存数据
            save_path = os.path.join(self.data_dir, "raw/market/daily/stock_list.csv")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            stock_list.to_csv(save_path, index=False)
            
            return stock_list
        except Exception as e:
            self.logger.error(f"收集股票列表时出错: {str(e)}")
            raise
            
    def collect_daily_data(self, 
                          symbol: str,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          adjust: str = "qfq") -> pd.DataFrame:
        """收集日线数据，默认收集近3年数据
        
        Args:
            symbol: 股票代码（如：000001）
            start_date: 开始日期（YYYYMMDD），默认为3年前
            end_date: 结束日期（YYYYMMDD），默认为当前日期
            adjust: 复权类型，qfq-前复权，hfq-后复权，None-不复权
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y%m%d")
                
            # 获取日线数据
            daily_data = ak.stock_zh_a_hist(symbol=symbol,
                                          start_date=start_date,
                                          end_date=end_date,
                                          adjust=adjust)
            
            # 保存数据
            save_dir = os.path.join(self.data_dir, f"raw/market/daily/{symbol}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{start_date}_{end_date}.csv")
            daily_data.to_csv(save_path, index=False)
            
            return daily_data
        except Exception as e:
            self.logger.error(f"收集{symbol}日线数据时出错: {str(e)}")
            raise
            
    def collect_minute_data(self,
                           symbol: str,
                           period: str = '1') -> pd.DataFrame:
        """收集分钟线数据，仅保留近3个月数据
        
        Args:
            symbol: 股票代码（如：000001）
            period: 周期（1, 5, 15, 30, 60分钟）
        """
        try:
            # 获取分钟线数据
            minute_data = ak.stock_zh_a_hist_min_em(symbol=symbol, period=period)
            
            # 打印列名以进行调试
            self.logger.info(f"分钟线数据列名: {minute_data.columns.tolist()}")
            
            # 只保留近3个月的数据
            three_months_ago = datetime.now() - timedelta(days=90)
            minute_data['时间'] = pd.to_datetime(minute_data['时间'])
            minute_data = minute_data[minute_data['时间'] >= three_months_ago]
            
            # 重命名列以统一格式
            minute_data = minute_data.rename(columns={
                '时间': 'datetime',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'
            })
            
            # 保存数据
            today = datetime.now().strftime("%Y%m%d")
            save_dir = os.path.join(self.data_dir, f"raw/market/minute/{symbol}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{today}_{period}min.csv")
            minute_data.to_csv(save_path, index=False)
            
            return minute_data
        except Exception as e:
            self.logger.error(f"收集{symbol}分钟线数据时出错: {str(e)}")
            raise
            
    def collect_fundamental_data(self,
                               symbol: str,
                               report_type: str = "资产负债表") -> pd.DataFrame:
        """收集基本面数据，保留近4个季度数据
        
        Args:
            symbol: 股票代码（如：000001）
            report_type: 报表类型（资产负债表、利润表、现金流量表）
        """
        try:
            # 获取财务数据
            if report_type == "资产负债表":
                fundamental_data = ak.stock_financial_report_sina(symbol, "资产负债表")
                save_subdir = "balance_sheet"
            elif report_type == "利润表":
                fundamental_data = ak.stock_financial_report_sina(symbol, "利润表")
                save_subdir = "income_statement"
            elif report_type == "现金流量表":
                fundamental_data = ak.stock_financial_report_sina(symbol, "现金流量表")
                save_subdir = "cash_flow"
            
            # 只保留近4个季度的数据
            fundamental_data = fundamental_data.head(4)
            
            # 保存数据
            save_dir = os.path.join(self.data_dir, f"raw/fundamental/financial_statement/{save_subdir}/{symbol}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{datetime.now().strftime('%Y%m%d')}.csv")
            fundamental_data.to_csv(save_path, index=False)
            
            return fundamental_data
        except Exception as e:
            self.logger.error(f"收集{symbol}基本面数据时出错: {str(e)}")
            raise
            
    def collect_market_sentiment(self,
                               symbol: str,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """收集市场情绪数据，保留近6个月数据
        
        Args:
            symbol: 股票代码（如：000001）
            start_date: 开始日期（YYYYMMDD），默认为6个月前
            end_date: 结束日期（YYYYMMDD），默认为当前日期
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
                
            sentiment_data = {}
            
            # 1. 个股资金流向
            sentiment_data['fund_flow'] = ak.stock_individual_fund_flow(stock=symbol)
            
            # 2. 个股主力资金净流入
            sentiment_data['main_money'] = ak.stock_individual_fund_flow_rank(indicator="今日")
            
            # 保存数据
            for data_type, df in sentiment_data.items():
                save_dir = os.path.join(self.data_dir, f"raw/sentiment/{data_type}/{symbol}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{start_date}_{end_date}.csv")
                df.to_csv(save_path, index=False)
            
            return sentiment_data
        except Exception as e:
            self.logger.error(f"收集{symbol}市场情绪数据时出错: {str(e)}")
            raise
            
    def collect_macro_indicators(self) -> Dict[str, pd.DataFrame]:
        """收集宏观经济指标，保留近5年数据"""
        try:
            macro_data = {}
            
            # 计算5年前的日期
            five_years_ago = (datetime.now() - timedelta(days=5*365)).strftime("%Y%m%d")
            today = datetime.now().strftime("%Y%m%d")
            
            # 1. GDP数据
            macro_data['gdp'] = ak.macro_china_gdp_yearly()
            
            # 2. CPI数据
            macro_data['cpi'] = ak.macro_china_cpi_yearly()
            
            # 3. PPI数据
            macro_data['ppi'] = ak.macro_china_ppi_yearly()
            
            # 4. PMI数据
            macro_data['pmi'] = ak.macro_china_pmi_yearly()
            
            # 对每个数据集进行时间过滤并保存
            for indicator, df in macro_data.items():
                if '年份' in df.columns:
                    df['年份'] = pd.to_datetime(df['年份'].astype(str))
                    df = df[df['年份'] >= pd.to_datetime(five_years_ago)]
                
                save_dir = os.path.join(self.data_dir, f"raw/macro/{indicator}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{today}.csv")
                df.to_csv(save_path, index=False)
            
            return macro_data
        except Exception as e:
            self.logger.error(f"收集宏观经济数据时出错: {str(e)}")
            raise
            
    def collect_industry_data(self,
                            industry_code: str,
                            start_date: str,
                            end_date: str = None) -> pd.DataFrame:
        """收集行业数据
        
        Args:
            industry_code: 行业代码
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD），默认为当前日期
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")
                
            # 获取行业指数数据
            industry_data = ak.stock_board_industry_hist_em(symbol=industry_code,
                                                          start_date=start_date,
                                                          end_date=end_date)
            
            # 保存数据
            save_dir = os.path.join(self.data_dir, "raw/market/industry")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{industry_code}_{start_date}_{end_date}.csv")
            industry_data.to_csv(save_path, index=False)
            
            return industry_data
        except Exception as e:
            self.logger.error(f"收集行业{industry_code}数据时出错: {str(e)}")
            raise 