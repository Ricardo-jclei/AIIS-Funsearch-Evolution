import os
import logging
from datetime import datetime
from src.data.collectors.market_data_collector import MarketDataCollector
from tqdm import tqdm
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collect_all_data():
    """执行完整的数据收集工作"""
    
    try:
        # 初始化收集器，设置为收集沪深300前50只股票
        collector = MarketDataCollector(data_dir="data", max_stocks=50)
        logger.info("开始数据收集工作...")
        
        # 1. 收集股票列表
        logger.info("收集股票列表...")
        stock_list = collector.collect_stock_list()
        logger.info(f"成功获取{len(stock_list)}只股票的信息")
        
        # 2. 对每只股票收集数据
        for _, stock in tqdm(stock_list.iterrows(), total=len(stock_list), desc="处理股票"):
            symbol = stock['code']
            logger.info(f"\n开始处理股票 {symbol}...")
            
            try:
                # 2.1 收集日线数据
                logger.info(f"收集{symbol}的日线数据...")
                daily_data = collector.collect_daily_data(symbol=symbol)
                logger.info(f"成功获取{len(daily_data)}条日线数据")
                
                # 2.2 收集分钟线数据
                logger.info(f"收集{symbol}的5分钟线数据...")
                minute_data = collector.collect_minute_data(symbol=symbol, period='5')
                logger.info(f"成功获取{len(minute_data)}条分钟线数据")
                
                # 2.3 收集基本面数据
                logger.info(f"收集{symbol}的基本面数据...")
                for report_type in ["资产负债表", "利润表", "现金流量表"]:
                    fundamental_data = collector.collect_fundamental_data(
                        symbol=symbol,
                        report_type=report_type
                    )
                    logger.info(f"成功获取{len(fundamental_data)}条{report_type}数据")
                
                # 2.4 收集市场情绪数据
                logger.info(f"收集{symbol}的市场情绪数据...")
                sentiment_data = collector.collect_market_sentiment(symbol=symbol)
                for data_type, df in sentiment_data.items():
                    logger.info(f"成功获取{len(df)}条{data_type}数据")
                
                # 休息1秒，避免请求过于频繁
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"处理股票{symbol}时出错: {str(e)}")
                continue
        
        # 3. 收集宏观经济指标
        logger.info("\n收集宏观经济指标...")
        try:
            macro_data = collector.collect_macro_indicators()
            for indicator, df in macro_data.items():
                logger.info(f"成功获取{len(df)}条{indicator}数据")
        except Exception as e:
            logger.error(f"收集宏观经济指标时出错: {str(e)}")
        
        logger.info("数据收集工作完成!")
        
    except Exception as e:
        logger.error(f"数据收集过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    collect_all_data() 