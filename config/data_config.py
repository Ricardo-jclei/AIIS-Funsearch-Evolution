import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class DataConfig:
    # Tushare配置
    TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '')
    
    # 数据库配置
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'investment_db')
    }
    
    # 数据收集配置
    DATA_SOURCES = ['tushare', 'akshare']
    UPDATE_FREQUENCY = '1d'  # 数据更新频率
    
    # 数据处理配置
    WINDOW_SIZE = 30  # 时间窗口大小
    FEATURE_COLUMNS = [
        'open', 'high', 'low', 'close', 
        'volume', 'amount', 'turnover'
    ]
    
    # 股票池配置
    STOCK_POOL = {
        'market': 'A股主板',
        'min_price': 5.0,
        'max_price': 100.0,
        'min_volume': 1000000
    }
    
    # 数据存储路径
    DATA_PATH = {
        'raw': 'data/raw',
        'processed': 'data/processed',
        'models': 'data/models'
    } 