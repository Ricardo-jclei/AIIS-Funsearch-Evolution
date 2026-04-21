from typing import Dict, Optional, Type
import logging
from .base_processor import BaseProcessor
from .market_processor import MarketProcessor
from .fundamental_processor import FundamentalProcessor
from .sentiment_processor import SentimentProcessor
from .macro_processor import MacroProcessor

class ProcessorFactory:
    """数据处理器工厂类"""
    
    # 处理器类型映射
    PROCESSOR_MAP = {
        'market': MarketProcessor,
        'fundamental': FundamentalProcessor,
        'sentiment': SentimentProcessor,
        'macro': MacroProcessor
    }
    
    def __init__(self, config: Dict):
        """
        初始化工厂类
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._processors = {}  # 存储已创建的处理器实例
        
    def get_processor(self, processor_type: str) -> Optional[BaseProcessor]:
        """
        获取指定类型的数据处理器
        
        Args:
            processor_type: 处理器类型，可选值：'market', 'fundamental', 'sentiment'
            
        Returns:
            对应的数据处理器实例，如果类型不存在则返回None
        """
        # 检查处理器类型是否有效
        if processor_type not in self.PROCESSOR_MAP:
            self.logger.error(f"无效的处理器类型: {processor_type}")
            return None
            
        # 如果处理器已创建，直接返回
        if processor_type in self._processors:
            return self._processors[processor_type]
            
        try:
            # 创建新的处理器实例
            processor_class = self.PROCESSOR_MAP[processor_type]
            processor = processor_class(self.config)
            self._processors[processor_type] = processor
            return processor
        except Exception as e:
            self.logger.error(f"创建处理器 {processor_type} 失败: {str(e)}")
            return None
            
    def process_data(self, data_type: str, data: Dict) -> Optional[Dict]:
        """
        处理数据
        
        Args:
            data_type: 数据类型，可选值：'market', 'fundamental', 'sentiment'
            data: 待处理的数据字典，键为股票代码，值为对应的DataFrame
            
        Returns:
            处理后的数据字典，如果处理失败则返回None
        """
        processor = self.get_processor(data_type)
        if processor is None:
            return None
            
        try:
            processed_data = {}
            for symbol, df in data.items():
                processed_df = processor.process(df)
                processed_data[symbol] = processed_df
            return processed_data
        except Exception as e:
            self.logger.error(f"处理 {data_type} 数据失败: {str(e)}")
            return None 