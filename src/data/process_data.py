import os
import pandas as pd
from src.data.processors.processor_factory import ProcessorFactory
import json
import numpy as np

# 加载配置文件
with open('config/data_processing_config.json', 'r') as f:
    config = json.load(f)

# 初始化处理器工厂
factory = ProcessorFactory(config)

# 定义数据目录
raw_data_dir = 'data/raw'
processed_data_dir = 'data/processed'
os.makedirs(processed_data_dir, exist_ok=True)

# 处理市场数据
def process_market():
    # 日线
    market_daily_dir = os.path.join(raw_data_dir, 'market', 'daily')
    for stock_code in os.listdir(market_daily_dir):
        stock_path = os.path.join(market_daily_dir, stock_code)
        if os.path.isdir(stock_path) and stock_code.isdigit() and len(stock_code) == 6:
            data = {}
            for file in os.listdir(stock_path):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(stock_path, file))
                    df.columns = df.columns.str.strip()
                    symbol = file.split('.')[0]
                    data[symbol] = df
            processed_data = factory.process_data('market', data)
            out_dir = os.path.join(processed_data_dir, stock_code, 'market', 'daily')
            os.makedirs(out_dir, exist_ok=True)
            if processed_data:
                for symbol, df in processed_data.items():
                    df.to_csv(os.path.join(out_dir, f'{symbol}_processed.csv'), index=False)
    # 分钟线
    market_minute_dir = os.path.join(raw_data_dir, 'market', 'minute')
    for stock_code in os.listdir(market_minute_dir):
        stock_path = os.path.join(market_minute_dir, stock_code)
        if os.path.isdir(stock_path) and stock_code.isdigit() and len(stock_code) == 6:
            data = {}
            for file in os.listdir(stock_path):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(stock_path, file))
                    df.columns = df.columns.str.strip()
                    symbol = file.split('.')[0]
                    data[symbol] = df
            processed_data = factory.process_data('market', data)
            out_dir = os.path.join(processed_data_dir, stock_code, 'market', 'minute')
            os.makedirs(out_dir, exist_ok=True)
            if processed_data:
                for symbol, df in processed_data.items():
                    df.to_csv(os.path.join(out_dir, f'{symbol}_processed.csv'), index=False)

# 处理基本面数据
def process_fundamental():
    base_dir = os.path.join(raw_data_dir, 'fundamental', 'financial_statement')
    for ftype in os.listdir(base_dir):
        type_path = os.path.join(base_dir, ftype)
        if os.path.isdir(type_path):
            for stock_code in os.listdir(type_path):
                stock_path = os.path.join(type_path, stock_code)
                if os.path.isdir(stock_path) and stock_code.isdigit() and len(stock_code) == 6:
                    data = {}
                    for file in os.listdir(stock_path):
                        if file.endswith('.csv'):
                            df = pd.read_csv(os.path.join(stock_path, file))
                            df.columns = df.columns.str.strip()
                            symbol = file.split('.')[0]
                            data[symbol] = df
                    processed_data = factory.process_data('fundamental', data)
                    out_dir = os.path.join(processed_data_dir, stock_code, 'fundamental', ftype)
                    os.makedirs(out_dir, exist_ok=True)
                    if processed_data:
                        for symbol, df in processed_data.items():
                            df.to_csv(os.path.join(out_dir, f'{symbol}_processed.csv'), index=False)

# 处理情绪数据
def process_sentiment():
    base_dir = os.path.join(raw_data_dir, 'sentiment')
    for stype in os.listdir(base_dir):
        type_path = os.path.join(base_dir, stype)
        if os.path.isdir(type_path):
            for stock_code in os.listdir(type_path):
                stock_path = os.path.join(type_path, stock_code)
                if os.path.isdir(stock_path) and stock_code.isdigit() and len(stock_code) == 6:
                    data = {}
                    for file in os.listdir(stock_path):
                        if file.endswith('.csv'):
                            df = pd.read_csv(os.path.join(stock_path, file))
                            df.columns = df.columns.str.strip()
                            symbol = file.split('.')[0]
                            data[symbol] = df
                    processed_data = factory.process_data('sentiment', data)
                    out_dir = os.path.join(processed_data_dir, stock_code, 'sentiment', stype)
                    os.makedirs(out_dir, exist_ok=True)
                    if processed_data:
                        for symbol, df in processed_data.items():
                            df.to_csv(os.path.join(out_dir, f'{symbol}_processed.csv'), index=False)

# 处理宏观数据
def process_macro():
    macro_dir = os.path.join(raw_data_dir, 'macro')
    for indicator in os.listdir(macro_dir):
        indicator_path = os.path.join(macro_dir, indicator)
        if os.path.isdir(indicator_path):
            for file in os.listdir(indicator_path):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(indicator_path, file))
                    df.columns = df.columns.str.strip()
                    processed_data = factory.process_data('macro', {indicator: df})
                    out_dir = os.path.join(processed_data_dir, 'macro', indicator)
                    os.makedirs(out_dir, exist_ok=True)
                    if processed_data:
                        for symbol, df in processed_data.items():
                            df.to_csv(os.path.join(out_dir, f'{symbol}_processed.csv'), index=False)

def align_fundamental_to_market(df_fund, df_market, date_col_fund, date_col_market):
    df_fund = df_fund.copy()
    df_market = df_market.copy()
    # 清理所有可能的tmp列
    for tmp_col in ['tmp', 'tmp_x', 'tmp_y']:
        if tmp_col in df_fund.columns:
            df_fund = df_fund.drop(columns=[tmp_col])
        if tmp_col in df_market.columns:
            df_market = df_market.drop(columns=[tmp_col])
    df_fund[date_col_fund] = pd.to_datetime(df_fund[date_col_fund])
    df_market[date_col_market] = pd.to_datetime(df_market[date_col_market])
    df_fund = df_fund.sort_values(date_col_fund)
    df_market = df_market.sort_values(date_col_market)
    df_fund['tmp'] = 1
    df_market['tmp'] = 1
    df = pd.merge_asof(df_market, df_fund, left_on=date_col_market, right_on=date_col_fund, direction='backward')
    # 再次清理所有tmp相关列
    for tmp_col in ['tmp', 'tmp_x', 'tmp_y']:
        if tmp_col in df.columns:
            df = df.drop(columns=[tmp_col])
    return df

stock_list = ['600519', '600030', '600036', '601318', '601988']
for code in stock_list:
    base_dir = f'data/processed/{code}'
    # 行情
    market_path = f'{base_dir}/market/daily/20220425_20250424_processed.csv'
    df_market = pd.read_csv(market_path)
    # 基本面
    bs_path = f'{base_dir}/fundamental/balance_sheet/20250424_processed.csv'
    is_path = f'{base_dir}/fundamental/income_statement/20250424_processed.csv'
    cf_path = f'{base_dir}/fundamental/cash_flow/20250424_processed.csv'
    df_bs = pd.read_csv(bs_path)
    df_is = pd.read_csv(is_path)
    df_cf = pd.read_csv(cf_path)
    # 情绪
    sent_path = f'{base_dir}/sentiment/fund_flow/20241026_20250424_processed.csv'
    df_sent = pd.read_csv(sent_path)
    df_sent['日期'] = pd.to_datetime(df_sent['日期'])
    # 对齐
    df = df_market.copy()
    df = align_fundamental_to_market(df_bs, df, '报告日', '日期')
    df = align_fundamental_to_market(df_is, df, '报告日', '日期')
    df = align_fundamental_to_market(df_cf, df, '报告日', '日期')
    df = df.merge(df_sent, on='日期', how='left')
    # 选取特征
    feature_cols = [
        '开盘','收盘','最高','最低','成交量','成交额','振幅','涨跌幅','涨跌额','换手率',
        'MA5','MA10','MA20','MA30','MA60','RSI6','RSI12','RSI24','MACD','MACD_Signal','MACD_Hist','BB_Upper','BB_Lower',
        '资产总计','负债合计','归属于母公司股东权益合计','资产负债率','权益乘数',
        '营业收入','净利润','营业总成本','营业利润','利润总额','净利润增长率','营收增长率',
        '经营活动产生的现金流量净额','投资活动产生的现金流量净额','筹资活动产生的现金流量净额',
        '主力净流入-净额','主力净流入-净占比','超大单净流入-净额','大单净流入-净额','中单净流入-净额','小单净流入-净额',
        '总净流入','资金集中度','主力控盘度','资金流向趋势','标准化资金流向'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]
    # 归一化
    df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / (df[feature_cols].std() + 1e-8)
    # 填充所有NaN为0
    df[feature_cols] = df[feature_cols].fillna(0)
    # 保存
    df_out = df[['日期'] + feature_cols]
    df_out.to_csv(f'{base_dir}/multi_factor.csv', index=False)
    print(f"[特征融合] {code} 完成，保存至 {base_dir}/multi_factor.csv")

if __name__ == '__main__':
    process_market()
    process_fundamental()
    process_sentiment()
    process_macro()
    print('Data processing complete.') 