from graphviz import Digraph

def create_lstm_structure():
    dot = Digraph('LSTM_Feature_Extractor', format='png')
    # 调整图片尺寸和DPI
    dot.attr(rankdir='TB', size='8,6', dpi='300')  # 改为上下布局
    # 设置全局字体
    dot.attr('node', fontname='SimHei', fontsize='14')
    dot.attr('edge', fontname='SimHei', fontsize='12')

    # 创建子图来强制节点位置
    with dot.subgraph(name='cluster_0') as c:
        c.attr(rank='same')  # 强制节点在同一行
        # 输入
        c.node('Input', '输入张量\n[batch, window, asset, factor]', 
               shape='rect', style='filled', fillcolor='#E3F2FD',
               fontsize='16', height='0.8', width='2.5')
        # LSTM层1
        c.node('LSTM1', 'LSTM层1\n128单元', 
               shape='rect', style='filled', fillcolor='#BBDEFB',
               fontsize='16', height='0.8', width='2.0')
        c.node('Dropout1', 'Dropout\np=0.3', 
               shape='rect', style='dashed', fillcolor='#E1F5FE',
               fontsize='16', height='0.6', width='1.5')
        # LSTM层2
        c.node('LSTM2', 'LSTM层2\n128单元', 
               shape='rect', style='filled', fillcolor='#90CAF9',
               fontsize='16', height='0.8', width='2.0')
        c.node('Dropout2', 'Dropout\np=0.3', 
               shape='rect', style='dashed', fillcolor='#E1F5FE',
               fontsize='16', height='0.6', width='1.5')

    with dot.subgraph(name='cluster_1') as c:
        c.attr(rank='same')  # 强制节点在同一行
        # LSTM层3
        c.node('LSTM3', 'LSTM层3\n128单元', 
               shape='rect', style='filled', fillcolor='#64B5F6',
               fontsize='16', height='0.8', width='2.0')
        c.node('Dropout3', 'Dropout\np=0.3', 
               shape='rect', style='dashed', fillcolor='#E1F5FE',
               fontsize='16', height='0.6', width='1.5')
        # Flatten
        c.node('Flatten', 'Flatten', 
               shape='rect', style='filled', fillcolor='#B3E5FC',
               fontsize='16', height='0.6', width='1.5')
        # 全连接层
        c.node('Dense', '全连接层\n(Dense)', 
               shape='rect', style='filled', fillcolor='#4FC3F7',
               fontsize='16', height='0.8', width='2.0')
        # PCA降维
        c.node('PCA', 'PCA降维\n(主成分分析)', 
               shape='rect', style='filled', fillcolor='#0288D1', fontcolor='white',
               fontsize='16', height='0.8', width='2.0')
        # 输出
        c.node('Output', '降维后特征向量\n[batch, reduced_dim]', 
               shape='rect', style='filled', fillcolor='#B2EBF2',
               fontsize='16', height='0.8', width='2.5')

    # 连线（使用折线）
    dot.edge('Input', 'LSTM1', constraint='false')
    dot.edge('LSTM1', 'Dropout1', constraint='false')
    dot.edge('Dropout1', 'LSTM2', constraint='false')
    dot.edge('LSTM2', 'Dropout2', constraint='false')
    dot.edge('Dropout2', 'LSTM3', constraint='false')
    dot.edge('LSTM3', 'Dropout3', constraint='false')
    dot.edge('Dropout3', 'Flatten', constraint='false')
    dot.edge('Flatten', 'Dense', constraint='false')
    dot.edge('Dense', 'PCA', constraint='false')
    dot.edge('PCA', 'Output', constraint='false')

    # 可选：标注自动化与存储
    dot.node('Save', '自动保存\n中间结果', 
             shape='note', style='filled', fillcolor='#FFF9C4',
             fontsize='14', height='0.6', width='1.8')
    dot.edge('PCA', 'Save', style='dotted')

    # 保存图片
    dot.render('output/图4_1_LSTM结构示意图', view=True)

if __name__ == '__main__':
    create_lstm_structure() 