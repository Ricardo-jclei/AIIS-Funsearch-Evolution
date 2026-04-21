from graphviz import Digraph

def create_ppo_structure():
    dot = Digraph('PPO_Policy_Network', format='png')
    # 调整图片尺寸和DPI
    dot.attr(rankdir='TB', size='8,6', dpi='300')
    # 设置全局字体
    dot.attr('node', fontname='SimHei', fontsize='14')
    dot.attr('edge', fontname='SimHei', fontsize='12')

    # 创建子图来强制节点位置
    with dot.subgraph(name='cluster_0') as c:
        c.attr(rank='same')
        # 输入状态
        c.node('Input', '输入状态\n[state_dim]', 
               shape='rect', style='filled', fillcolor='#E3F2FD',
               fontsize='16', height='0.8', width='2.0')
        # 第一层全连接
        c.node('FC1', '全连接层1\n128单元 + ReLU', 
               shape='rect', style='filled', fillcolor='#BBDEFB',
               fontsize='16', height='0.8', width='2.0')
        # 第二层全连接
        c.node('FC2', '全连接层2\n128单元 + ReLU', 
               shape='rect', style='filled', fillcolor='#90CAF9',
               fontsize='16', height='0.8', width='2.0')

    with dot.subgraph(name='cluster_1') as c:
        c.attr(rank='same')
        # 策略头
        c.node('Policy', '策略头\n(动作分布)', 
               shape='rect', style='filled', fillcolor='#4FC3F7',
               fontsize='16', height='0.8', width='2.0')
        # 价值头
        c.node('Value', '价值头\n(状态价值)', 
               shape='rect', style='filled', fillcolor='#0288D1', fontcolor='white',
               fontsize='16', height='0.8', width='2.0')

    # 连线
    dot.edge('Input', 'FC1', constraint='false')
    dot.edge('FC1', 'FC2', constraint='false')
    dot.edge('FC2', 'Policy', constraint='false')
    dot.edge('FC2', 'Value', constraint='false')

    # 保存图片
    dot.render('output/图4_2_PPO策略网络结构示意图', view=True)

if __name__ == '__main__':
    create_ppo_structure() 