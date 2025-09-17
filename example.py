import numpy as np

# 定义图结构 (邻接矩阵)
# 节点: 0-A, 1-B, 2-C, 3-D, 4-E
graph = np.array([
    [0, 1, 1, 0, 0],  # A(0) -> B(1), C(2)
    [1, 0, 1, 1, 0],  # B(1) -> A(0), C(2), D(3)
    [1, 1, 0, 0, 1],  # C(2) -> A(0), B(1), E(4)
    [0, 1, 0, 0, 1],  # D(3) -> B(1), E(4)
    [0, 0, 1, 1, 0]  # E(4) -> C(2), D(3)
])

# 超参数
alpha = 0.1  # 学习率
gamma = 0.6  # 折扣因子
epsilon = 0.1  # 探索率
episodes = 1000  # 训练轮数

# 初始化Q-table (状态=当前节点，动作=下一个节点)
num_nodes = 5
Q = np.zeros((num_nodes, num_nodes))

# 定义起点和终点
src = 0  # A
dst = 4  # E

# 训练过程
for _ in range(episodes):
    current_state = src
    while current_state != dst:
        # 获取当前节点的所有可能动作（邻居节点）
        possible_actions = np.where(graph[current_state] == 1)[0]

        # 探索或利用
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(possible_actions)
        else:
            # 仅从可达节点中选择Q值最大的动作
            action = possible_actions[np.argmax(Q[current_state, possible_actions])]

        # 计算奖励
        reward = -1  # 每走一步惩罚
        if action == dst:
            reward = 100  # 到达终点奖励

        # 更新Q值
        old_value = Q[current_state, action]
        next_max = np.max(Q[action, np.where(graph[action] == 1)[0]]) if action != dst else 0
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        Q[current_state, action] = new_value

        current_state = action

# 测试最优路径
current_state = src
path = [current_state]
while current_state != dst:
    possible_actions = np.where(graph[current_state] == 1)[0]
    action = possible_actions[np.argmax(Q[current_state, possible_actions])]
    path.append(action)
    current_state = action

print("最优路径:", [chr(65 + n) for n in path])  # 正确输出: A -> C -> E