# 导入必要的库
import os  # 操作系统接口
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 可视化库
import gymnasium as gym
import random  # 随机函数库
from collections import defaultdict # 用于Q表格存储


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_discrete_position(pos):
    if pos < 0:
        # 起点附近划分
        bucket = int((pos + 1.2) / (1.2) * 30)  # [-1.2, 0)映射
    else:
        # 终点附近划分
        bucket = 30 + int(pos / 0.6 * 10)  # [0, 0.6]映射
    return min(bucket, 39)  # 确保不超过最大索引

def get_discrete_velocity(vel):
    bucket = int((vel + 0.07) / 0.14 * 40)  # [-0.07, 0.07]映射到40个区间
    return min(bucket, 39)

# Q-table的状态近似函数
def get_discrete_state(state):
    pos, vel = state
    pos_bucket = get_discrete_position(pos)
    vel_bucket = get_discrete_velocity(vel)
    return (pos_bucket, vel_bucket)

# 主函数
def main():
    # 创建 MountainCar-v0 环境
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    seed = 42  # 设置随机种子
    np.random.seed(seed)  # 设置 Numpy 随机种子
    random.seed(seed)  # 设置 Python 随机种子
    # 默认将Action 0,1,2的价值初始化为0
    Q = defaultdict(lambda: [0, 0, 0])

    # 参数
    episodes = 3000
    epsilon = 1.0 # ε-greedy策略
    min_epsilon = 0.1
    epsilon_decay = 0.995
    score_list = []
    moving_avg_score = []
    lr = 0.8
    gamma = 0.9
    ma_window = 10

    for i in range(episodes):
        state, _ = env.reset()
        state = get_discrete_state(state)
        score = 0
        flag = True
        while flag:
            # 行动
            # ε-greedy策略
            if random.random() < epsilon:
                action = env.action_space.sample()  # 随机探索
            else:
                action = np.argmax(Q[state])  # 利用已有知识
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = get_discrete_state(next_state)

            # 更新Q-table
            Q[state][action] = Q[state][action] + lr * (reward + gamma * max(Q[next_state]) - Q[state][action])
            score += reward
            state = next_state

            # 判断是否结束
            if done:
                score_list.append(score)
                moving_avg = np.mean(score_list[-ma_window:])  # 计算最近 ma_window 个回合的滑动平均回报
                moving_avg_score.append(moving_avg)
                # print('episode:', i, 'score:', score, 'max:', max(score_list))
                break
        # 更新探索率
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    env.close()
    print("Max Score: ", max(score_list))
    # 绘图
    plt.figure(figsize=(10, 5))
    # 两条曲线
    plt.plot(score_list, label='score')  # 绘制每个回合的回报
    plt.plot(moving_avg_score, label='Moving Average (10)')  # 绘制10个回合的滑动平均回报
    # 添加成功阈值线
    success_threshold = -110  # MountainCar成功标准
    plt.axhline(y=success_threshold, color='r', linestyle='--', label='Success Threshold')
    plt.xlabel('Episode')  # x轴标签：回合数
    plt.ylabel('Score')  # y轴标签：回报
    plt.title('Q-Learning Training Return')  # 图表标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()  # 展示图表

# 运行主函数
if __name__ == '__main__':
    main()

