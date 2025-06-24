# 导入必要的库
import os  # 操作系统接口
import math  # 数学函数库，用于计算 epsilon 衰减
import torch  # PyTorch 主库
import torch.nn.functional as F  # 提供激活函数、损失函数等
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 可视化库
import gymnasium as gym
import random  # 随机函数库
from collections import deque, namedtuple  # 双端队列与命名元组
from torch.utils.tensorboard import SummaryWriter  # 用于日志记录和可视化
from moviepy.editor import ImageSequenceClip  # 将图像序列转换为视频（用于结果可视化）

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义一个 Transition 命名元组，用于经验回放中的单个样本
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

# 定义经验回放缓冲区类
class ReplayBuffer:
    def __init__(self, capacity):
        # 初始化一个最大长度为 capacity 的双端队列，用于存储经验
        self.buffer = deque(maxlen=capacity)

    def add(self, *args):
        # 添加一个 transition 到 buffer 中
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        # 从经验池中随机采样 batch_size 个 transition，返回解压后的 batch
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))  # 解压为 (states, actions, rewards, next_states, dones)

    def __len__(self):
        # 返回当前 buffer 中的经验数量
        return len(self.buffer)

# 定义 DQN 的神经网络结构
class DeepQNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        # 标准DQN网络结构 - 三层全连接神经网络
        self.net = torch.nn.Sequential(
            # 第一层：输入状态，输出隐藏特征
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),

            # 第二层：更深层次的隐藏层
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),

            # 输出层：直接输出每个动作的Q值
            torch.nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        # 直接通过网络输出所有动作的Q值
        return self.net(x)

# 定义 DQN 智能体类
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.99, lr=1e-3,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500,
                 target_update=100, buffer_size=10000, batch_size=64,
                 device='cuda'):

        self.device = torch.device(device)  # 指定运行设备（CPU/GPU）
        self.action_dim = action_dim  # 动作空间维度

        # 初始化策略网络和目标网络
        self.policy_net = DeepQNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_net = DeepQNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 同步目标网络参数

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)  # 优化器

        self.replay_buffer = ReplayBuffer(buffer_size)  # 经验回放池
        self.gamma = gamma  # 折扣因子
        self.batch_size = batch_size  # 批次大小
        self.target_update = target_update  # 目标网络更新周期

        # ε-greedy 策略参数
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0  # 用于记录步数以衰减 ε

        self.writer = SummaryWriter()  # TensorBoard 写入器

    def take_action(self, state):
        # 根据 ε-greedy 策略选择动作
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  math.exp(-1. * self.total_steps / self.epsilon_decay)
        self.writer.add_scalar("Epsilon", epsilon, self.total_steps)  # 记录 ε 的变化
        self.total_steps += 1
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)  # 随机探索
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.policy_net(state).argmax().item()  # 贪婪选择

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # 样本不足，暂不更新

        # 从经验池采样一批数据
        transitions = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(transitions.state, dtype=torch.float32).to(self.device)
        actions = torch.tensor(transitions.action, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(transitions.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(transitions.next_state, dtype=torch.float32).to(self.device)
        dones = torch.tensor(transitions.done, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 计算当前 Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)  # 根据动作索引取对应的 Q 值

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]

        # TD 目标值计算：r + γ * Q'
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算均方误差损失
        loss = F.mse_loss(q_values, targets)

        # 反向传播更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 记录损失到 TensorBoard
        self.writer.add_scalar("Loss", loss.item(), self.total_steps)

        # 每隔 target_update 步更新目标网络
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # 保存模型
    def save(self, filename):
        # 将当前策略网络（policy_net）的参数保存到指定文件中
        torch.save(self.policy_net.state_dict(), filename)

    # 加载模型
    def load(self, filename):
        # 加载保存的策略网络参数到当前网络
        self.policy_net.load_state_dict(torch.load(filename))
        # 同时将目标网络（target_net）参数设置为策略网络的参数
        self.target_net.load_state_dict(self.policy_net.state_dict())

# 测试训练得到的策略
def test_policy(env, agent, episodes=5):
    total_returns = []  # 用于存储每个测试回合的总回报
    for _ in range(episodes):
        state, _ = env.reset()  # 重置环境，获取初始状态
        terminated, truncated = False, False
        total_reward = 0  # 初始化总回报
        while not (terminated or truncated):
            action = agent.take_action(state)  # 由智能体选择一个动作
            next_state, reward, terminated, truncated, _ = env.step(action)  # 执行该动作，获取下一个状态、奖励和是否结束
            state = next_state
            total_reward += reward  # 累加奖励
        total_returns.append(total_reward)  # 将本回合的总回报加入列表
    # 返回所有回合的平均回报
    return np.mean(total_returns)

# 录制训练过程的视频
def record_video(env, agent, filename="mountaincar_demo.mp4", episodes=3, fps=30):
    frames = []  # 存储每一帧图像
    for _ in range(episodes):
        state, _ = env.reset()   # 重置环境，获取初始状态
        terminated, truncated = False, False
        while not (terminated or truncated):
            frame = env.render()  # 获取当前环境的图像帧
            frames.append(frame)  # 将当前帧加入到帧列表中
            action = agent.take_action(state)  # 由智能体选择一个动作
            state, _, terminated, truncated, _ = env.step(action)  # 执行动作并获取下一个状态和结束标志
    # 使用 MoviePy 将所有帧合成视频
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(filename)  # 保存为视频文件

# 绘制训练过程中的回报图
def plot_metrics(returns, moving_avgs):
    plt.figure(figsize=(10, 5))
    plt.plot(returns, label='Return')  # 绘制每个回合的回报
    plt.plot(moving_avgs, label='Moving Average (10)')  # 绘制10个回合的滑动平均回报
    # 添加成功阈值线
    success_threshold = -110  # MountainCar成功标准
    plt.axhline(y=success_threshold, color='r', linestyle='--', label='Success Threshold')
    plt.xlabel('Episode')  # x轴标签：回合数
    plt.ylabel('Return')  # y轴标签：回报
    plt.title('DQN Training Return')  # 图表标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()  # 展示图表

# 计算训练过程的稳定性指标
def compute_stability(rewards, window=10):
    # 计算过去10个回合奖励的标准差，作为稳定性指标
    stability = [np.std(rewards[max(0, i - window):i + 1]) for i in range(len(rewards))]
    plt.figure()
    plt.plot(stability, label='Reward Std Deviation')  # 绘制奖励标准差
    plt.xlabel('Episode')  # x轴标签：回合数
    plt.ylabel('Std Dev')  # y轴标签：标准差
    plt.title('Stability Metric')  # 图表标题
    plt.grid(True)  # 显示网格
    plt.legend()  # 显示图例
    plt.show()  # 展示图表

# 主函数
def main():
    # 创建 MountainCar-v0 环境
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    seed = 42  # 设置随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    np.random.seed(seed)  # 设置 Numpy 随机种子
    random.seed(seed)  # 设置 Python 随机种子
    state, _ = env.reset(seed=seed)  # 设置环境的随机种子  # 设置环境的随机种子

    # 获取状态空间和动作空间的维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # 创建 DQN 智能体，使用 GPU（如果可用），否则使用 CPU
    agent = DQNAgent(state_dim, action_dim, device='cuda' if torch.cuda.is_available() else 'cpu')

    num_episodes = 300  # 设置总的训练回合数
    return_list = []  # 存储每个回合的回报
    moving_avg_rewards = []  # 存储回报的滑动平均
    ma_window = 10  # 设置滑动平均窗口大小

    # 训练过程
    for i in range(num_episodes):
        state, _ = env.reset()  # 重置环境
        episode_return = 0  # 当前回合的总回报
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.take_action(state)  # 由智能体选择一个动作
            next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作并获取下一个状态、奖励、是否结束
            # clipped_reward = np.clip(reward, -1, 1)  # 奖励裁剪至[-1, 1]区间
            agent.replay_buffer.add(state, action, reward, next_state, terminated or truncated)  # 将经验加入经验池
            agent.update()  # 更新智能体
            state = next_state  # 更新状态
            episode_return += reward  # 累加奖励

        return_list.append(episode_return)  # 将当前回合的回报加入回报列表
        moving_avg = np.mean(return_list[-ma_window:])  # 计算最近 ma_window 个回合的滑动平均回报
        moving_avg_rewards.append(moving_avg)

        # 将训练回报和滑动平均回报写入 TensorBoard
        agent.writer.add_scalar("Return/train", episode_return, i)
        agent.writer.add_scalar("Return/MovingAverage", moving_avg, i)

        # 每10个回合测试一次模型
        if (i + 1) % 10 == 0:
            avg_return = test_policy(env, agent)  # 测试模型
            print(f"Episode {i+1}, Return: {episode_return:.2f}, Moving Avg: {moving_avg:.2f}, Test Avg: {avg_return:.2f}")
            agent.writer.add_scalar("Return/test_avg", avg_return, i)  # 将测试回报写入 TensorBoard

    # 保存训练好的模型
    # agent.save("dqn_mountaincar.pth")

    # 绘制回报图和稳定性图
    plot_metrics(return_list, moving_avg_rewards)
    compute_stability(return_list)

    # print("\n🎬 正在录制演示视频 ...")
    # # 录制演示视频
    # record_video(env, agent)
    # print("✅ 视频已保存为 mountaincar_demo.mp4")

# 运行主函数
if __name__ == '__main__':
    main()