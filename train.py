import numpy as np
import torch
from ai import DQN
import random
import xlrd
import os
import time

# 定义状态和动作维度
state_dim = 20  # 例如：hp1, hp2, zer1, zer2, 之前使用的技能
action_dim = 16  # 假设有16种技能

dqn = DQN(state_dim, action_dim)

# 定义保存模型的路径
model_save_path = "dqn_model.pth"

if os.path.exists(model_save_path):
    dqn.eval_net.load_state_dict(torch.load(model_save_path))
    dqn.target_net.load_state_dict(torch.load(model_save_path))
    print(f"Loaded existing model from {model_save_path}")
else:
    print("No existing model found, starting training from scratch.")

def excel_to_list(file, index=0):
    workbook = xlrd.open_workbook(file)
    worksheet = workbook.sheet_by_index(index)
    rows = worksheet.nrows
    all_data = []
    for i in range(rows):
        row = worksheet.row_values(i)[:]
        all_data.append(row)
    return all_data

HP_table = excel_to_list("./move.xls", 0)
name_table = excel_to_list("./move.xls", 1)[0]
zer_table = excel_to_list("./move.xls", 2)


# 假设 zer_compute 和 HP_compute 函数已经定义
def zer_compute(key, ai_spell, zer1, zer2):
    for i in range(len(zer1)):
        zer1[i] += zer_table[i][key]
        zer2[i] += zer_table[i][ai_spell]

def HP_compute(key, ai_spell, hp1, hp2):
    if HP_table[key][ai_spell] > 0:
        hp2 -= HP_table[key][ai_spell]
    else:
        hp1 += HP_table[key][ai_spell]
    return hp1, hp2

def get_state(hp1, hp2, zer1, zer2):
    # 获取当前状态
    state = np.array([hp1, hp2] + zer1 + zer2)
    return state

def get_reward(prev_hp1, prev_hp2, hp1, hp2):
    # 奖励基于生命值变化
    reward = (prev_hp2 - hp2) - (prev_hp1 - hp1)
    return reward

def is_done(hp1, hp2):
    # 判断游戏是否结束
    return hp1 <= 0 or hp2 <= 0

def train():
    for episode in range(100000):
        # 初始化状态
        hp1 = hp2 = 2
        zer1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        zer2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        state = get_state(hp1, hp2, zer1, zer2)
        prev_hp1, prev_hp2 = hp1, hp2

        while True:
            action_player = dqn.choose_action(state)
            action_opponent = random.randint(0, action_dim - 1)

            # 执行动作
            zer_compute(action_player, action_opponent, zer1, zer2)
            hp1, hp2 = HP_compute(action_player, action_opponent, hp1, hp2)

            # 更新状态
            next_state = get_state(hp1, hp2, zer1, zer2)
            reward = get_reward(prev_hp1, prev_hp2, hp1, hp2)
            done = is_done(hp1, hp2)

            dqn.store_transition(state, action_player, reward, next_state)
            state = next_state

            if done:
                dqn.update_target()
                break

            dqn.update()
            prev_hp1, prev_hp2 = hp1, hp2

    # 保存模型
    torch.save(dqn.eval_net.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    start_time = time.time()
    
    train()

    end_time = time.time()  # 获取当前时间
    run_time = end_time - start_time  # 计算运行时间
    print("程序运行时间为:", run_time, "秒")