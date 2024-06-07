# -*- coding: utf-8 -*-
import pygame
import sys
import traceback
import xlrd
from pygame.locals import *
from random import *
import numpy as np
from ai import DQN
import torch

pygame.init()
pygame.mixer.init()

bg_size = width, height = 1000, 500
screen = pygame.display.set_mode(bg_size)
pygame.display.set_caption("加子")
font = pygame.font.Font("font/simhei.ttf", 24)
background = pygame.image.load("images/background.png").convert()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


# 定义状态和动作维度
state_dim = 20  # 例如：hp1, hp2, zer1, zer2, 之前使用的技能
action_dim = 16  # 假设有16种技能

# 加载模型
model_save_path = "dqn_model.pth"
dqn = DQN(state_dim, action_dim)
dqn.eval_net.load_state_dict(torch.load(model_save_path))
dqn.eval_net.eval()

def get_state():
    global hp1, hp2, zer1, zer2
    # 获取当前状态
    state = np.array([hp1, hp2] + zer1 + zer2)
    return state

def get_reward():
    global hp1, hp2, prev_hp1, prev_hp2
    # 奖励基于生命值变化
    reward = (prev_hp2 - hp2) - (prev_hp1 - hp1)
    return reward

def is_done():
    global hp1, hp2
    # 判断游戏是否结束
    return hp1 <= 0 or hp2 <= 0



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

print(name_table)

def game_over(winner):
    if winner == 1:
        gameover_text1 = font.render("你赢了", True, (255, 255, 255))
    else:
        gameover_text1 = font.render("你输了", True, (255, 255, 255))
    
    gameover_text1_rect = gameover_text1.get_rect()
    gameover_text1_rect.left, gameover_text1_rect.top = (100, 100)
    screen.blit(gameover_text1, gameover_text1_rect)
    pygame.display.flip()

def zer_compute(key, ai_spell):
    global zer1, zer2

    for i in range(len(zer1)):
        zer1[i] += zer_table[i][key]
        zer2[i] += zer_table[i][ai_spell]

def HP_compute(key, ai_spell):
    global hp1, hp2

    if HP_table[key][ai_spell] > 0:
        hp2 -= HP_table[key][ai_spell]
    else:
        hp1 += HP_table[key][ai_spell]

def print_move(key, ai_spell):
    screen.blit(background, (0, 0))

    print_table = [[name_table[key], (200, 350)], [name_table[ai_spell], (200, 150)],
                   [f"HP:{int(hp1)}", (300, 350)], [f"HP:{int(hp2)}", (300, 150)],
                   [f"子儿：{int(zer1[0])}", (300, 400)], [f"子儿：{int(zer2[0])}", (300, 200)]]

    for i in range(6):
        text = font.render(print_table[i][0], True, WHITE)
        text_rect = text.get_rect()
        text_rect.left, text_rect.top = print_table[i][1]
        screen.blit(text, text_rect)

    spell_image = pygame.image.load(f"./images/spell/{key}.png").convert_alpha()
    spell_rect = spell_image.get_rect()
    spell_rect.center = (100, 350)
    screen.blit(spell_image, spell_rect)

    spell_image = pygame.image.load(f"./images/spell/{ai_spell}.png").convert_alpha()
    spell_rect = spell_image.get_rect()
    spell_rect.center = (100, 150)
    screen.blit(spell_image, spell_rect)

    pygame.display.flip()

def print_button():
    button_image = pygame.image.load("images/button.png").convert_alpha()
    button_rect = button_image.get_rect()


def main():
    global ai_spell, hp1, hp2, zer1, zer2, prev_hp1, prev_hp2

    state_dim = 20  # 例如：hp1, hp2, zer1, zer2, 之前使用的技能
    action_dim = 16  # 假设有16种技能

    dqn = DQN(state_dim, action_dim)

    ai_spell = 0
    hp1 = hp2 = 2
    zer1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    zer2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    running = True

    # pygame.mixer.music.play(-1)
    clock = pygame.time.Clock()

    screen.blit(background, (0, 0))
    pygame.display.flip()
    print_move(0, 0)

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                key = event.key
                tmp = {K_KP0 : 0, K_KP1 : 1, K_KP2 : 2, K_KP3 : 3, K_KP4 : 4, K_KP5 : 5, K_KP6 : 6, K_KP7 : 7, K_KP8 : 8}

                if key not in tmp:
                    continue

                prev_hp1, prev_hp2 = hp1, hp2
                zer_compute(tmp[key], ai_spell)
                HP_compute(tmp[key], ai_spell)
                print_move(tmp[key], ai_spell)
                
                state = get_state()
                reward = get_reward()
                done = is_done()
                next_state = get_state()

                dqn.store_transition(state, ai_spell, reward, next_state)
                dqn.update()

                if done:
                    dqn.update_target()
                    game_over(1 if hp2 <= 0 else 2)
                    running = False
                else:
                    ai_spell = dqn.choose_action(state)

        clock.tick(60)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            if event.type == KEYDOWN:
                main()

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except:
        traceback.print_exc()
        pygame.quit()
        input()
