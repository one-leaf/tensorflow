# coding=utf-8

# This is heavily based off https://github.com/asrivat1/DeepLearningVideoGames
import os
import random
from collections import deque
import tensorflow as tf
import numpy as np
import cv2
import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
import platform,sys

BLACK     = (0  ,0  ,0  )
WHITE     = (255,255,255)
 
SCREEN_SIZE = [320,400]
BAR_SIZE = [50, 5]
BALL_SIZE = [15, 15]
 
MOVE_LEFT = [1, 0, 0]   # 左移
MOVE_STAY = [0, 1, 0]   # 不动
MOVE_RIGHT = [0, 0, 1]  # 右移

class Game(object):
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption('Simple Game')
 
        self.ball_pos_x = SCREEN_SIZE[0]//2 - BALL_SIZE[0]/2
        self.ball_pos_y = SCREEN_SIZE[1]//2 - BALL_SIZE[1]/2
 
        self.ball_dir_x = -1 # -1 = left 1 = right  
        self.ball_dir_y = -1 # -1 = up   1 = down
        self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])
 
        self.bar_pos_x = SCREEN_SIZE[0]//2-BAR_SIZE[0]//2
        self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1]-BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])
 
    # action是MOVE_STAY、MOVE_LEFT、MOVE_RIGHT
    def step(self, action):
        if action == MOVE_LEFT:
            self.bar_pos_x = self.bar_pos_x - 2
        elif action == MOVE_RIGHT:
            self.bar_pos_x = self.bar_pos_x + 2
        else:
            pass
        if self.bar_pos_x < 0:
            self.bar_pos_x = 0
        if self.bar_pos_x > SCREEN_SIZE[0] - BAR_SIZE[0]:
            self.bar_pos_x = SCREEN_SIZE[0] - BAR_SIZE[0]
            
        self.screen.fill(BLACK)
        self.bar_pos.left = self.bar_pos_x
        pygame.draw.rect(self.screen, WHITE, self.bar_pos)
 
        self.ball_pos.left += self.ball_dir_x * 2
        self.ball_pos.bottom += self.ball_dir_y * 3
        pygame.draw.rect(self.screen, WHITE, self.ball_pos)
 
        if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (SCREEN_SIZE[1] - BAR_SIZE[1]+1):
            self.ball_dir_y = self.ball_dir_y * -1
        if self.ball_pos.left <= 0 or self.ball_pos.right >= (SCREEN_SIZE[0]):
            self.ball_dir_x = self.ball_dir_x * -1
 
        reward = 0
        if self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left < self.ball_pos.right and self.bar_pos.right > self.ball_pos.left):
            reward = 1    # 击中奖励
        elif self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left > self.ball_pos.right or self.bar_pos.right < self.ball_pos.left):
            reward = -1   # 没击中惩罚
 
        # 获得游戏界面像素
        screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
        # 坑：交换高宽数据
        # screen_image = np.swapaxes(screen_image, 0, 1)
        pygame.display.update()
        # 返回游戏界面像素和对应的奖励
        return reward, screen_image

# 参数设置
DEBUG = True
ACTIONS_COUNT = 3  # 可选的动作，针对 左移 不动 右移
FUTURE_REWARD_DISCOUNT = 0.99  # 下一次奖励的衰变率
OBSERVATION_STEPS = 50000.  # 在学习前观察的次数
EXPLORE_STEPS = 500000.  # 每次机器自动参与的概率的除数
INITIAL_RANDOM_ACTION_PROB = 1.0  # 随机移动的最大概率
FINAL_RANDOM_ACTION_PROB = 0.05  # 随机移动的最小概率
MEMORY_SIZE = 500000  # 记住的观察队列
MINI_BATCH_SIZE = 100  # 每次学习的批次
STATE_FRAMES = 4  # 每次保存的状态数
RESIZED_SCREEN_X, RESIZED_SCREEN_Y = (80, 100)   # 图片缩小后的尺寸
OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_MAX_PROBABILITY_INDEX = range(5)
SAVE_EVERY_X_STEPS = 100  # 每学习多少轮后保存
LEARN_RATE = 1e-6           # 学习的速率
STORE_SCORES_LEN = 200.     # 分数保留的长度

# 初始化保存对象，如果有数据，就恢复
def restore(sess):
    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, "game_model")
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    saver_prefix = os.path.join(model_dir, "model.ckpt")        
    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver = tf.train.Saver(max_to_keep=1)
    if ckpt and ckpt.model_checkpoint_path:
        print("restore model ...")
        saver.restore(sess, ckpt.model_checkpoint_path)
    return saver, model_dir, saver_prefix

# 神经网络定义
def get_network():
    x = tf.placeholder("float", [None, RESIZED_SCREEN_X, RESIZED_SCREEN_Y, STATE_FRAMES])   # 输入的图片，是每4张一组
    def get_w_b(w_shape,w_name,b_name):
        w = tf.get_variable(w_name, w_shape, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(b_name, [w_shape[-1]], initializer=tf.constant_initializer(0.01))
        return w,b
    w1, b1 = get_w_b([8, 8, STATE_FRAMES, 32],"w1","b1")
    w2, b2 = get_w_b([4, 4, 32, 64],"w2","b2")
    w3, b3 = get_w_b([3, 3, 64, 64],"w3","b3")
    fw1, fb1 = get_w_b([3 * 4 * 64, 256],"fw1","fb1")
    fw2, fb2 = get_w_b([256, ACTIONS_COUNT],"fw2","fb2")
    hidden_layer_1  = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, 4, 4, 1], padding="SAME") + b1)
    hidden_pool_1   = tf.nn.max_pool(hidden_layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    hidden_layer_2  = tf.nn.relu(tf.nn.conv2d(hidden_pool_1, w2, strides=[1, 2, 2, 1], padding="SAME") + b2)
    hidden_pool_2   = tf.nn.max_pool(hidden_layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    hidden_layer_3  = tf.nn.relu(tf.nn.conv2d(hidden_layer_2, w3, strides=[1, 1, 1, 1], padding="SAME") + b3)
    hidden_pool_3   = tf.nn.max_pool(hidden_layer_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    hidden_pool_3_flat = tf.reshape(hidden_pool_3, [-1, 3 * 4 * 64])
    final_hidden_activations = tf.nn.relu(tf.matmul(hidden_pool_3_flat, fw1) + fb1)
    y = tf.matmul(final_hidden_activations, fw2) + fb2
    return x, y

# 学习
def train():    
    _input_layer , _output_layer = get_network()
    
    _action = tf.placeholder("float", [None, ACTIONS_COUNT])    # 移动的方向
    _target = tf.placeholder("float", [None])                   # 得分
    _probability = tf.placeholder("float", [None])              # 概率

    # 将预测的结果和移动的方向相乘，按照第二维度求和 [0.1,0.2,0.7] * [0, 1, 0] = [0, 0.2 ,0] = [0.2]  得到当前移动的概率
    readout_action = tf.reduce_sum(tf.multiply(_output_layer, _action), reduction_indices=1)
    # 将（结果和评价相减）的平方，再求平均数。 得到和评价的距离。
    cost = tf.reduce_mean(tf.square(_target - readout_action - _probability))
    # 学习函数
    _train_operation = tf.train.AdamOptimizer(LEARN_RATE).minimize(cost)

    _observations = deque()
    _last_scores = deque()
    
    # 设置最后一步是固定
    _last_action = MOVE_STAY
    _last_state = None          #4次的截图
    _probability_of_random_action = INITIAL_RANDOM_ACTION_PROB
    _last_probability = None    #4次的概率

    _time = 0
    game = Game()

    _session = tf.Session()       
    _session.run(tf.global_variables_initializer())

    _saver,_model_dir,_checkpoint_path = restore(_session)

    if DEBUG:
        tf.summary.scalar("cost", cost)
        _train_summary_op = tf.summary.merge_all()
        _train_summary_writer = tf.summary.FileWriter(_model_dir, _session.graph)

    while True:
        reward, image = game.step(list(_last_action))
        # 将游戏抓图缩小并扁平化
#        plt.imshow(image)
#        plt.show()

        if platform.system()!="Linux":
            for event in pygame.event.get():  # Linux不需要事件循环，其余需要否则白屏
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()   

        image = cv2.resize(image,(RESIZED_SCREEN_Y, RESIZED_SCREEN_X))

        screen_resized_grayscaled = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _, screen_resized_binary = cv2.threshold(screen_resized_grayscaled, 1, 255, cv2.THRESH_BINARY)
        if reward != 0.0:
            _last_scores.append(reward)
            if len(_last_scores) > STORE_SCORES_LEN:
                _last_scores.popleft()

        if _last_state is None:  # 填充第一次的4张图片            
            _last_state = np.stack(tuple(screen_resized_binary for _ in range(STATE_FRAMES)), axis=2)

        if _last_probability is None:  # 填充概率
            _last_probability = np.zeros([STATE_FRAMES])
        
        #上一次步数的最大概率预测
        before_action = _session.run(_output_layer, feed_dict={_input_layer: [_last_state]})
        before_action_probability_max = np.max(before_action)
        before_action_probability = np.append(_last_probability[1:], before_action_probability_max)

        screen_resized_binary = np.reshape(screen_resized_binary, (RESIZED_SCREEN_X, RESIZED_SCREEN_Y, 1))
        current_state = np.append(_last_state[:, :, 1:], screen_resized_binary, axis=2)
        

        _observations.append((_last_state, _last_action, reward, current_state, before_action_probability))
        if len(_observations) > MEMORY_SIZE:
            _observations.popleft()
        
        if len(_observations) > OBSERVATION_STEPS:
            mini_batch = random.sample(_observations, MINI_BATCH_SIZE)
            previous_states = [d[OBS_LAST_STATE_INDEX] for d in mini_batch]
            actions = [d[OBS_ACTION_INDEX] for d in mini_batch]
            rewards = [d[OBS_REWARD_INDEX] for d in mini_batch]
            current_states = [d[OBS_CURRENT_STATE_INDEX] for d in mini_batch]
            before_action_probability = [d[OBS_MAX_PROBABILITY_INDEX] for d in mini_batch]

            agents_expected_reward = []
            agents_reward_per_action = _session.run(_output_layer, feed_dict={_input_layer: current_states})
            agents_before_action_probability = []
            for i in range(len(mini_batch)):
                # 如果是扣分，没有未来的奖励
                if rewards[i]==-1:
                    agents_expected_reward.append(rewards[i] * STATE_FRAMES)
                else:    
                    agents_expected_reward.append(rewards[i] * STATE_FRAMES + FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))
                agents_before_action_probability.append(np.sum(before_action_probability[i]))

            if DEBUG:
                _, train_summary_op =  _session.run([_train_operation,_train_summary_op], feed_dict={_input_layer: previous_states,_action: actions,
                        _target: agents_expected_reward,_probability:agents_before_action_probability})
            else:            
                _session.run(_train_operation, feed_dict={_input_layer: previous_states,_action: actions,
                        _target: agents_expected_reward,_probability:agents_before_action_probability})

            if _time % SAVE_EVERY_X_STEPS == 0:
                _saver.save(_session, _checkpoint_path, global_step=_time)
                if DEBUG:
                    _train_summary_writer.add_summary(train_summary_op, _time)

            _time += 1

        _last_state = current_state

        # 下一步
        _last_action = np.zeros([ACTIONS_COUNT],dtype=np.int)
        if random.random() <= _probability_of_random_action:
            action_index = random.randrange(ACTIONS_COUNT)
        else:
            readout_t = _session.run(_output_layer, feed_dict={_input_layer: [_last_state]})[0]
            print("Action Q-Values are %s" % readout_t)
            action_index = np.argmax(readout_t)
        _last_action[action_index] = 1

        if _probability_of_random_action > FINAL_RANDOM_ACTION_PROB and len(_observations) > OBSERVATION_STEPS:
            _probability_of_random_action -= (INITIAL_RANDOM_ACTION_PROB - FINAL_RANDOM_ACTION_PROB) / EXPLORE_STEPS
            print("Time: %s random_action_prob: %s reward %s scores differential %s" %
                  (_time, _probability_of_random_action, reward, sum(_last_scores) / STORE_SCORES_LEN))


if __name__ == '__main__':
    train()
