# coding=utf-8
import tensorflow as tf
import numpy as np
import cv2
import pygame
from pygame.locals import *
import platform,sys,os

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
        pygame.display.update()
        # 返回游戏界面像素和对应的奖励
        return reward, screen_image

# 恢复模型
def restore():
    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, "game_model")
    if not os.path.exists(model_dir): 
        print("error: can't load game model")
        sys.exit() 
    saver_prefix = os.path.join(model_dir, "model.ckpt")    
    metaFile= sorted(
    [
        (x, os.path.getctime(os.path.join(model_dir,x)))                  
        for x in os.listdir(model_dir) if x.endswith('.meta')  
    ],
    key=lambda i: i[1])[-1][0]
    sess = tf.Session()
    saver = tf.train.import_meta_graph(os.path.join(model_dir,metaFile))
    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver = tf.train.Saver(max_to_keep=1)
    if ckpt and ckpt.model_checkpoint_path:
        print("restore model ...")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("error: can't load checkpoint data")
        sys.exit()     
    return sess


if __name__ == '__main__':
    with tf.device('/gpu:1'):
        _session = restore()
    # names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # for name in names:
        # print(name)
    # 这里后续最好在变量定义时就指定名字，不然不好找
        _input_layer = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
        _output_layer = tf.get_default_graph().get_tensor_by_name('MatMul_1:0')
  
    _last_state = None          #4次的截图
    _last_action = MOVE_STAY    
    RESIZED_SCREEN_X, RESIZED_SCREEN_Y = (80, 100)   # 图片缩小后的尺寸
    STATE_FRAMES = 4  # 每次保存的状态数

    game = Game()    
    while True:
        reward, image = game.step(list(_last_action))
        if platform.system()!="Linux":
            for event in pygame.event.get():  # Linux不需要事件循环，其余需要否则白屏
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()   

        image = cv2.resize(image,(RESIZED_SCREEN_Y, RESIZED_SCREEN_X))

        screen_resized_grayscaled = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _, screen_resized_binary = cv2.threshold(screen_resized_grayscaled, 1, 255, cv2.THRESH_BINARY)
        if _last_state is None:  # 填充第一次的4张图片            
            _last_state = np.stack(tuple(screen_resized_binary for _ in range(STATE_FRAMES)), axis=2)

        screen_resized_binary = np.reshape(screen_resized_binary, (RESIZED_SCREEN_X, RESIZED_SCREEN_Y, 1))
        _last_state = np.append(_last_state[:, :, 1:], screen_resized_binary, axis=2)

        _last_action = np.zeros([3],dtype=np.int)
        with tf.device('/gpu:1'):
            readout_t = _session.run(_output_layer, feed_dict={_input_layer: [_last_state]})[0]
        action_index = np.argmax(readout_t)
        _last_action[action_index] = 1
