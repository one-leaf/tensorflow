# coding=utf-8
# 按照每一局完成后统一评价和学习

import pygame,sys,time,random
from pygame.locals import *
#########
import numpy as np
import random
import platform, sys, os
from collections import deque
import tensorflow as tf
import cv2
import copy
import time
import pickle

winx = 400
winy = 500
boxsize = 20
boardwidth = 10
boardheight = 20
xmargin = int(winx-boardwidth*boxsize)/5
topmargin = int(winy-boardheight*boxsize-5)
templatenum = 5
 
white = (255,255,255)
black = (0,0,0)
blue = (0,0,255)
yellow = (255,255,0)
green = (0,255,0)
purple = (255,0,255)
red = (255,0,0)
blank = '.'
colors = (yellow,green,purple,red)

KEY_ROTATION  = [0,1,0,0]
KEY_LEFT      = [1,0,0,0]
KEY_RIGHT     = [0,0,1,0]
KEY_DOWN      = [0,0,0,1]

stemplate = [['.....',
              '..00.',
              '.00..',
              '.....',
              '.....'],
             ['.....',
              '..0..',
              '..00.',
              '...0.',
              '.....']]
 
ztemplate = [['.....',
              '.00..',
              '..00.',
              '.....',
              '.....'],
             ['.....',
              '...0.',
              '..00.',
              '..0..',
              '.....']]
 
itemplate = [['..0..',
              '..0..',
              '..0..',
              '..0..',
              '.....'],
             ['.....',
              '.0000',
              '.....',
              '.....',
              '.....']]
 
otemplate = [['.....',
              '..00.',
              '..00.',
              '.....',
              '.....']]
 
ltemplate = [['.....',
              '..0..',
              '..0..',
              '..00.',
              '.....'],
             ['.....',
              '...0.',
              '.000.',
              '.....',
              '.....'],
             ['.....',
              '..00.',
              '...0.',
              '...0.',
              '.....'],
             ['.....',
              '.000.',
              '.0...',
              '.....',
              '.....']]
 
jtemplate = [['.....',
              '..0..',
              '..0..',
              '.00..',
              '.....'],
             ['.....',
              '.000.',
              '...0.',
              '.....',
              '.....'],
             ['.....',
              '..00.',
              '..0..',
              '..0..',
              '.....'],
             ['.....',
              '.0...',
              '.000.',
              '.....',
              '.....']]
 
ttemplate = [['.....',
              '..0..',
              '.000.',
              '.....',
              '.....'],
             ['..0..',
              '.00..',
              '..0..',
              '.....',
              '.....'],
             ['.....',
              '.000.',
              '..0..',
              '.....',
              '.....'],
             ['..0..',
              '..00.',
              '..0..',
              '.....',
              '.....']]
pieces = {'s':stemplate,
          'z':ztemplate,
          'i':itemplate,
          'o':otemplate,
          'l':ltemplate,
          'j':jtemplate,
          't':ttemplate}

class Tetromino(object):
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('tetromino')
        self.disp = pygame.display.set_mode((winx,winy))
        self.bigfont = pygame.font.Font('freesansbold.ttf',100)
        self.basicfont = pygame.font.Font('freesansbold.ttf',20)
        self.reset()

    def reset(self):
        self.fallpiece = self.getnewpiece()
        self.nextpiece = self.getnewpiece()
        self.score = 0
        self.board = self.getblankboard()
        self.calc_reward = 0.0 

    def step(self, action):
        moveleft = False
        moveright = False
        is_epoch_end = False
        is_terminal = False
        shape  = self.fallpiece['shape']
        
        level = self.calculate(self.score)

        if action == KEY_LEFT and self.validposition(self.board,self.fallpiece,ax = -1):
            self.fallpiece['x']-=1
            moveleft = True

        if action == KEY_RIGHT and self.validposition(self.board,self.fallpiece,ax = 1):
            self.fallpiece['x']+=1  
            moveright = True 

        if action == KEY_ROTATION:
            self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])
            if not self.validposition(self.board,self.fallpiece):
                self.fallpiece['rotation'] = (self.fallpiece['rotation'] - 1) % len(pieces[self.fallpiece['shape']])

        if not self.validposition(self.board,self.fallpiece,ay = 1):
            self.addtoboard(self.board,self.fallpiece)
            self.score += self.removecompleteline(self.board)            
            level = self.calculate(self.score)   
            self.fallpiece = None
            is_epoch_end = True
        else:
            self.fallpiece['y'] +=1

        # self.disp.fill(black)
        # self.drawboard(self.board)
        # self.drawstatus(self.score,level)
        # self.drawnextpiece(self.nextpiece)
        # if self.fallpiece !=None:
        #     self.drawpiece(self.fallpiece)

        screen_image = np.zeros((boardwidth, boardheight))
        for y in range(boardheight):
            for x in range(boardwidth):
                if self.board[x][y]!=blank:
                    screen_image[x,y]=255

        if self.fallpiece !=None:
            shapedraw = pieces[shape][self.fallpiece['rotation']]
            for x in range(templatenum):
                for y in range(templatenum):
                    if shapedraw[y][x]!=blank:
                        if (y+self.fallpiece['y'])<0 or (x + self.fallpiece['x'])<0 : continue
                        screen_image[x + self.fallpiece['x'],y + self.fallpiece['y']] = 255
#        screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
#        pygame.display.update()

        if self.fallpiece == None:
            self.fallpiece = self.nextpiece
            self.nextpiece = self.getnewpiece()
            if not self.validposition(self.board,self.fallpiece):   
                is_terminal = True       
                self.reset()     
                return screen_image, is_terminal, is_epoch_end 
        return screen_image, is_terminal, is_epoch_end

    def calculate(self,score):
        level = int(score/10)+1
        return level
        
    def terminal(self):
        pygame.quit()
        sys.exit()
        
    def checkforquit(self):
        for event in pygame.event.get(QUIT):
            self.terminal()
        for event in pygame.event.get(KEYUP):
            if event.key == K_ESCAPE:
                self.terminal()
            pygame.event.post(event)
            
    def checkforpress(self):
        self.checkforquit()
        for event in pygame.event.get([KEYDOWN,KEYUP]):
            if event.type == KEYDOWN:
                continue
            return event.key
        return None
    
    def maketext(self,text,font,color):
        surf = font.render(text,1,color)
        return surf,surf.get_rect()
        
    def showtextscreen(self,text):
        tilesurf,tilerect = self.maketext(text,self.bigfont,white)
        tilerect.center = (int(winx/2),int(winy/2))
        self.disp.blit(tilesurf,tilerect)
    
        presssurf,pressrect = self.maketext('press a key to play',self.basicfont,white)
        pressrect.center = (int(winx/2),int(winy/2)+100)
        self.disp.blit(presssurf,pressrect)
    
        while self.checkforpress() == None:
            pygame.display.update()
            self.fpsclock.tick()
        
    def getnewpiece(self):
        shape = random.choice(list(pieces.keys()))
        newpiece = {'shape':shape,
                    'rotation': random.randint(0,len(pieces[shape])-1),
                    'x': int(boardwidth)//2-int(templatenum//2),
                    'y': -2,
                    'color': random.randint(0,len(colors)-1)}
        return newpiece
    
    def getblankboard(self):
        board = []
        for x in range(boardwidth):
                board.append([blank]*boardheight)
        return board
    
    def addtoboard(self,board,piece):
        for x in range(templatenum):
            for y in range(templatenum):
                if pieces[piece['shape']][piece['rotation']][y][x]!=blank:
                    board[x + piece['x']][y + piece['y']] = piece['color']
                
    def onboard(self,x,y):
        return x >=0 and x<boardwidth and y<boardheight
        
    def validposition(self,board,piece,ax = 0,ay = 0):
        for x in range(templatenum):
            for y in range(templatenum):
                aboveboard = y +piece['y'] +ay < 0
                if aboveboard or pieces[piece['shape']][piece['rotation']][y][x]== blank:
                    continue
                if not self.onboard(x + piece['x']+ax,y+piece['y']+ay):
                    return False
                if board[x+piece['x']+ax][y+piece['y']+ay]!=blank:
                    return False
        return True           

    def completeline(self,board,y):
        for x in range(boardwidth):
            if board[x][y]==blank:
                return False
        return True
    
    def removecompleteline(self,board):
        numremove = 0
        y = boardheight-1
        while y >=0:
            if self.completeline(board,y):
                for pulldowny in range(y,0,-1):
                    for x in range (boardwidth):
                        board[x][pulldowny] = board[x][pulldowny-1]
                for x in range(boardwidth):
                    board[x][0] = blank
                numremove+=1
            else:
                y-=1
        return numremove
    
    def convertsize(self,boxx,boxy):
        return (boxx*boxsize+xmargin,boxy*boxsize+topmargin)
    
    def drawbox(self,boxx,boxy,color,pixelx = None,pixely= None):
        if color == blank:
            return
        if pixelx == None and pixely == None:
            pixelx,pixely = self.convertsize(boxx,boxy)
        pygame.draw.rect(self.disp,colors[color],(pixelx+1 , pixely+1,boxsize-1,boxsize-1))
        
    def drawboard(self,board):
        pygame.draw.rect(self.disp,blue,(xmargin-3,topmargin-7,boardwidth*boxsize+8,boardheight*boxsize+8),5)
        for x in range(boardwidth):
            for y in range(boardheight):
                self.drawbox(x,y,board[x][y])
    
    def drawstatus(self,score,level):
        scoresurf = self.basicfont.render('Score: %s'%score,True,white)
        scorerect = scoresurf.get_rect()
        scorerect.topleft = (winx-150,20)
        self.disp.blit(scoresurf,scorerect)
    
        levelsurf = self.basicfont.render('level: %s'%level,True, white)
        levelrect = levelsurf.get_rect()
        levelrect.topleft = (winx-150,50)
        self.disp.blit(levelsurf,levelrect)
    
    def drawpiece(self,piece,pixelx = None,pixely = None):
        shapedraw = pieces[piece['shape']][piece['rotation']]
        if pixelx == None and pixely == None:
            pixelx,pixely = self.convertsize(piece['x'],piece['y'])
        for x in range(templatenum):
            for y in range(templatenum):
                if shapedraw[y][x]!=blank:
                    self.drawbox(None,None,piece['color'],pixelx+(x*boxsize),pixely + y*boxsize)
    
    def drawnextpiece(self,piece):
        nextsurf = self.basicfont.render('Next:',True,white)
        nextrect =nextsurf.get_rect()
        nextrect.topleft = (winx-120,80)
        self.disp.blit(nextsurf,nextrect)
        self.drawpiece(piece,pixelx = winx-120,pixely = 100)


# 参数设置
ACTIONS_COUNT = 4  # 可选的动作，针对 左移 翻转 右移 下移
FUTURE_REWARD_DISCOUNT = 0.99  # 下一次奖励的衰变率 
OBSERVATION_STEPS = 15000.  # 在学习前观察的次数
MIN_RANDOM_ACTION_PROB = 0.05    # 随机移动的最小概率
MAX_RANDOM_ACTION_PROB = 0.95    # 随机移动的最大概率 
MEMORY_SIZE = 10000  # 记住的观察队列
TRAIN_BATCH_SIZE = 100  # 每次学习的批次
TRAIN_EPOCHS = 2   # 每次学习轮数
STATE_FRAMES = 4  # 每次保存的状态数
RESIZED_SCREEN_X, RESIZED_SCREEN_Y = (boardwidth, boardheight)   # 图片的尺寸 
OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)
SAVE_EVERY_X_STEPS = 1000   # 每学习多少轮后保存
STORE_SCORES_LEN = 200      # 分数保留的长度
LEARNING_RATE = 1e-6        # 学习速率

# 初始化保存对象，如果有数据，就恢复
def restore(sess):
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    saver_prefix = os.path.join(model_dir, "model.ckpt")        
    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver = tf.train.Saver(max_to_keep=5)
    if ckpt and ckpt.model_checkpoint_path:
        print("restore model ...")
        saver.restore(sess, ckpt.model_checkpoint_path)
    return saver, model_dir, saver_prefix

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def add_conv_layer(inputs, patch_size, in_size, out_size, activation_function=None, pool_function=None):
    Weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, in_size, out_size], stddev=0.1))
    biases = tf.Variable(tf.zeros([out_size]) + 0.1)
    layer = tf.nn.conv2d(inputs, Weights, strides=[1, 1, 1, 1], padding='SAME')
    Wconvlayer_plus_b = layer + biases
    if activation_function is None:
        convlayer = Wconvlayer_plus_b
    else:
        convlayer = activation_function(Wconvlayer_plus_b)
    if pool_function is None:
        outputs = convlayer
    else:
        outputs = pool_function(convlayer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return outputs

# 学习
def train():    
    # 输入的图片，是每4张一组
    x =  tf.placeholder("float", [None, RESIZED_SCREEN_X, RESIZED_SCREEN_Y, STATE_FRAMES], name='input_layer')   
    y = tf.placeholder("float", [None, ACTIONS_COUNT])    # 移动的方向
    keep_prob = tf.placeholder(tf.float32)

    layer = add_conv_layer(x, 5, 4, 16, activation_function=tf.nn.relu)
    layer = tf.nn.dropout(layer, keep_prob)
    layer = add_conv_layer(layer, 3, 16, 32, activation_function=tf.nn.relu)
    layer = tf.nn.dropout(layer, keep_prob)
    layer = add_conv_layer(layer, 3, 32, 64, activation_function=tf.nn.relu)
    layer = tf.nn.dropout(layer, keep_prob)
    layer = add_conv_layer(layer, 3, 64, 128, activation_function=tf.nn.relu)
    layer = tf.nn.dropout(layer, keep_prob)
    layer_size = RESIZED_SCREEN_X * RESIZED_SCREEN_Y * 128
    full_layer =  tf.reshape(layer, [-1,layer_size])    
    prediction = add_layer(full_layer, layer_size, ACTIONS_COUNT, tf.nn.softmax) 

    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))

    train_operation = tf.train.AdamOptimizer(0.001).minimize(cost)

    game = Tetromino()

    _session = tf.Session()       
    _session.run(tf.global_variables_initializer())

    # 恢复游戏进度
    _saver,_model_dir,_checkpoint_path = restore(_session)
    # 游戏最大进行步数
    _game_max_epoch_dump_file = os.path.join(_model_dir,"game_max_epoch.dump")
    if os.path.exists(_game_max_epoch_dump_file):
        _game_max_epoch = pickle.load(open(_game_max_epoch_dump_file,'rb'))
    else:
        _game_max_epoch = 0


    _last_x = []
    _last_y = []
    while True:
        _game_times = 0
        _action = KEY_DOWN
        _state = None    
        _epoch_num = 0    
        _curr_x = []
        _curr_y = []
        while _game_times < 50:
            image, terminal, is_epoch_end = game.step(list(_action))
            if is_epoch_end:
                _epoch_num += 1

            if terminal:
                _game_times += 1
                print(_game_times, _epoch_num//_game_times, _game_max_epoch)
                continue

            for event in pygame.event.get():  # 需要事件循环，否则白屏
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()    

            if _state is None:  # 填充第一次的4张图片            
                _state = np.stack(tuple(image for _ in range(STATE_FRAMES)), axis=2)

            image = np.reshape(image, (RESIZED_SCREEN_X, RESIZED_SCREEN_Y, 1))
            _state = np.append(_state[:, :, 1:], image, axis=2)
            _curr_x.append(_state)
            _curr_state = np.reshape(_state,[1, RESIZED_SCREEN_X, RESIZED_SCREEN_Y, 4])

            _action = np.zeros([ACTIONS_COUNT],dtype=np.int)
            _curr_random_action_prob = MAX_RANDOM_ACTION_PROB - (MAX_RANDOM_ACTION_PROB * _game_max_epoch / 100)
            if _curr_random_action_prob < MIN_RANDOM_ACTION_PROB:
                _curr_random_action_prob = MIN_RANDOM_ACTION_PROB
            if random.random() <= _curr_random_action_prob:
                action_index = random.randrange(ACTIONS_COUNT)
            else:
                _per_action = _session.run(prediction, feed_dict={x: _curr_state, keep_prob: 1.0})
                action_index = np.argmax(_per_action)
            _action[action_index] = 1

            _curr_y.append(_action)

            _y = np.reshape(_action,[1,4])

        _epoch_num = _epoch_num//_game_times
        if _epoch_num > _game_max_epoch:
            _game_max_epoch = _epoch_num
            for i in range(_game_max_epoch):
                _ = _session.run(train_operation, feed_dict={x: _curr_x, y: _curr_y, keep_prob: 0.75})
            _last_x = _curr_x
            _last_y = _curr_y
            pickle.dump(_game_max_epoch, open(_game_max_epoch_dump_file, 'wb'))
        else:
            if len(_last_x) > 0 :
                for i in range(_game_max_epoch):
                    _ = _session.run(train_operation, feed_dict={x: _last_x, y: _last_y, keep_prob: 0.75})

        print(_game_max_epoch, _epoch_num, "save model ...")
        _saver.save(_session, _checkpoint_path)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    start = time.clock()
    train()
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)
