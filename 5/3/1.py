# coding=utf-8

import pygame,sys,time,random
from pygame.locals import *
#########
import numpy as np
import random
import platform, sys, os
from collections import deque
import tensorflow as tf
import cv2

winx = 400
winy = 500
boxsize = 20
boardwidth = 10
boardheight = 20
xmargin = int(winx-boardwidth*boxsize)/5
topmargin = int(winy-boardheight*boxsize-5)
templatenum = 5
 
movedownfreq = 0.1
movesidefreq = 0.15
 
white = (255,255,255)
black = (0,0,0)
blue = (0,0,255)
yellow = (255,255,0)
green = (0,255,0)
purple = (255,0,255)
red = (255,0,0)
blank = '.'
colors = (yellow,green,purple,red)

KEY_ROTATION  = [0,1,0]
KEY_LEFT      = [1,0,0]
KEY_RIGHT     = [0,0,1]

stemplate = [['.....',
              '..00.',
              '.00..',
              '.....',
              '.....'],
             ['.....',
              '..o..',
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
        reward  = 0
        moveleft = False
        moveright = False
        is_terminal = False
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
            reward = self.removecompleteline(self.board)             
            self.score += reward            
            level = self.calculate(self.score)   

            reward = self.calcreward(self.board, self.fallpiece, reward)
            self.fallpiece = None
        else:
            self.fallpiece['y'] +=1

        self.disp.fill(black)
        self.drawboard(self.board)
        self.drawstatus(self.score,level)
        self.drawnextpiece(self.nextpiece)
        if self.fallpiece !=None:
            self.drawpiece(self.fallpiece)

        screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()

        if self.fallpiece == None:
            self.fallpiece = self.nextpiece
            self.nextpiece = self.getnewpiece()
            if not self.validposition(self.board,self.fallpiece):   
                is_terminal = True       
                self.reset()
                return reward, screen_image, is_terminal   # 虽然游戏结束了，但还是正常返回分值，而不是返回 -1

        return reward, screen_image, is_terminal

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
                # print(piece['x'],piece['y'])
                if board[x+piece['x']+ax][y+piece['y']+ay]!=blank:
                    return False
        return True
    
    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    # 下落高度
    def landingHeight(self,board,piece):
        shape=pieces[piece['shape']][piece['rotation']]
        for y in range(templatenum):
            for x in range(templatenum):
                if shape[x][y] != blank:
                    return boardheight - (piece['y'] + y)

    # 行变化次数
    def rowTransitions(self,board):
        totalTransNum = 0
        for y in range(boardheight):
            nowTransNum = 0
            currisBlank = False
            for x in range(boardwidth):
                isBlank = board[x][y] == blank
                if currisBlank != isBlank:
                    nowTransNum += 1
                    currisBlank = isBlank
            if currisBlank:   
                nowTransNum += 1
            totalTransNum += nowTransNum
        return totalTransNum  

    # 列变化次数
    def colTransitions(self,board):
        totalTransNum = 0
        for x in range(boardwidth):
            nowTransNum = 0
            currisBlank = False
            for y in range(boardheight):
                isBlank = board[x][y] == blank
                if currisBlank != isBlank:
                    nowTransNum += 1
                    currisBlank = isBlank
            if  currisBlank:   
                nowTransNum += 1
            totalTransNum += nowTransNum
        return totalTransNum   

    # 空洞个数
    def emptyHoles(self, board):
        totalEmptyHoles = 0
        for x in range(boardwidth):
            y = 0
            emptyHoles = 0
            while y < boardheight:
                # print(board[x][y])
                if board[x][y]!=blank:
                    y += 1
                    break
                y += 1 
            while y < boardheight:
                # print(board[x][y])                
                if board[x][y]==blank:
                    emptyHoles += 1
                y += 1
            totalEmptyHoles += emptyHoles
        return totalEmptyHoles

    # 井的个数
    def wellNums(self, board):
        totalWellDepth  = 0
        wellDepth = 0
        tDepth = 0
        # 获取左边的井数
        for y in range(boardheight):            
            if board[0][y] == blank and board[1][y] != blank:
                tDepth += 1
            else:
                wellDepth += tDepth * (tDepth+1) / 2    
                tDepth = 0
        wellDepth += tDepth * (tDepth+1) / 2  
        totalWellDepth += wellDepth
        # 获取中间的井数
        wellDepth = 0.
        for x in range(1,boardwidth-1):
            tDepth = 0.
            for y in range(boardheight):
                if board[x][y]==blank and board[x-1][y]!=blank and board[x+1][y]!=blank:
                    tDepth += 1
                else:
                    wellDepth += tDepth * (tDepth+1) / 2
                    tDepth = 0
            wellDepth += tDepth * (tDepth+1) / 2
        totalWellDepth += wellDepth
        # 获取最右边的井数
        wellDepth = 0
        tDepth = 0
        for y in range(boardheight):
            if board[boardwidth-1][y] == blank and board[boardwidth-2][y] != blank:
                tDepth += 1
            else:
                wellDepth += tDepth * (tDepth +1 )/2
                tDepth = 0
        wellDepth += tDepth * (tDepth +1 )/2
        totalWellDepth += wellDepth
        return totalWellDepth        


    # 修改了价值评估 下落高度 消行个数 行变化次数 列变化次数 空洞个数 井的个数
    def calcreward(self,board,shape,completelines):
        _landingHeight = self.landingHeight(board, shape)
        _rowsEliminated = completelines * boardwidth
        _rowTransitions  = self.rowTransitions(board)
        _colTransitions = self.colTransitions(board)
        _emptyHoles = self.emptyHoles(board)
        _wellNums = self.wellNums(board)
        return -4.500158825082766 * _landingHeight \
                    + 3.4181268101392694 * _rowsEliminated \
                    + -3.2178882868487753 * _rowTransitions \
                    + -9.348695305445199 * _colTransitions \
                    + -7.899265427351652 * _emptyHoles \
                    + -3.3855972247263626 * _wellNums;                 

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
DEBUG = True    # 是否开启调试 到程序目录执行 tensorboard --logdir=game_model ，访问 http://127.0.0.1:6006
ACTIONS_COUNT = 3  # 可选的动作，针对 左移 翻转 右移
FUTURE_REWARD_DISCOUNT = 0.99  # 下一次奖励的衰变率 
OBSERVATION_STEPS = 5000.  # 在学习前观察的次数
EXPLORE_STEPS = 500000.  # 每次机器自动参与的概率的除数
INITIAL_RANDOM_ACTION_PROB = 1.0  # 随机移动的最大概率 1.0
FINAL_RANDOM_ACTION_PROB = 0.05  # 随机移动的最小概率
MEMORY_SIZE = 10000  # 记住的观察队列
MINI_BATCH_SIZE = 100  # 每次学习的批次
STATE_FRAMES = 4  # 每次保存的状态数
RESIZED_SCREEN_X, RESIZED_SCREEN_Y = (80, 100)   # 图片缩小后的尺寸 
OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)
SAVE_EVERY_X_STEPS = 1000  # 每学习多少轮后保存
STORE_SCORES_LEN = 100.     # 分数保留的长度
TRAIN_GAME_MAX_STEP = 1     # 学习的方块数
LEARNING_RATE = 1e-6        # 学习速率

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

# CNN网络
def get_network(x, output_size, filter_size=[3,3,3,3], filter_nums=[32,32,64,64], pool_scale=[2,2,2,2], full_nums=128, output_name="output_layer"):
    def get_w_b(w_shape,w_name="w",b_name="b"):
        w = tf.get_variable(w_name, w_shape, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(b_name, [w_shape[-1]], initializer=tf.constant_initializer(0.01))
        if DEBUG:
            tf.summary.histogram('w'.format(i), w)
            tf.summary.histogram('b'.format(i), b)
        return w,b    

    filter_layers=[]
    for i in range(len(filter_size)):         
        with tf.variable_scope('filter-layer-{}'.format(i)):
            if i == 0:
                input = x
            else:
                input = filter_layers[-1]
            w,b = get_w_b([filter_size[i],filter_size[i],int(input.get_shape()[-1]),filter_nums[i]])
            filter_layer = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME"), b))
            pool_layer = tf.nn.max_pool(filter_layer, ksize=[1, pool_scale[i], pool_scale[i], 1], strides=[1, pool_scale[i], pool_scale[i], 1], padding='SAME')
            filter_layers.append(pool_layer)
            if DEBUG:
                filter_map = tf.transpose(filter_layer[-1], perm=[2, 0, 1])
                filter_map = tf.reshape(filter_map, (filter_nums[i], int(filter_map.get_shape()[1]), int(filter_map.get_shape()[2]), 1))
                tf.summary.image('filterlayer', tensor=filter_map, max_outputs=filter_nums[i])

    full_connects=[]
    with tf.variable_scope('full-connect'):
        _out = filter_layers[-1]
        in_size = int(_out.get_shape()[1]) * int(_out.get_shape()[2]) * int(_out.get_shape()[3])
        input = tf.reshape(_out, [-1, in_size])
        w,  b = get_w_b([in_size,full_nums])
        full_connect = tf.nn.relu(tf.matmul(input, w) + b)
        full_connects.append(full_connect)

    with tf.variable_scope('output'):
        w, b = get_w_b([full_nums, output_size])
        output = tf.nn.bias_add(tf.matmul(full_connects[-1], w) , b, name=output_name)
        return output

# 学习
def train():    
    # 输入的图片，是每4张一组
    _input_layer =  tf.placeholder("float", [None, RESIZED_SCREEN_X, RESIZED_SCREEN_Y, STATE_FRAMES], name='input_layer')   
    _output_layer = get_network(_input_layer, ACTIONS_COUNT)
    
    _action = tf.placeholder("float", [None, ACTIONS_COUNT])    # 移动的方向
    _target = tf.placeholder("float", [None])                   # 得分

    # 将预测的结果和移动的方向相乘，按照第二维度求和 [0.1,0.2,0.7] * [0, 1, 0] = [0, 0.2 ,0] = [0.2]  得到当前移动的概率
    readout_action = tf.reduce_sum(tf.multiply(_output_layer, _action), reduction_indices=1)
    # 将（结果和评价相减）的平方，再求平均数。 得到和评价的距离。
    cost = tf.reduce_mean(tf.square(_target - readout_action))

    # 定义学习速率和优化方法,因为大部分匹配都是0，所以学习速率必需订的非常小
    global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(1e-6, global_step, 100000, 0.98, staircase=True)
    # 学习函数
    _train_operation = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost, global_step=global_step)

    _observations = deque()
    _last_scores = deque()

    # 设置最后一步是固定
    _last_action = KEY_LEFT
    _last_state = None          #4次的截图
    _probability_of_random_action = INITIAL_RANDOM_ACTION_PROB

    game = Tetromino()

    _session = tf.Session()       
    _session.run(tf.global_variables_initializer())

    _saver,_model_dir,_checkpoint_path = restore(_session)

    if DEBUG:
        tf.summary.scalar("cost", cost)
        # tf.summary.scalar("learning_rate", learning_rate)        
        _train_summary_op = tf.summary.merge_all()
        _train_summary_writer = tf.summary.FileWriter(_model_dir, _session.graph)
    _min_reward = 20000.
    _max_reward = -20000.
    _game_step  = 0
    while True:
        reward, image, terminal = game.step(list(_last_action))

        for event in pygame.event.get():  # 需要事件循环，否则白屏
            if event.type == QUIT:
                pygame.quit()
                sys.exit()         

        #将奖励分数归一化
        if reward!=0:
            if reward < _min_reward:
                _min_reward = reward
            elif reward > _max_reward:
                _max_reward = reward
            _avg_reward =  (_max_reward+_min_reward)/2
            if _avg_reward == 0:
                continue
            reward = (reward - _avg_reward) / _avg_reward  
            _game_step += 1

        image = cv2.resize(image,(RESIZED_SCREEN_Y, RESIZED_SCREEN_X))
        screen_resized_grayscaled = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _, screen_resized_binary = cv2.threshold(screen_resized_grayscaled, 1, 255, cv2.THRESH_BINARY)
       
        if reward != 0.0:
            _last_scores.append(reward)
            if len(_last_scores) > STORE_SCORES_LEN:
                _last_scores.popleft()
        if _last_state is None:  # 填充第一次的4张图片            
            _last_state = np.stack(tuple(screen_resized_binary for _ in range(STATE_FRAMES)), axis=2)

        screen_resized_binary = np.reshape(screen_resized_binary, (RESIZED_SCREEN_X, RESIZED_SCREEN_Y, 1))
        current_state = np.append(_last_state[:, :, 1:], screen_resized_binary, axis=2)

        _observations.append((_last_state, _last_action, reward, current_state, terminal))
        if len(_observations) > MEMORY_SIZE:
            _observations.popleft()
        
        if len(_observations) > OBSERVATION_STEPS:
            mini_batch = random.sample(_observations, MINI_BATCH_SIZE)
            previous_states = [d[OBS_LAST_STATE_INDEX] for d in mini_batch]
            actions = [d[OBS_ACTION_INDEX] for d in mini_batch]
            rewards = [d[OBS_REWARD_INDEX] for d in mini_batch]
            current_states = [d[OBS_CURRENT_STATE_INDEX] for d in mini_batch]
            terminals = [d[OBS_TERMINAL_INDEX] for d in mini_batch]

            agents_expected_reward = []
            agents_reward_per_action = _session.run(_output_layer, feed_dict={_input_layer: current_states})
            for i in range(len(mini_batch)):
                if terminals[i]:    # 如果是游戏结束没有下一步奖励
                    agents_expected_reward.append(rewards[i])
                else:
                    agents_expected_reward.append(rewards[i] + FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))

            if DEBUG:
                _, _step, train_summary_op =  _session.run([_train_operation,global_step,_train_summary_op], feed_dict={_input_layer: previous_states,_action: actions,
                        _target: agents_expected_reward})
            else:            
                _, _step = _session.run([_train_operation,global_step], feed_dict={_input_layer: previous_states,_action: actions,
                        _target: agents_expected_reward})

            if _step % SAVE_EVERY_X_STEPS == 0:
                _saver.save(_session, _checkpoint_path, global_step=_step)
                if DEBUG:
                    _train_summary_writer.add_summary(train_summary_op, _step)
                print("step: %s random_action_prob: %s reward %s scores differential %s" %
                  (_step, _probability_of_random_action, reward, sum(_last_scores) / STORE_SCORES_LEN))

        _last_state = current_state

        # 游戏执行下一步,按概率选择下一次是随机还是机器进行移动
        _last_action = np.zeros([ACTIONS_COUNT],dtype=np.int)
        if random.random() <= _probability_of_random_action and _game_step == TRAIN_GAME_MAX_STEP :
            action_index = random.randrange(ACTIONS_COUNT)
        else:
            readout_t = _session.run(_output_layer, feed_dict={_input_layer: [_last_state]})[0]
            action_index = np.argmax(readout_t)
        _last_action[action_index] = 1

        # 随机移动的概率逐步降低
        if _probability_of_random_action > FINAL_RANDOM_ACTION_PROB and len(_observations) > OBSERVATION_STEPS:
            _probability_of_random_action -= (INITIAL_RANDOM_ACTION_PROB - FINAL_RANDOM_ACTION_PROB) / EXPLORE_STEPS
 
        if  _game_step >= TRAIN_GAME_MAX_STEP:
            _game_step = 0
            game.reset()

if __name__ == '__main__':
    train()