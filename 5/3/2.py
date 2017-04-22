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
import copy

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

KEY_ROTATION  = [0,1,0]
KEY_LEFT      = [1,0,0]
KEY_RIGHT     = [0,0,1]

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
        self.rewards, self.reward_r, self.reward_x = self.calcAllRewards(self.board, self.fallpiece)      

    def step(self, action):
        reward  = 0
        moveleft = False
        moveright = False
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
            reward = self.calcReward(self.board, self.fallpiece)
            # reward = self.softmax(reward,self.rewards)            
            self.score += self.removecompleteline(self.board)            
            level = self.calculate(self.score)   

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
                # reward = self.softmax(reward,self.rewards)
                return reward, screen_image, is_terminal, shape, self.rewards  # 虽然游戏结束了，但还是正常返回分值，而不是返回 -1
            self.rewards, self.reward_r, self.reward_x  = self.calcAllRewards(self.board, self.fallpiece) # 计算下一步最佳分值
        return reward, screen_image, is_terminal, shape, self.rewards

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
   
    # 本次下落的方块中点地板的距离
    def landingHeight(self,board,piece):
        shape=pieces[piece['shape']][piece['rotation']]
        for y in range(templatenum):
            for x in range(templatenum):
                if shape[x][y] != blank:
                    return boardheight - (piece['y'] + y)

    # 本次下落后此方块贡献（参与完整行组成的个数）*完整行的行数
    def rowsEliminated(self,board,piece):
        eliminatedNum = 0
        eliminatedGridNum = 0
        shape=pieces[piece['shape']][piece['rotation']]
        for y in range(boardheight):
            flag = True
            for x in range(boardwidth):
                if board[x][y] == blank:
                    flag = False
                    break
            if flag:
                eliminatedNum += 1
                if (y>piece['y']) and (y <piece['y']+templatenum):
                    for s in range(templatenum):
                        if shape[y-piece['y']][s] != blank:
                             eliminatedGridNum += 1
        return eliminatedNum * eliminatedGridNum

    # 在同一行，方块 从无到有 或 从有到无 算一次（边界算有方块）
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

    # 在同一列，方块 从无到有 或 从有到无 算一次（边界算有方块）
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

    # 空洞的数量。空洞无论有多大，只算一个。一个图中可能有多个空洞
    def emptyHoles(self, board):
        totalEmptyHoles = 0
        for x in range(boardwidth):
            y = 0
            emptyHoles = 0
            while y < boardheight:
                if board[x][y]!=blank:
                    y += 1
                    break
                y += 1 
            while y < boardheight:
                if board[x][y]==blank:
                    emptyHoles += 1
                y += 1
            totalEmptyHoles += emptyHoles
        return totalEmptyHoles

    # 井就是两边都有方块的空列。（空洞也可以是井，一列中可能有多个井）。此值为所有的井以1为公差首项为1的等差数列的总和
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
    def calcReward(self,board,piece):
        _landingHeight = self.landingHeight(board, piece)
        _rowsEliminated = self.rowsEliminated(board, piece)
        _rowTransitions  = self.rowTransitions(board)
        _colTransitions = self.colTransitions(board)
        _emptyHoles = self.emptyHoles(board)
        _wellNums = self.wellNums(board)
        print("shape",piece['shape'],"rotation",piece['rotation'],"x",piece['x'],"y",piece["y"])
        print("_landingHeight",_landingHeight,"_rowsEliminated",_rowsEliminated)
        print("_rowTransitions",_rowTransitions,"_colTransitions",_colTransitions)
        print("_emptyHoles",_emptyHoles,"_wellNums",_wellNums)
        print("===============================")
        return -4.500158825082766 * _landingHeight \
                    + 3.4181268101392694 * _rowsEliminated \
                    + -3.2178882868487753 * _rowTransitions \
                    + -9.348695305445199 * _colTransitions \
                    + -7.899265427351652 * _emptyHoles \
                    + -3.3855972247263626 * _wellNums;     

    # 在游戏开始就计算多有的可能分值
    def calcAllRewards(self,board,piece):
        rewards=[]
        rotationCount = len(pieces[piece['shape']]) 
        maxReward=-10000
        x_reward=0
        r_reward=0
        for r in range(rotationCount):
            m_piece = copy.deepcopy(piece)  
            m_piece['rotation']=r
            for x in range(boardwidth+10):
                m_board =  copy.deepcopy(board)
                m_piece['x']=x-5                
                for y in range(boardheight+10):
                    m_piece['y']=y-1  
                    if not self.validposition(m_board, m_piece):
                        continue

                    if not self.validposition(m_board, m_piece, ay = 1):
                        self.addtoboard(m_board,m_piece)
                        reward = self.calcReward(m_board, m_piece)
                        if reward > maxReward :
                            maxReward = reward
                            x_reward=m_piece['x']
                            r_reward=r
                        rewards.append(reward)
                        break
        print(rewards,r_reward,x_reward )                        
        return rewards,r_reward,x_reward

    def main(self):
        while True:
            # time.sleep(1)
            for event in pygame.event.get():  # 需要事件循环，否则白屏
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()    
            if self.fallpiece['rotation']!=self.reward_r:
                self.step(KEY_ROTATION)
                continue
            if self.fallpiece['x']>self.reward_x:
                self.step(KEY_LEFT)
                continue
            if self.fallpiece['x']<self.reward_x:
                self.step(KEY_RIGHT)
                continue
            self.step(None)
                    

if __name__ == '__main__':
    tetromino = Tetromino()
    tetromino.main()