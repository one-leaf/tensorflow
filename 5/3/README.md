俄罗斯方块 AI

评价算法

http://imake.ninja/el-tetris-an-improvement-on-pierre-dellacheries-algorithm/
https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/

Pierre Dellacherie 算法
V（s）= -（Landing height）+（Eroded piece cells）–（Row transitions）–（Column transitions）- 4（Holes）–（Cumulative wells） 

rating = (-1.0) * landingHeight          + ( 1.0) * erodedPieceCellsMetric
         + (-1.0) * boardRowTransitions + (-1.0) * boardColTransitions
         + (-4.0) * boardBuriedHoles 　  + (-1.0) * boardWells;
其中，
landingHeight指当前落子落下去之后，落子中点距底部的方格数；
erodedPieceCellsMetric = 消去行 * 当前落子被消去的格子数；
boardRowTransitions指各行的“变换次数”之和，一行中从有方块到无方块、无方块到有方块被视为一次“变换”，游戏区域左右边界也视作有方块；boardColTransitions指各列的“变换次数”之和；
boardBuriedHoles指各列中间的“空洞”方格个数之和；空洞无论有多大，只算一个。一个图中可能有多个空洞
boardWells指各“井”的深度的连加到1的和之和，“井”指两边皆有方块的空列。（空洞也可以是井，一列中可能有多个井）。此值为所有的井以1为公差首项为1的等差数列的总和   例：共三个井，2，3，1   wellsum=（1+2）+（1+2+3）+1；

它们的权值分别为
1 -4.500158825082766
2 3.4181268101392694
3 -3.2178882868487753
4 -9.348695305445199
5 -7.899265427351652
6 -3.3855972247263626

评价还包括优先度。优先度在两个局面的评分相同时发挥作用，取评分相同但优先度高者。优先度的计算方法为：

若落子落于左侧:priority = 100 * 落子水平平移格子数 + 10 + 落子旋转次数;

若落子落于右侧:priority = 100 * 落子水平平移格子数 + 落子旋转次数;

比较每一种落法的评分与优先度。在同为最高评分的落法中，取优先度最高者。

其它：

http://colinfahey.com/tetris/ApplyingReinforcementLearningToTetris_DonaldCarr_RU_AC_ZA.pdf

http://www.bbsmax.com/A/QW5YYKXN5m/

性能：

LEARNING_RATE = 1e-3        
filter_size=[8,8,5,5,3], filter_nums=[32,32,32,32,32], pool_scale=[2,2,2,2,2], pool_type=[0,0,0,0,0],full_nums=32,
STEP：Fail
TIME：Fail

LEARNING_RATE = 1e-4        
filter_size=[8,8,5,5,3], filter_nums=[32,32,32,32,32], pool_scale=[2,2,2,2,2], pool_type=[0,0,0,0,0],full_nums=32,
STEP：375000
TIME：49974 秒

LEARNING_RATE = 1e-6
filter_size=[8,8,5,5,3], filter_nums=[32,32,32,32,32], pool_scale=[2,2,2,2,2], pool_type=[0,0,0,0,0],full_nums=32,
STEP：697000       
TIME: 93728  秒

LEARNING_RATE = 1e-4        
filter_size=[3,3,3,3,3], filter_nums=[32,32,32,32,32], pool_scale=[2,2,2,2,2], pool_type=[0,0,0,0,0],full_nums=32, 
STEP：179000
TIME：18412 秒

LEARNING_RATE = 1e-4        
filter_size=[3,3,3,3,3,3], filter_nums=[32,32,32,32,32,32], pool_scale=[2,2,2,2,2,2], 
        pool_type=[0,0,0,0,0,0],full_nums=32, 
STEP：125000
TIME：13088 秒

LEARNING_RATE = 1e-4
filter_size=[3,3,3,3,3,3,3], filter_nums=[32,32,32,32,32,32,32], pool_scale=[2,2,2,2,2,2,2], 
        pool_type=[0,0,0,0,0,0,0],full_nums=32,
STEP：344000
TIME：35372 秒

LEARNING_RATE = 1e-5
filter_size=[8,8,5,5,3], filter_nums=[32,32,32,32,32], pool_scale=[2,2,2,2,2], pool_type=[0,0,0,0,0],full_nums=4096,
STEPSTOP: 第3步失败

LEARNING_RATE = 1e-5
filter_size=[8,8,5,5,3], filter_nums=[32,32,32,32,32], pool_scale=[2,2,2,2,2], pool_type=[0,0,0,0,0],full_nums=384*5,
STEPSTOP: 第3步失败

LEARNING_RATE = 1e-7
filter_size=[3,3,3,3,3], filter_nums=[32,32,32,32,32], pool_scale=[2,2,2,2,2], pool_type=[0,0,0,0,0],full_nums=384*5,
STEPSTOP: 第2步失败 

LEARNING_RATE = 1e-5
filter_size=[3,3,3,3,3,3,3,3,3,3], filter_nums=[32,32,32,32,32,32,32,32,32,32], pool_scale=[2,2,2,2,2,-1,-1,-1,-1,-1], 
        pool_type=[0,0,0,0,0,-1,-1,-1,-1,-1],full_nums=384,
game_max_step = 1
STEPSTOP: 第3步失败

LEARNING_RATE = 1e-5
filter_size=[3,3,3,3,3,3,3,3,3,3], filter_nums=[32,32,32,32,32,32,32,32,32,32], pool_scale=[2,2,2,2,2,-1,-1,-1,-1,-1], 
        pool_type=[0,0,0,0,0,-1,-1,-1,-1,-1],full_nums=384,
game_max_step = 10
STEPSTOP: ?