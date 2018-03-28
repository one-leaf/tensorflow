import numpy as np

class Config:
    MEAN=np.float32([102.9801, 115.9465, 122.7717])
    #MEAN=np.float32([100.0, 100.0, 100.0])
    TEST_GPU_ID=0
    SCALE=900
    MAX_SCALE=1500
    TEXT_PROPOSALS_WIDTH=0
    MIN_RATIO=0.01
    LINE_MIN_SCORE=0.6
    TEXT_LINE_NMS_THRESH=0.3
    MAX_HORIZONTAL_GAP=30
    TEXT_PROPOSALS_MIN_SCORE=0.7
    TEXT_PROPOSALS_NMS_THRESH=0.3
    MIN_NUM_PROPOSALS=0
    MIN_V_OVERLAPS=0.6
    MIN_SIZE_SIM=0.6