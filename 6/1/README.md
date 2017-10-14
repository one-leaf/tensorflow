写一个简单的ocr引擎

思路：

1 获得图片分割成小单元
2 将小单元统一到同一尺寸
3 学习,采用 CNN-LSTM-CTC 模型


参考：  
https://github.com/JinpengLI/deep_ocr
https://github.com/pannous/tensorflow-ocr
http://www.jianshu.com/p/45828b18f133
https://github.com/synckey/tensorflow_lstm_ctc_ocr/
http://ilovin.me/2017-04-06/tensorflow-lstm-ctc-ocr/
http://ilovin.me/2017-04-23/tensorflow-lstm-ctc-input-output/
很有意思：
https://github.com/fchollet/keras/blob/master/examples/image_ocr.py
https://github.com/mozilla/DeepSpeech/blob/master/DeepSpeech.py