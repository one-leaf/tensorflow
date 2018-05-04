# 运行流程整理
- _train.py_
- common_flags.create_dataset建立数据
- model = common_flags.create_model 建立模型
- model.create_base
    + _model.py_
    + b: batch_size, n: num_char_classes, s: seq_length 
    + conv_tower_fn 创建CNN，inception_v3
    + encode_coordinates_fn 插入坐标信息
    + sequence_logit_fn 序列模型
        - layer_class = sequence_layers.get_layer_class 返回 AttentionWithAutoregression
        - layer = layer_class(net, labels_one_hot..)
            + 创建 _softmax_w _softmax_b [b, n]
            + 创建 _zero_label zeros [b, n]
        - layer.create_logits()
            + 定义 decoder_inputs [0,None,None.....] -- [s]
            + unroll_cell 定义LSTM 模型
                - 直接返回 tf.contrib.legacy_seq2seq.attention_decoder
                - 参数包括初始化 decoder_inputs 和 get_input 下一个输入回调
                - get_input -- get_train_input -- get_eval_input -- 
                - if 0 _zero_label else char_one_hot(char_logit(prev, i)) 
                - char_logit(i) -- (prev * w +b),每一个位置都单独做了一个全连接层 返回 [1]
            + lstm [b,s,c] ==> [b, [s,c]*w+b]==>[s,n]] ==> [s,1,n]] ==> [b, s, 1, n]


- model.create_loss
- model.create_summaries
- model.create_init_fn_to_restore
- train