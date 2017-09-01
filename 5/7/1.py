# coding=utf-8

import tensorflow as tf
import numpy as np
import os
from utils import readImgFile, img2vec, show
import time
import random

curr_dir = os.path.dirname(__file__)

# 图片的高度为12X256，宽度为不定长
image_size = (12,256)

#LSTM
num_hidden = 64
num_layers = 1

chars = u"阿富汗巴林孟加拉国不丹文莱缅甸柬埔寨塞浦路斯朝鲜香港印度尼西亚伊朗克以色列日本约旦科威特老挝黎嫩澳门马来尔代夫蒙古泊联邦民主共和曼基坦勒菲律宾卡塔沙伯新坡韩里兰叙利泰土耳其酋也越南中台澎金关税区东帝汶哈萨吉库乌兹别洲他家(地)及安哥贝宁博茨瓦纳布隆迪喀麦那群岛佛得角非卜休达乍摩罗刚果提埃赤道几内俄比蓬冈绍肯维毛求洛莫桑米留汪卢旺圣多美普舌昂索撒苏突干法赞津韦托梅士厄立时英德爱意大森堡荷希腊葡萄牙班奥保芬直陀匈冰支敦登挪波力诺瑞典脱陶宛格鲁拜疆白黑山捷伐前顿梵蒂城欧瓜根廷族玻开智伦属圭危海洪都买墨拿秘各丁凯委京皮密陵百慕北斐济盖瑙努图福社会所汤艾帕劳浮洋详合机构际组织性包装原产市辖阳丰石景淀头沟房通州顺义昌平兴怀柔谷云延庆天河红桥丽青辰武清宝坻滨静县蓟省庄长华井陉矿裕藁鹿泉栾正定行唐灵寿高邑深泽皇无极元氏赵辛集晋乐冶润曹妃滦亭迁玉田遵化秦戴抚龙满自治邯郸丛复峰临漳成名涉磁肥乡永年邱鸡广馆魏曲周邢丘柏尧任巨宗宫竞秀莲池苑徐水涞阜容源望易蠡野雄涿碑店张口宣下花园康沽尚蔚万全崇礼承双鹰手营子宽围场沧运光盐肃吴献村回黄骅间廊坊次固厂霸三衡桃枣强饶故冀太小迎杏岭尖草坪娄烦交同郊荣镇浑左盂襄垣屯壶沁潞川朔阴应右仁榆权昔祁遥介湖猗闻喜稷绛夏陆芮忻府五繁峙神岢岚偏汾沃翼洞隰蒲侯霍吕梁离柳楼方孝呼浩赛罕默旗昆仑拐鄂九茂明勃湾松什腾翁牛喇敖汉辽后奈扎郭胜准杭锦审盟赉斡春温陈虎额彦淖磴察卓资商凉四王锡二连嘎珠穆仆寺镶蓝善沈姑铁于岗甘旅鞍千岫岩溪桓振凤凌站鲅鱼圈边细彰宏伟弓灯盘洼银调兵建票葫芦绥绿农树惠潭船蛟桦舒磐梨公江辉靖宇乾扶余洮们珲外依木志常齐锋碾讷冠恒滴麻鹤向工萝鸭贤友谊让胡肇杜岔好翠峦带星上嘉荫佳进风远七茄牡棱逊孙奎玛漠汇闸虹杨闵奉玄淮邺鼓栖霞雨六溧淳塘宜贾铜沛睢沂邳钟坛相熟仓如启皋赣灌涟盱眙响射扬邗仪征邮徒句姜堰宿豫沭泗浙拱墅萧桐庐曙鄞象姚慈瓯苍浔柯虞诸暨嵊婺衢游舟岱椒环仙居缙遂畲徽瑶蜀巢芜镜弋鸠为蚌埠禹庵谢八潘当涂含烈濉官狮观枞潜岳歙黟滁琅琊谯颍界首埇砀璧亳涡贵至郎泾绩旌尾闽厦思翔莆厢涵荔屿流尤将鲤芗霄诏政邵夷汀蕉屏柘鼎谱萍湘栗修彭渝分月章贡信犹寻峡袁载樟铅横鄱历槐崂李胶即淄薛峄儿滕垦烟芝罘牟招潍寒朐兖微祥邹乳照莒钢郯费聊莘茌沾棣菏单郓鄄郑管街巩荥封符杞许尉考瀍涧嵩汝偃师顶卫湛叶郏舞殷滑壁淇浚牧获焦作解放陟濮范鄢葛漯郾召陕渑卧淅邓浉始潢息项驻驿蔡舆确泌级划岸硚陂十茅箭郧竹伍点军猇秭归枝樊荆掇刀感悟梦监滋团浠蕲穴咸随曾恩施苗架芙蓉心麓浏株淞攸茶炎醴韶晖雁蒸耒步君汨澧植益赫沅郴桂禾零冷滩牌溆晃侗芷底泸凰丈番禺从增浈圳斗汕濠潮澄禅坎廉雷电端要紫壮莞揭榕郁良邕鸣融叠彩恭梧圩藤岑防钦覃绵业贺昭峨仫佬等凭琼棠涯崖儋指迈重涪渡坝碚綦足黔潼垫忠节巫柱酉羊堂郫邛崃沿攀蔺邡梓羌油剑阁犍研夹沐彝眉充部陇阆雅珙筠邻蓥渠经棉简藏理壤若孜孚炉稻拖觉冕烽真仡务湄习毕雍碧阡晴贞谟册亨秉穗匀瓮独呈禄劝麒麟傣冲巧蒗洱祜佤澜耿楚谋个旧弥砚畴版勐漾濞巍颇芒盈怒傈僳堆则迦结仲聂类隅乃囊措查错浪申戈札噶革改勤灞未央阎户耀渭岐彬旬功起勉略脂柞峪积祝掖崆峒酒煌岷宕两迭碌湟互助循晏久杂称谦令峻嘴吾磨坂碱吐鄯坤奇垒精音楞轮犁且末焉耆硕车恰疏附莎伽策敏蕴位可境先人币盾铢镑桶闭镀锌铝圆板纤塑料琵琶罐箱漏再生纸袋/席编薄膜硬璃瓷条筐膨箩笼物铺材散裸挂捆然座辆艘套只件把块卷副片份幅对棵筒盆具疋担扇盒亿伏升尺吨短司斤磅盎码寸毫制批打匹发枚粒瓶舱净种样标每品总航蹲领域企实验室素模页证50pm12934678PQ.RSNMVWL"


# 所有的字符加 + blank + ctc blank
num_classes = len(chars) + 1 + 1

#初始化学习速率
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 500
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
MOMENTUM = 0.9

BATCHES = 100
BATCH_SIZE = 64
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = 10

train_files = open(os.path.join(curr_dir, "data", "index.txt")).readlines()

def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,12]
    inputs = tf.placeholder(tf.float32, [None, None, image_size[0]])
    # 定义 ctc_loss 是稀疏矩阵
    labels = tf.sparse_placeholder(tf.int32)
    # 1维向量 序列长度 [batch_size,]
    seq_len = tf.placeholder(tf.int32, [None])
    # 定义 LSTM 网络
    # 可以为:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    
    # 第二个输出状态，不会用到
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)

    batch_s, max_timesteps = shape[0], shape[1]
    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")

    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    logits = tf.transpose(logits, (1, 0, 2))
    return logits, inputs, labels, seq_len, W, b


# 生成一个训练batch
def get_next_batch(batch_size=128):
    inputs = np.zeros([batch_size, image_size[1], image_size[0]])
    codes = []

    batch = random.sample(train_files, batch_size)

    for i, line in enumerate(batch):
        lines = line.split(" ")
        imageFileName = lines[0]+".png"
        text = lines[1].strip()
        image = readImgFile(os.path.join(curr_dir,"data",imageFileName))
        image_vec = img2vec(image,image_size[0],image_size[1])
        #np.transpose 矩阵转置 (12*256,) => (12,256) => (256,12)
        inputs[i,:] = np.transpose(image_vec.reshape((image_size[0],image_size[1])))
        #标签转成列表保存在codes
        text_list = []
        for char in text:
            text_list.append(chars.index(char))
        codes.append(text_list)
    #比如batch_size=2，两条数据分别是"12"和"1"，则labels [['1','2'],['1']]

    labels = [np.asarray(i) for i in codes]
    #labels转成稀疏矩阵
    sparse_labels = sparse_tuple_from(labels)
    #(batch_size,) sequence_length值都是256，最大划分列数
    seq_len = np.ones(inputs.shape[0]) * image_size[1]
    return inputs, sparse_labels, seq_len

# 转化一个序列列表为稀疏矩阵    
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
 
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result
    
def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = spars_tensor[1][m]
        decoded.append(str)
    return decoded

def list_to_chars(list):
    return "".join([chars[v] for v in list])

def train():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                                global_step,
                                                DECAY_STEPS,
                                                LEARNING_RATE_DECAY_FACTOR,
                                                staircase=True)
    logits, inputs, labels, seq_len, W, b = neural_networks()

    loss = tf.nn.ctc_loss(labels=labels,inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=MOMENTUM).minimize(cost, global_step=global_step)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))

    init = tf.global_variables_initializer()

    def report_accuracy(decoded_list, test_labels):
        original_list = decode_sparse_tensor(test_labels)
        detected_list = decode_sparse_tensor(decoded_list)
        true_numer = 0
        
        if len(original_list) != len(detected_list):
            print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                  " test and detect length desn't match")
            return
        print("T/F: original(length) <-------> detectcted(length)")
        for idx, number in enumerate(original_list):
            detect_number = detected_list[idx]
            hit = (number == detect_number)
            print(hit, list_to_chars(number), "(", len(number), ") <-------> ", list_to_chars(detect_number), "(", len(detect_number), ")")
            if hit:
                true_numer = true_numer + 1
        print("Test Accuracy:", true_numer * 1.0 / len(original_list))

    def do_report():
        test_inputs,test_labels,test_seq_len = get_next_batch(TEST_BATCH_SIZE)
        test_feed = {inputs: test_inputs,
                     labels: test_labels,
                     seq_len: test_seq_len}
        dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
        report_accuracy(dd, test_labels)
 
    def do_batch():
        train_inputs, train_labels, train_seq_len = get_next_batch(BATCH_SIZE)
        
        feed = {inputs: train_inputs, labels: train_labels, seq_len: train_seq_len}
        
        b_loss,b_labels, b_logits, b_seq_len,b_cost, steps, _ = session.run([loss, labels, logits, seq_len, cost, global_step, optimizer], feed)

        if steps > 0 and steps % REPORT_STEPS == 0:
            do_report()
        return b_cost, steps

    def restore(sess):
        curr_dir = os.path.dirname(__file__)
        model_dir = os.path.join(curr_dir, "model")
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        saver_prefix = os.path.join(model_dir, "model.ckpt")        
        ckpt = tf.train.get_checkpoint_state(model_dir)
        saver = tf.train.Saver(max_to_keep=5)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model ...")
            saver.restore(sess, ckpt.model_checkpoint_path)
        return saver, model_dir, saver_prefix

    with tf.Session() as session:
        session.run(init)
        saver, model_dir, checkpoint_path = restore(session) # tf.train.Saver(tf.global_variables(), max_to_keep=100)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():            
                train_cost = train_ler = 0
                for batch in range(BATCHES):
                    start = time.time()
                    c, steps = do_batch()
                    train_cost += c * BATCH_SIZE
                    seconds = time.time() - start
                    print("Step:", steps, ", Cost:", c, ", batch seconds:", seconds)
                
                # train_cost /= TRAIN_SIZE
                
                # train_inputs, train_labels, train_seq_len = get_next_batch(BATCH_SIZE)
                # val_feed = {inputs: train_inputs,
                #             labels: train_labels,
                #             seq_len: train_seq_len}
    
                # val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)
    
                # log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
                # print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cost, val_ler, time.time() - start, lr))
                saver.save(session, checkpoint_path, global_step=steps)
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train()