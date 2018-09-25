# 测试 PNASNet5
# 参考 https://github.com/chenxi116/PNASNet.TF

import tensorflow as tf
arg_scope = tf.contrib.framework.arg_scope
slim = tf.contrib.slim

# 定义NAS单元
class NASBaseCell(object):
  """NASNet Cell class that is used as a 'layer' in image architectures.
  Args:
    num_conv_filters: The number of filters for each convolution operation.
    operations: List of operations that are performed in the NASNet Cell in
      order.
    used_hiddenstates: Binary array that signals if the hiddenstate was used
      within the cell. This is used to determine what outputs of the cell
      should be concatenated together.
    hiddenstate_indices: Determines what hiddenstates should be combined
      together with the specified operations to create the NASNet cell.
  """

  def __init__(self, num_conv_filters, operations, used_hiddenstates,
               hiddenstate_indices, drop_path_keep_prob, total_num_cells,
               total_training_steps):
    assert len(hiddenstate_indices) == len(operations)
    assert len(operations) % 2 == 0
    self._num_conv_filters = num_conv_filters
    self._operations = operations
    self._used_hiddenstates = used_hiddenstates
    self._hiddenstate_indices = hiddenstate_indices
    self._drop_path_keep_prob = drop_path_keep_prob
    self._total_num_cells = total_num_cells
    self._total_training_steps = total_training_steps

  
  def __call__(self, net, scope, filter_scaling, stride, prev_layer, cell_num):
    self._cell_num = cell_num
    self._filter_scaling = filter_scaling
    self._filter_size = int(self._num_conv_filters * filter_scaling)
    # cell_num:        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # _filter_scaling: [0.25, 0.5, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4]
    # _filter_size:    [54, 108, 216, 216, 216, 216, 432, 432, 432, 432, 864, 864, 864, 864]
    # _used_hiddenstates:   [1, 1, 0, 0, 0, 0, 0]
    # _hiddenstate_indices: [1, 1, 0, 0, 0, 0, 4, 0, 1, 0]

    with tf.variable_scope(scope):
      # 返回了2个值，[net, prev_layer]
      net = self._cell_base(net, prev_layer)
      # 0         net: [(?, 7, 7, 270), (?, 13, 13, 96)]
      # 1,2,3,4,5 net: [(?, 4, 4, 216), (?, 4, 4, 216)]
      # 6         net: [(?, 4, 4, 432), (?, 4, 4, 432)]
      # 7,8,9     net: [(?, 2, 2, 432), (?, 2, 2, 432)]
      # 10        net: [(?, 2, 2, 864), (?, 2, 2, 864)]
      # 11,12,13  net: [(?, 1, 1, 864), (?, 1, 1, 864)]

      # 循环次数 10/2 = 5
      for i in range(int(len(self._operations) / 2)):
        with tf.variable_scope('comb_iter_{}'.format(i)):
          h1 = net[self._hiddenstate_indices[i * 2]]
          h2 = net[self._hiddenstate_indices[i * 2 + 1]]
          with tf.variable_scope('left'):
            h1 = self._apply_operation(h1, self._operations[i * 2], stride, 
                                       self._hiddenstate_indices[i * 2] < 2)
          with tf.variable_scope('right'):
            h2 = self._apply_operation(h2, self._operations[i * 2 + 1], stride,
                                       self._hiddenstate_indices[i * 2 + 1] < 2)
          with tf.variable_scope('combine'):
            h = h1 + h2
          net.append(h)

      with tf.variable_scope('cell_output'):
        net = self._combine_unused_states(net)

      return net

  def _cell_base(self, net, prev_layer):
    # cell_num: 0 net: (?, 13, 13, 96)  prev_layer: None  filter_size: 54
    # cell_num: 1 net: (?, 7, 7, 270)   prev_layer: (?, 13, 13, 96) filter_size: 108
    # cell_num: 2 net: (?, 4, 4, 540)   prev_layer: (?, 7, 7, 270)  filter_size: 216
    # cell_num: 3 net: (?, 4, 4, 1080)  prev_layer: (?, 4, 4, 540)  filter_size: 216
    # cell_num: 4 net: (?, 4, 4, 1080)  prev_layer: (?, 4, 4, 1080) filter_size: 216
    # cell_num: 5 net: (?, 4, 4, 1080)  prev_layer: (?, 4, 4, 1080) filter_size: 216
    # cell_num: 6 net: (?, 4, 4, 1080)  prev_layer: (?, 4, 4, 1080) filter_size: 432
    # cell_num: 7 net: (?, 2, 2, 2160)  prev_layer: (?, 4, 4, 1080) filter_size: 432
    # cell_num: 8 net: (?, 2, 2, 2160)  prev_layer: (?, 2, 2, 2160) filter_size: 432
    # cell_num: 9 net: (?, 2, 2, 2160)  prev_layer: (?, 2, 2, 2160) filter_size: 432
    # cell_num: 10 net: (?, 2, 2, 2160) prev_layer: (?, 2, 2, 2160) filter_size: 864
    # cell_num: 11 net: (?, 1, 1, 4320) prev_layer: (?, 2, 2, 2160) filter_size: 864
    # cell_num: 12 net: (?, 1, 1, 4320) prev_layer: (?, 1, 1, 4320) filter_size: 864
    # cell_num: 13 net: (?, 1, 1, 4320) prev_layer: (?, 1, 1, 4320) filter_size: 864

    filter_size = self._filter_size
    if prev_layer is None:
      prev_layer = net
    elif net.shape[2] != prev_layer.shape[2]:
      prev_layer = tf.nn.relu(prev_layer)
      # 将图片分割为2个独立进行CNN再合并最后项，达到图片高宽一致的效果
      prev_layer = self._factorized_reduction(prev_layer, filter_size, stride=2)
    elif filter_size != prev_layer.shape[3]:
      prev_layer = tf.nn.relu(prev_layer)
      prev_layer = slim.conv2d(prev_layer, filter_size, 1, scope='prev_1x1')
      prev_layer = slim.batch_norm(prev_layer, scope='prev_bn')

    # cell_num: 0 prev_layer: (?, 13, 13, 96) 
    # cell_num: 1 prev_layer: (?, 7, 7, 108)
    # cell_num: 2 prev_layer: (?, 4, 4, 216)
    # cell_num: 3，4，5 prev_layer: (?, 4, 4, 216)
    # cell_num: 6 prev_layer: (?, 4, 4, 432)
    # cell_num: 7，8，9 prev_layer: (?, 2, 2, 432)
    # cell_num: 10 prev_layer: (?, 2, 2, 864)
    # cell_num: 11，12，13 prev_layer: (?, 1, 1, 864)

    net = tf.nn.relu(net)
    net = slim.conv2d(net, filter_size, 1, scope='1x1')
    net = slim.batch_norm(net, scope='beginning_bn')

    # cell_num: 0 net: (?, 13, 13, 54)
    # cell_num: 1 net: (?, 7, 7, 108)
    # cell_num: 2，3，4，5 net: (?, 4, 4, 216)
    # cell_num: 6 net: (?, 4, 4, 432)
    # cell_num: 7，8，9 net: (?, 2, 2, 432)
    # cell_num: 10 net: (?, 2, 2, 864)
    # cell_num: 11，12，13 net: (?, 1, 1, 864)

    net = tf.split(axis=3, num_or_size_splits=1, value=net)
    # net: [net]

    for split in net:
      assert split.shape[3] == filter_size
    net.append(prev_layer)
    # net: [net, prev_layer]
 
    return net

  # 针对每个CELL单独做卷积
  def _apply_operation(self, net, operation, stride, is_from_original_input):
    if stride > 1 and not is_from_original_input:
      stride = 1
    input_filters = net.shape[3]
    filter_size = self._filter_size
    if 'separable' in operation:
      num_layers = int(operation.split('_')[-1])
      kernel_size = int(operation.split('x')[0][-1])
      for layer_num in range(num_layers):
        net = tf.nn.relu(net)
        # 可分卷积
        net = slim.separable_conv2d(
            net,
            filter_size,
            kernel_size,
            depth_multiplier=1,
            scope='separable_{0}x{0}_{1}'.format(kernel_size, layer_num + 1),
            stride=stride)
        net = slim.batch_norm(
            net, scope='bn_sep_{0}x{0}_{1}'.format(kernel_size, layer_num + 1))
        stride = 1
    elif operation in ['none']:
      if stride > 1 or (input_filters != filter_size):
        net = tf.nn.relu(net)
        net = slim.conv2d(net, filter_size, 1, stride=stride, scope='1x1')
        net = slim.batch_norm(net, scope='bn_1')
    elif 'pool' in operation:
      pooling_type = operation.split('_')[0]
      pooling_shape = int(operation.split('_')[-1].split('x')[0])
      if pooling_type == 'avg':
        net = slim.avg_pool2d(net, pooling_shape, stride=stride, padding='SAME')
      elif pooling_type == 'max':
        net = slim.max_pool2d(net, pooling_shape, stride=stride, padding='SAME')
      else:
        raise ValueError('Unimplemented pooling type: ', pooling_type)
      if input_filters != filter_size:
        net = slim.conv2d(net, filter_size, 1, stride=1, scope='1x1')
        net = slim.batch_norm(net, scope='bn_1')
    else:
      raise ValueError('Unimplemented operation: ', operation)

    if operation != 'none':
      net = self._apply_drop_path(net)
    return net

  def _combine_unused_states(self, net):
    used_hiddenstates = self._used_hiddenstates
    states_to_combine = (
        [h for h, is_used in zip(net, used_hiddenstates) if not is_used])
    net = tf.concat(values=states_to_combine, axis=3)
    return net

  # 逐步增加dropout，防止早期局部过拟合
  # 没有用标准的dropout
  def _apply_drop_path(self, net):
    drop_path_keep_prob = self._drop_path_keep_prob
    if drop_path_keep_prob < 1.0:
      # Scale keep prob by layer number
      assert self._cell_num != -1
      # [1/14, 2/14 ... 14/14]
      layer_ratio = (self._cell_num + 1) / float(self._total_num_cells)
      # [0.96, ... 0.5 ] 
      drop_path_keep_prob = 1 - layer_ratio * (1 - drop_path_keep_prob)
      # Decrease keep prob over time
      current_step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
      # [0.0001, ... 1]
      current_ratio = tf.minimum(1.0, current_step / self._total_training_steps)
      # [1, ... 0.5 ]
      drop_path_keep_prob = 1 - current_ratio * (1 - drop_path_keep_prob)
      # Drop path
      noise_shape = [net.shape[0], 1, 1, 1]
      random_tensor = drop_path_keep_prob
      random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
      # 向下取整，[0,1]
      # 越往后面 0，会增加
      binary_tensor = tf.cast(tf.floor(random_tensor), net.dtype)
    
      # 同时将留下的数据进行放大，防止过拟合
      keep_prob_inv = tf.cast(1.0 / drop_path_keep_prob, net.dtype)
      net = net * keep_prob_inv * binary_tensor
    return net

  def _factorized_reduction(self, net, output_filters, stride):
    assert output_filters % 2 == 0
    if stride == 1:
      net = slim.conv2d(net, output_filters, 1, scope='path_conv')
      net = slim.batch_norm(net, scope='path_bn')
      return net
    stride_spec = [1, stride, stride, 1]
    
    # Skip path 1
    # 图像数据取一半数据
    path1 = tf.nn.avg_pool(net, [1, 1, 1, 1], stride_spec, 'VALID')
    path1 = slim.conv2d(path1, int(output_filters / 2), 1, scope='path1_conv')

    # Skip path 2
    # 图像数据取剩下的一半数据
    pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
    path2 = tf.pad(net, pad_arr)[:, 1:, 1:, :]
    path2 = tf.nn.avg_pool(path2, [1, 1, 1, 1], stride_spec, 'VALID')
    path2 = slim.conv2d(path2, int(output_filters / 2), 1, scope='path2_conv')

    # Concat and apply BN
    # 合并数据做归一化
    final_path = tf.concat(values=[path1, path2], axis=3)
    final_path = slim.batch_norm(final_path, scope='final_path_bn')
    return final_path

# PNAS 单元
class PNASCell(NASBaseCell):
  """PNASNet Cell."""

  def __init__(self, num_conv_filters, drop_path_keep_prob, total_num_cells,
               total_training_steps):
    # Configuration for the PNASNet-5 model.
    operations = [
        'separable_5x5_2', 'max_pool_3x3', 'separable_7x7_2', 'max_pool_3x3',
        'separable_5x5_2', 'separable_3x3_2', 'separable_3x3_2', 'max_pool_3x3',
        'separable_3x3_2', 'none'
    ]
    used_hiddenstates = [1, 1, 0, 0, 0, 0, 0]
    hiddenstate_indices = [1, 1, 0, 0, 0, 0, 4, 0, 1, 0]

    super(PNASCell, self).__init__(
        num_conv_filters, operations, used_hiddenstates, hiddenstate_indices,
        drop_path_keep_prob, total_num_cells, total_training_steps)


# PNSNet -5 num_cells=12; num_reduction_layers=2; return [4, 8]
def calc_reduction_layers(num_cells, num_reduction_layers):
  """Figure out what layers should have reductions."""
  reduction_layers = []
  for pool_num in range(1, num_reduction_layers + 1):
    layer_num = (float(pool_num) / (num_reduction_layers + 1)) * num_cells
    layer_num = int(layer_num)
    reduction_layers.append(layer_num)
  return reduction_layers

# 定义常用参数，包括L2正则和归一化参数
def pnasnet_large_arg_scope(weight_decay=4e-5,
                            batch_norm_decay=0.9997,
                            batch_norm_epsilon=1e-3):
  batch_norm_params = {
      'decay': batch_norm_decay, # decay for the moving averages
      'epsilon': batch_norm_epsilon, # epsilon to prevent 0s in variance
      'scale': True,
      'fused': True,
  }
  weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  weights_initializer = tf.contrib.layers.variance_scaling_initializer(
      mode='FAN_OUT')
  with arg_scope([slim.fully_connected, slim.conv2d, slim.separable_conv2d],
                 weights_regularizer=weights_regularizer,
                 weights_initializer=weights_initializer):
    with arg_scope([slim.fully_connected],
                   activation_fn=None, scope='FC'):
      with arg_scope([slim.conv2d, slim.separable_conv2d],
                     activation_fn=None, biases_initializer=None):
        with arg_scope([slim.batch_norm], **batch_norm_params) as sc:
          return sc

# 输入 inputs: (?, 28, 28, 1)
def _imagenet_stem(inputs, hparams, stem_cell):
  """Stem used for models trained on ImageNet."""

  num_stem_filters = int(32 * hparams.stem_multiplier)
  net = slim.conv2d(inputs, num_stem_filters, [3, 3], stride=2, scope='conv0', padding='VALID')
  net = slim.batch_norm(net, scope='conv0_bn')
  # net (?, 13, 13, 96)

  # Run the reduction cells, 4
  cell_outputs = [None, net]
  filter_scaling = 1.0 / (hparams.filter_scaling_rate**hparams.num_stem_cells)
  # filter_scaling : 0.25

  # 将模型转为了2份
  # cell_num：[0, 1]
  for cell_num in range(hparams.num_stem_cells):
    print("stem_cell", net, cell_num, filter_scaling, cell_outputs[-2] )
    net = stem_cell(
        net,
        scope='cell_stem_{}'.format(cell_num),
        filter_scaling=filter_scaling,
        stride=2,
        prev_layer=cell_outputs[-2],
        cell_num=cell_num)
    cell_outputs.append(net)
    filter_scaling *= hparams.filter_scaling_rate
  # cell_outputs: [None, (?, 13, 13, 96), (?, 7, 7, 270), (?, 4, 4,540)]
  return net, cell_outputs

# 辅助
def _build_aux_head(net, end_points, num_classes, hparams, scope):
  """Auxiliary head used for all models across all datasets."""
  with tf.variable_scope(scope):
    aux_logits = tf.identity(net)
    with tf.variable_scope('aux_logits'):
      aux_logits = slim.avg_pool2d(
          aux_logits, [5, 5], stride=3, padding='VALID')
      aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='proj')
      aux_logits = slim.batch_norm(aux_logits, scope='aux_bn0')
      aux_logits = tf.nn.relu(aux_logits)
      # Shape of feature map before the final layer.
      shape = aux_logits.shape
      shape = shape[1:3]
      aux_logits = slim.conv2d(aux_logits, 768, shape, padding='VALID')
      aux_logits = slim.batch_norm(aux_logits, scope='aux_bn1')
      aux_logits = tf.nn.relu(aux_logits)
      aux_logits = tf.contrib.layers.flatten(aux_logits)
      aux_logits = slim.fully_connected(aux_logits, num_classes)
      end_points['AuxLogits'] = aux_logits

# 模型参数
def large_imagenet_config():
  """Large ImageNet configuration based on PNASNet-5."""
  return tf.contrib.training.HParams(
      stem_multiplier=3.0,
      dense_dropout_keep_prob=0.5,
      num_cells=12,
      filter_scaling_rate=2.0,
      num_conv_filters=216,
      drop_path_keep_prob=0.6,
      use_aux_head=1,
      num_reduction_layers=2,
      total_training_steps=250000,
      num_stem_cells=2,
  )

# 构造模型
def build_pnasnet_large(images,
                        num_classes,
                        is_training=True,
                        final_endpoint=None,
                        config=None):
  """Build PNASNet Large model for the ImageNet Dataset."""
  hparams = large_imagenet_config()

  # 如果是运行设置保留Drop数据为100%
  if not is_training:
    hparams.set_hparam('drop_path_keep_prob', 1.0)

  total_num_cells = hparams.num_cells + hparams.num_stem_cells

  # 216 层， 0.5 drop, cells 14, 训练250000步 
  cell = PNASCell(hparams.num_conv_filters, hparams.drop_path_keep_prob,
                  total_num_cells, hparams.total_training_steps)

  with arg_scope([slim.dropout, slim.batch_norm], is_training=is_training):
    end_points = {}
    def add_and_check_endpoint(endpoint_name, net):
      end_points[endpoint_name] = net
      return final_endpoint and (endpoint_name == final_endpoint)

    # Find where to place the reduction cells or stride normal cells
    # 返回 [4, 8]
    reduction_indices = calc_reduction_layers(
        hparams.num_cells, hparams.num_reduction_layers)

    net, cell_outputs = _imagenet_stem(images, hparams, cell)
    # net ： (?, 4, 4, 540)
    # cell_outputs： [None, (?, 13, 13, 96), (?, 7, 7, 270), (?, 4, 4,540)]

    # 主干
    if add_and_check_endpoint('Stem', net):
      return net, end_points

    # 继续设置附加
    # Setup for building in the auxiliary head.
    aux_head_cell_idxes = []
    if len(reduction_indices) >= 2:
      aux_head_cell_idxes.append(reduction_indices[1] - 1)
    # aux_head_cell_idxes: [7]

    # Run the cells
    # 继续做12次单元
    filter_scaling = 1.0
    for cell_num in range(hparams.num_cells):
      # 按 8-1 步时，进行一次缩放采样，同时增加层数 
      is_reduction = cell_num in reduction_indices
      stride = 2 if is_reduction else 1
      if is_reduction: filter_scaling *= hparams.filter_scaling_rate
      net = cell(
          net,
          scope='cell_{}'.format(cell_num),
          filter_scaling=filter_scaling,
          stride=stride,
          prev_layer=cell_outputs[-2],
          cell_num=hparams.num_stem_cells + cell_num)
      if add_and_check_endpoint('Cell_{}'.format(cell_num), net):
        return net, end_points
      cell_outputs.append(net)
      # 0, cell_outputs: [None, (?, 13, 13, 96), (?, 7, 7, 270), (?, 4, 4, 540), (?, 4, 4, 1080)]
      # 1,2,3 cell_outputs: cell_outputs + (?, 4, 4, 1080)
      # 4,5,6,7 cell_outputs: cell_outputs + (?, 2, 2, 2160)
      # 8,9,10,11 cell_outputs: cell_outputs + (?, 1, 1, 4320)

      # 在第 7 层中间插入 aux_net
      if (hparams.use_aux_head and cell_num in aux_head_cell_idxes and
          num_classes and is_training):
        aux_net = tf.nn.relu(net)
        _build_aux_head(aux_net, end_points, num_classes, hparams,
                        scope='aux_{}'.format(cell_num))

    # Final softmax layer
    with tf.variable_scope('final_layer'):
      net = tf.nn.relu(net)
      net = tf.reduce_mean(net, [1, 2])
      if add_and_check_endpoint('global_pool', net) or not num_classes:
        return net, end_points

      net = slim.dropout(net, hparams.dense_dropout_keep_prob, scope='dropout')
      logits = slim.fully_connected(net, num_classes)
      if add_and_check_endpoint('Logits', logits):
        return net, end_points

      predictions = tf.nn.softmax(logits, name='predictions')
      if add_and_check_endpoint('Predictions', predictions):
        return net, end_points

    return logits, end_points


# 训练
def main():
  # 输入图像
  images = tf.placeholder(tf.float32, (None, 28, 28, 1))
  with slim.arg_scope(pnasnet_large_arg_scope()):
    logits, _ = build_pnasnet_large(images, num_classes=1001, is_training=False)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  ckpt_restorer = tf.train.Saver()
  ckpt_restorer.restore(sess, 'data/model.ckpt')

  c1, c5 = 0, 0
  val_dataset = datasets.ImageFolder(args.valdir)
  for i, (image, label) in enumerate(val_dataset):
    logits_val = sess.run(logits, feed_dict={image_ph: image})
    top5 = logits_val.squeeze().argsort()[::-1][:5]
    top1 = top5[0]
    if label + 1 == top1:
      c1 += 1
    if label + 1 in top5:
      c5 += 1
    print('Test: [{0}/{1}]\t'
          'Prec@1 {2:.3f}\t'
          'Prec@5 {3:.3f}\t'.format(
          i + 1, len(val_dataset), c1 / (i + 1.), c5 / (i + 1.)))


if __name__ == '__main__':
  main()