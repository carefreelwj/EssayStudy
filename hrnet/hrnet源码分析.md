![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029151116718.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5ODYyMjIz,size_16,color_FFFFFF,t_70#pic_center)


论文链接：[https://arxiv.org/abs/1902.09212](https://arxiv.org/abs/1902.09212)

代码链接：[https://github.com/leoxiaobin/deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

论文源码分析：

## 1 源码准备

在指定文件夹下，输入命令：

```bash
git clone https://github.com/leoxiaobin/deep-high-resolution-net.pytorch.git
```

下载完成后，得到HRNet源码

## 2 源码结构

下表列出HRNet中比较重要的文件：

|         文件名称          |           功能            |
| :-----------------------: | :-----------------------: |
|      tools/trian.py       |         训练脚本          |
|       tools/test.py       |         测试脚本          |
|    lib/dataset/mpii.py    |  对MPII数据集进行预处理   |
| lib/dataset/JointsDataSet |       数据读取脚本        |
| lib/models/pose_hrnet.py  |     网络结构构建脚本      |
|         lib/utils         |      HRNet的一些方法      |
|  experiments/mpii/hrnet   | HRNet网络的初始化参数脚本 |

接下来对一些重要文件，将一一讲解，并且说清数据流的走向和函数调用关系。

## 3 源码分析（准备阶段）

### 3.1 数据准备

#### 3.1.1 mpii.py

通过阅读源码可以知道，通过mpii.py文件中的MPIIDataset的初始化函数，将获得一个rec的数据，其中包含：coco中所有人体，对应关键点的信息、图片路径、标准化以及缩放比例等信息。

**(1) _init_函数**

```python
class MPIIDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 16
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))
```

MPIIDataSet类的初始化方法_init_需要如下参数：

- num_joints : MPII数据集中人体关键点标记个数
- flip_pairs : 人体水平对称关键映射
- parents_ids : 父母ids
- upper_body_ids : 定义上半身关键点
- lower_body_ids : 定义下半身关键点
- db : 读取目标检测模型

(2) **_get_db函数**

```python
def _get_db(self):
        # create train/val split
        file_name = os.path.join(
            self.root, 'annot', self.image_set+'.json'
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d, 
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db
```

首先找到MPII数据集的分割依据文件annotaion，之后循环遍历该数据集，读取每张图片的名称、中心点位置、大小、人体关键节点位置（用三维坐标表示）、可见的人体关键节点位置并保存，形成一个字典不断加入到gt_db，循环结束返回。数据预处理到这并没有结束，因为还需要进一步处理，原因在于当计算loss的时候，我们需要的是热图(heatmap)。

#### 3.1.2 JointsDataset.py

接下来，我们需要根据get_db中的信息，读取图片像素(用于训练)，同时把标签信息转化为heatmap。

**(1) _init_.py**

```python
class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0# 人体关节的数目
        self.pixel_std = 200# 像素标准化参数
        self.flip_pairs = []# 水平翻转
        self.parent_ids = []# 父母ID==

        self.is_train = is_train# 是否进行训练
        self.root = root# 训练数据根目录
        self.image_set = image_set# 图片数据集名称,如‘train2017’

        self.output_path = cfg.OUTPUT_DIR# 输出目录
        self.data_format = cfg.DATASET.DATA_FORMAT# 数据格式如‘jpg’

        self.scale_factor = cfg.DATASET.SCALE_FACTOR# 缩放因子
        self.rotation_factor = cfg.DATASET.ROT_FACTOR # 旋转角度
        self.flip = cfg.DATASET.FLIP# 是否进行水平翻转
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY# 人体一半关键点的数目，默认为8
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY# 人体一半的概率
        self.color_rgb = cfg.DATASET.COLOR_RGB# 图片格式，默认为rgb

        self.target_type = cfg.MODEL.TARGET_TYPE# 目标数据的类型，默认为高斯分布
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)# 网络训练图片大小,如[192,256]
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)# 标签热图的大小
        self.sigma = cfg.MODEL.SIGMA# sigma参数，默认为2
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT# 是否对每个关节使用不同的权重,默认为false
        self.joints_weight = 1# 关节权重

        self.transform = transform# 数据增强,转换等
        self.db = []# 用于保存训练数据的信息,由子类提供
```

_init_函数的功能在于初始化JointsDataset模型，设置一些参数和参数默认值，每个参数值的作用已经注释。通过这些初始化操作，可以获得一些基本信息，如人体关节数目、图片格式、标签热图的大小、关节权重等。

**(2) _getitem_函数**

```python
  def __getitem_(self,idx): 
        db_rec = copy.deepcopy(self.db[idx])
        image_file = db_rec['image']
        filename = db_rec['filename'] if 'fename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))
  
        joints = db_rec['joints_3d']# 人体3d关键点的所有坐标
        joints_vis = db_rec['joints_3d_vis']# 人体3d关键点的所有可视坐标

        # 获取训练样本转化之后的center以及scale,
        c = db_rec['center']
        s = db_rec['scale']
        
        # 如果训练样本中没有设置score,则加载该属性,并且设置为1
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body
                  
            sf = self.scale_factor
            rf = self.rotation_factor
          
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0
                
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)

        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
  
        if self.transform:
            input = self.transform(input)
  
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
  
        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_weight, meta
```

- 首先根据idx从db获取样本信息，包括图片路径和图片序号等，如果数据格式为zip则解压，否则直接读取图像，获得像素值；再次读取db，获取人体关键点坐标、训练样本转化之后的center以及scale
- 之后如果是进行训练，则判断可见关键点是否大于人体一半关键点，并且生成的随机数小于self.prob_half_body=0.3，如果是，则需要重新调整center和scale；再设置缩放因子和旋转因子大小，对数据进行数据增强操作，包括缩放、旋转、翻转等。
- 因为进行反射变换，样本数据关键点发生角度旋转之后,每个像素也旋转到对应位置，所以人体的关键点也要进行反射变化
- 最终通过self.generate_target(joints, joints_vis)函数获得target，target_weight，shape为target[17,64,48], target_weight[17,1]。

将之前得到的所有信息按照字典的形式，存储于meta中，并且返回input、target、 target_weight, meta

**(3) generate_target函数**

```python
def generate_target(self, joints, joints_vis):
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]
        
        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'
            
        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)
            tmp_size = self.sigma * 3

            # 为每个关键点生成热图target以及对应的热图权重target_weight
            for joint_id in range(self.num_joints):
                # 先计算出原图到输出热图的缩小倍数
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds 根据tmp_size参数,计算出关键点范围左上角和右下角坐标
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                # 判断该关键点是否处于热图之外,如果处于热图之外,则把该热图对应的target_weight设置为0,然后continue
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian 产生高斯分布的大小
                size = 2 * tmp_size + 1
                # x[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12.]
                x = np.arange(0, size, 1, np.float32)
                # y[[ 0.][ 1.][ 2.][ 3.][ 4.][ 5.][ 6.][ 7.][ 8.][ 9.][10.][11.][12.]]
                y = x[:, np.newaxis]
                # x0 = y0 = 6
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1 g形状[13,13], 该数组中间的[7,7]=1,离开该中心点越远数值越小
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range 判断边界,获得有效高斯分布的范围
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                
                # Image range 判断边界,获得有有效的图片像素边界
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])
    
                # 如果该关键点对应的target_weight>0.5(即表示该关键点可见),则把关键点附近的特征点赋值成gaussian
                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        # 如果各个关键点训练权重不一样
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight
```

该函数的功能主要在于产生热图，并且制作热图的方式必须为gaussion。会为每个关键点生成热图target以及对应的热图权重target_weight，在生成期间还需判断该关键点是否处于热图之外,如果处于热图之外,则把该热图对应的target_weight设置为0,然后continue。最终生成高斯分布的热图表示，返回target和target_weight。

### 3.2 模型设计

#### 3.2.1 基本模块

如下的左图对应于resnet-18/34使用的基本块，右图是50/101/152所使用的，由于他们都比较深，所以有图相比于左图使用了1x1卷积来降维。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029151244399.png#pic_center)

基本模块主要是BasicBlock、Bottleneck，现在进行逐个分析：

1. BasicBlock ： 搭建上图左边的模块。

- 每个卷积块后面连接BN层进行归一化；

- 残差连接前的3x3卷积之后只接入BN，不使用ReLU，避免加和之后的特征皆为正，保持特征的多样；

- 跳层连接：两种情况，当模块输入和残差支路（3x3->3x3）的通道数一致时，直接相加；当两者通道不一致时（一般发生在分辨率降低之后，同分辨率一般通道数一致），需要对模块输入特征使用1x1卷积进行升/降维（步长为2，上面说了分辨率会降低），之后同样接BN，不用ReLU。

  ```python
  # 提前写好一个类，在HighResolutionModule中使用
  # x: [batch_size, 256, 8, 8]
  # out: [batch_size, 256, 8, 8]
  class BasicBlock(nn.Module):
      expansion = 1
  
      def __init__(self, inplanes, planes, stride=1, downsample=None):
          super(BasicBlock, self).__init__()
          self.conv1 = conv3x3(inplanes, planes, stride)
          self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
          self.relu = nn.ReLU(inplace=True)
          self.conv2 = conv3x3(planes, planes)
          self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
          self.downsample = downsample
          self.stride = stride
  
      def forward(self, x):
          residual = x
  
          out = self.conv1(x)
          out = self.bn1(out)
          out = self.relu(out)
  
          out = self.conv2(out)
          out = self.bn2(out)
  
          if self.downsample is not None:
              residual = self.downsample(x)
  
          out += residual
          out = self.relu(out)
  
          return out
  ```

  

2. Bottleneck ：搭建上图右边的模块。

- 使用1x1卷积先降维，再使用3x3卷积进行特征提取，最后再使用1x1卷积把维度升回去；

- 每个卷积块后面连接BN层进行归一化

- 残差连接前的1x1卷积之后只接入BN，不使用ReLU，避免加和之后的特征皆为正，保持特征的多样性。

- 跳层连接：两种情况，当模块输入和残差支路（1x1->3x3->1x1）的通道数一致时，直接相加；当两者通道不一致时（一般发生在分辨率降低之后，同分辨率一般通道数一致），需要对模块输入特征使用1x1卷积进行升/降维（步长为2，上面说了分辨率会降低），之后同样接BN，不用ReLU。

  ```python
  # 在layer1中使用4个Bottleneck，验证论文中以高分辨率子网为第一阶段，维持高分辨率表示
  # x: [batch_size, 256, 64, 64]
  # output : [batch_size, 256, 64, 64]
  class Bottleneck(nn.Module):
      expansion = 4
  
      def __init__(self, inplanes, planes, stride=1, downsample=None):
          super(Bottleneck, self).__init__()
          self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
          self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
          self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                 padding=1, bias=False)
          self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
          self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                                 bias=False)
          self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                    momentum=BN_MOMENTUM)
          self.relu = nn.ReLU(inplace=True)
          self.downsample = downsample
          self.stride = stride
  
      def forward(self, x):
          residual = x
  
          out = self.conv1(x)
          out = self.bn1(out)
          out = self.relu(out)
  
          out = self.conv2(out)
          out = self.bn2(out)
          out = self.relu(out)
  
          out = self.conv3(out)
          out = self.bn3(out)
      
          if self.downsample is not None:
              residual = self.downsample(x)
  
          out += residual
          out = self.relu(out)
  
          return out
  ```




#### 3.2.2 高分辨率模块-HighResolutionModule

当仅包含一个分支时，生成该分支，没有融合模块，直接返回；当包含不仅一个分支时，先将对应分支的输入特征输入到对应分支，得到对应分支的输出特征；紧接着执行融合模块。

1. ***_check_branches函数***

   判断`num_branches (int)` 和 `num_blocks, num_inchannels, num_channels (list)` 三者的长度是否一致，否则报错；

   ```python
    # 判断三个参数长度是否一致，否则报错
       def _check_branches(self, num_branches, blocks, num_blocks,
                           num_inchannels, num_channels):
           if num_branches != len(num_blocks):
               error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                   num_branches, len(num_blocks))
               logger.error(error_msg)
               raise ValueError(error_msg)
   
           if num_branches != len(num_channels):
               error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                   num_branches, len(num_channels))
               logger.error(error_msg)
               raise ValueError(error_msg)
   
           if num_branches != len(num_inchannels):
               error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                   num_branches, len(num_inchannels))
               logger.error(error_msg)
               raise ValueError(error_msg)
   ```

   

2.  ***_make_one_branch函数***

搭建一个分析，单个分支内部分辨率相等，一个分支由num_branches[branch_index]个block组成，block可以是两种ResNet模块中的一种；

- 首先判断是否降维或者输入输出的通道(`num_inchannels[branch_index]和 num_channels[branch_index] * block.expansion(通道扩张率)`)是否一致，不一致使用1z1卷积进行维度升/降，后接BN，不使用ReLU；
- 顺序搭建`num_blocks[branch_index]`个block，第一个block需要考虑是否降维的情况，所以单独拿出来，后面`1 到 num_blocks[branch_index]`个block完全一致，使用循环搭建就行。此时注意在执行完第一个block后将`num_inchannels[branch_index`重新赋值为` num_channels[branch_index] * block.expansion`。

```python
 def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )
        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )
        return nn.Sequential(*layers)
```

3. ***_make_branches函数***

循环调用`_make_one_branch`函数创建多个分支；

```python
def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)
```

4. ***_make_fuse_layers函数***

(1) 如果分支数等于1，返回none，说明此时不需要使用融合模块

```python
 if self.num_branches == 1:
     return None
```

(2) 双层循环

```python
for i in range(num_branches if self.multi_scale_output else 1)
```

该语句的作用在于，如果需要产生多分辨率的结果，就双层循环num_branches次，如果只需要产生最高分辨率的表示，就将i确定为0。

- 如果`j > i`，此时的目标是将所有分支上采样到和`i`分支相同的分辨率并融合，也就是说`j`所代表的分支分辨率比`i`分支低，`2**(j-i)`表示`j`分支上采样这么多倍才能和`i`分支分辨率相同。先使用1x1卷积将`j`分支的通道数变得和`i`分支一致，进而跟着BN，然后依据上采样因子将`j`分支分辨率上采样到和`i`分支分辨率相同，此处使用最近邻插值；
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029151341482.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5ODYyMjIz,size_16,color_FFFFFF,t_70#pic_center)

```python
if j > i:
  fuse_layer.append(
    nn.Sequential(
      nn.Conv2d(
        num_inchannels[j],
        num_inchannels[i],
        1, 1, 0, bias=False
      ),
      nn.BatchNorm2d(num_inchannels[i]),
      nn.Upsample(scale_factor=2**(j-i), mode='nearest')
    )
  )
```


- 如果`j = i`，也就是说自身与自身之间不需要融合，nothing to do；

```python
elif j == i:
    fuse_layer.append(None)
```

- 如果`j < i`，转换角色，此时最终目标是将所有分支采样到和`i`分支相同的分辨率并融合，注意，此时`j`所代表的分支分辨率比`i`分支高，正好和(2.1)相反。此时再次内嵌了一个循环，这层循环的作用是当`i-j > 1`时，也就是说两个分支的分辨率差了不止二倍，此时还是两倍两倍往上采样，例如`i-j = 2`时，`j`分支的分辨率比`i`分支大4倍，就需要上采样两次，循环次数就是2；

**i.**  当`k == i - j - 1`时，举个例子，`i = 2`,`j = 1`, 此时仅循环一次，并采用当前模块，此时直接将`j`分支使用3x3的步长为2的卷积下采样(不使用bias)，后接BN，不使用ReLU；

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029151407745.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5ODYyMjIz,size_16,color_FFFFFF,t_70#pic_center)


```python
for k in range(i-j):
    if k == i - j - 1:
        num_outchannels_conv3x3 = num_inchannels[i]
        conv3x3s.append(
            nn.Sequential(
                nn.Conv2d(
                    num_inchannels[j],
                    num_outchannels_conv3x3,
                    3, 2, 1, bias=False
                ),
                nn.BatchNorm2d(num_outchannels_conv3x3)
            )
        )
```


**ii.**  当`k != i - j - 1`时，举个例子，`i = 3`,`j = 1`, 此时循环两次，先采用当前模块，将`j`分支使用3x3的步长为2的卷积下采样(不使用bias)两倍，后接BN和ReLU，紧跟着再使用(2.3.1)中的模块，这是为了保证最后一次二倍下采样的卷积操作不使用ReLU，猜测也是为了保证融合后特征的多样性；

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029151640401.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5ODYyMjIz,size_16,color_FFFFFF,t_70#pic_center)


```python
else:
  num_outchannels_conv3x3 = num_inchannels[j]
  conv3x3s.append(
    nn.Sequential(
      nn.Conv2d(
        num_inchannels[j],
        num_outchannels_conv3x3,
        3, 2, 1, bias=False
      ),
      nn.BatchNorm2d(num_outchannels_conv3x3),
      nn.ReLU(True)
    )
  )
```

#### 3.2.3 整合模块-PoseHighResolutionNet

1. **stage1**

   进行一系列的卷积操作，获得最初的特征图N11

   ```python
      self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                                  bias=False)
           self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
           self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                                  bias=False)
           self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
           self.relu = nn.ReLU(inplace=True)
           self.layer1 = self._make_layer(Bottleneck, 64, 4)
   ```

2. **stage2**

- 首先根据原先设定，获得相关配置信息。对于第二阶段，num_channels=[32,64]，num_channels表示输出通道，最后的64是新建平行分支N2的输出通道数；block为Bottleneck，在论文中提到，第一个stage到第二个stage变换时，使用Bottleneck.
- 之后会生成新的平行N2分支网络，即N11 --> N21,N22这个过程，同时如果输入输出通道不一致时。会对输入的特征图x进行通道变换.
- 最后对平行子网进行加工，让其输出的y，可以当做下一个stage的输入x，这里的pre_stage_channels为当前阶段的输出通道数，也就是一个stage的输入通道数，同时平行子网信息交换模块，也包含其中

```python
    self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        
        self.transition1 = self._make_transition_layer([256], num_channels)
      
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)
```

3. **stage 3**

- 首先根据原先设定，获取stage3的相关配置信息。对于第三阶段，num_channels=[32,64,128],num_channels表示输出通道,最后的128是新建平行分支N3的输出通道数；这里的block为BasicBlock,在论文中有提到,**除了第一个stage到第二个stage变换时使用Bottleneck,其余的都是使用BasicBlock**
- 之后会生成新的平行分支N3网络,即N22-->N32,N33这个过程时，如果输入输出通道不一致时。会对输入的特征图x进行通道变换.
- 最后对平行子网进行加工，让其输出的y，可以当做下一个stage的输入x，这里的pre_stage_channels为当前阶段的输出通道数，也就是一个stage的输入通道数，同时平行子网信息交换模块，也包含其中

```python
    self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
   
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)
```

4. **stage 4**

- 首先根据原先设定，获取stage4的相关配置信息。对于第四阶段，num_channels=[32,64,128,256],num_channels表示输出通道,最后的256是新建平行分支N3的输出通道数；这里的block为BasicBlock
- 之后会生成新的平行分支N3网络,即N33-->N43,N44这个过程时，如果输入输出通道不一致时。会对输入的特征图x进行通道变换.
- 最后对平行子网进行加工，让其输出的y，可以当做下一个stage的输入x，这里的pre_stage_channels为当前阶段的输出通道数，也就是一个stage的输入通道数，同时平行子网信息交换模块，也包含其中

```python
    self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)
```

5. **整合预测**

   对最终的特征图混合之后，进行一次卷积，预测人体关键点的heatmap

   ```python
      self.final_layer = nn.Conv2d(
               in_channels=pre_stage_channels[0],
               out_channels=cfg['MODEL']['NUM_JOINTS'],
               kernel_size=extra['FINAL_CONV_KERNEL'],
               stride=1,
               padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
           )
    
           self.pretrained_layers = extra['PRETRAINED_LAYERS']
   ```

6. **重要函数**

   i. _make_transition_layer

   该函数的作用在于生成下一阶段同等分辨率和一般分辨率的分支。首先会进行循环遍历，对每个分支进行处理

   不是最后一个分支：如果当前一层的输入通道和输出通道不相等，则通过卷积对通道数进行变换；如果当前层的输入=输出通道数，则维持原样；

   是最后一个分支：新建一个分支，并且这个分支分辨率会减少一半

   ```python
   def _make_transition_layer(
               self, num_channels_pre_layer, num_channels_cur_layer):
           num_branches_cur = len(num_channels_cur_layer)
           num_branches_pre = len(num_channels_pre_layer)
   
           transition_layers = []
           for i in range(num_branches_cur):
               if i < num_branches_pre:
                   if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                       transition_layers.append(
                           nn.Sequential(
                               nn.Conv2d(
                                   num_channels_pre_layer[i],
                                   num_channels_cur_layer[i],
                                   3, 1, 1, bias=False
                               ),
                               nn.BatchNorm2d(num_channels_cur_layer[i]),
                               nn.ReLU(inplace=True)
                           )
                       )
                   else:
                       transition_layers.append(None)
               else:
                   conv3x3s = []
                   for j in range(i+1-num_branches_pre):
                       inchannels = num_channels_pre_layer[-1]
                       outchannels = num_channels_cur_layer[i] \
                           if j == i-num_branches_pre else inchannels
                       conv3x3s.append(
                           nn.Sequential(
                               nn.Conv2d(
                                   inchannels, outchannels, 3, 2, 1, bias=False
                               ),
                               nn.BatchNorm2d(outchannels),
                               nn.ReLU(inplace=True)
                           )
                       )
                   transition_layers.append(nn.Sequential(*conv3x3s))
   
           return nn.ModuleList(transition_layers)
   ```

   ii. _make_stage函数

   该函数的作用在于生成stage时的*HighResolutionModule*

   ```python
    def _make_stage(self, layer_config, num_inchannels,
                       multi_scale_output=True):
           """
                   当stage=2时： num_inchannels=[32,64]           multi_scale_output=Ture
                   当stage=3时： num_inchannels=[32,64,128]       multi_scale_output=Ture
                   当stage=4时： num_inchannels=[32,64,128,256]   multi_scale_output=False
           """
           # 当stage=2,3,4时,num_modules分别为：1,4,3
           # 表示HighResolutionModule（平行之网络交换信息模块）模块的数目
           num_modules = layer_config['NUM_MODULES']
   
           # 当stage=2,3,4时,num_branches分别为：2,3,4,表示每个stage平行网络的数目
           num_branches = layer_config['NUM_BRANCHES']
   
           # 当stage=2,3,4时,num_blocks分别为：[4,4], [4,4,4], [4,4,4,4],
           # 表示每个stage blocks(BasicBlock或者BasicBlock)的数目
           num_blocks = layer_config['NUM_BLOCKS']
   
           # 当stage=2,3,4时,num_channels分别为：[32,64],[32,64,128],[32,64,128,256]
           # 在对应stage, 对应每个平行子网络的输出通道数
           num_channels = layer_config['NUM_CHANNELS']
   
           # 当stage=2,3,4时,分别为：BasicBlock,BasicBlock,
           block = blocks_dict[layer_config['BLOCK']]
   
           # 当stage=2,3,4时,都为：SUM,表示特征融合的方式
           fuse_method = layer_config['FUSE_METHOD']
   
           modules = []
           # 根据num_modules的数目创建HighResolutionModule
           for i in range(num_modules):
               # multi_scale_output is only used last module
               # multi_scale_output 只被用再最后一个HighResolutionModule
               if not multi_scale_output and i == num_modules - 1:
                   reset_multi_scale_output = False
               else:
                   reset_multi_scale_output = True
   
               # 根据参数,添加HighResolutionModule到
               modules.append(
                   HighResolutionModule(
                       num_branches,
                       block,
                       num_blocks,
                       num_inchannels,
                       num_channels,
                       fuse_method,
                       reset_multi_scale_output
                   )
               )
               # 获得最后一个HighResolutionModule的输出通道数
               num_inchannels = modules[-1].get_num_inchannels()
   
           return nn.Sequential(*modules), num_inchannels
   ```


#### 3.2.4 forward

1. 第一阶段：经过一系列的卷积, 获得初步特征图,总体过程为x[b,3,256,192]-->x[b,256,64,48]

```python
    x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
```

2. 第二阶段：其中包含了创建分支的过程,即 N11-->N21,N22

   总体过程为:
              x[b,256,64,48] ---> y[b, 32, 64, 48]  因为通道数不一致,通过卷积进行通道数变换
                                         y[b, 64, 32, 24]  通过新建平行分支生成

   ```python
      x_list = []
           for i in range(self.stage2_cfg['NUM_BRANCHES']):
               if self.transition1[i] is not None:
                   x_list.append(self.transition1[i](x))
               else:
                   x_list.append(x)
            y_list = self.stage2(x_list)
   ```

3. 第三阶段：其中包含了创建分支的过程,即 N22-->N32,N33

   总体过程为：

   ​    y[b, 32, 64, 48] ---> x[b, 32,  64, 48]   因为通道数一致,没有做任何操作
   ​        y[b, 64, 32, 24] ---> x[b, 64,  32, 24]   因为通道数一致,没有做任何操作
   ​                                           x[b, 128, 16, 12]   通过新建平行分支生成

   ```python
      x_list = []
           for i in range(self.stage3_cfg['NUM_BRANCHES']):
               if self.transition2[i] is not None:
                   x_list.append(self.transition2[i](y_list[-1]))
               else:
                   x_list.append(y_list[i])
           y_list = self.stage3(x_list)
   ```

4. 第四阶段：其中包含了创建分支的过程,即 N33-->N43,N44

   总体过程为：

   ​    y[b, 32,  64, 48] ---> x[b, 32,  64, 48]  因为通道数一致,没有做任何操作
   ​         y[b, 64,  32, 24] ---> x[b, 64,  32, 24]  因为通道数一致,没有做任何操作
   ​         y[b, 128, 16, 12] ---> x[b, 128, 16, 12]  因为通道数一致,没有做任何操作
   ​                                             x[b, 256, 8,  6 ]  通过新建平行分支生成

   ```python
      x_list = []
           for i in range(self.stage4_cfg['NUM_BRANCHES']):
               if self.transition3[i] is not None:
                   x_list.append(self.transition3[i](y_list[-1]))
               else:
                   x_list.append(y_list[i])
   ```

   之后进行多尺度特征融合：

   ​    x[b, 32,  64, 48]  --->
   ​        x[b, 64,  32, 24]  --->
   ​        x[b, 128, 16, 12] --->
   ​        x[b, 256, 8,  6 ]   --->       y[b, 32,  64, 48]

5. 预测阶段：

   y[b, 32, 64, 48] --> x[b, 16, 64, 48]

   ```python
   x = self.final_layer(y_list[0])
   ```

## 4 源码分析（训练阶段）

解析参数->构建网络模型->加载训练测试数据集迭代器->迭代训练->模型评估保存

```python
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general 指定yaml文件的路径
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
# 暂时没有具体实现
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly 模型的目录
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    # log 输出tensorboard的目录
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    # data 训练数据的目录
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    # premodel 预训练模型的目录
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args() # 对输入的参数进行解析
    update_config(cfg, args) # 根据输入参数对cfg进行更新 

# 创建logger，用于记录训练过程的打印信息
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting 使用GPU的一些相关设置
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

#根据配置文件构建网络
    print('models.'+cfg.MODEL.NAME+'.get_pose_net')
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file 拷贝lib/models/pose_hrnet.py文件到输出目录之中
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    # 用于训练信息的图形化表示
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_model_summary(model, dump_input))

    # 让模型支持多GPU训练
    model = torch.nn.DataParallel(model, device_ids=[0,]).cuda()

    # 用于计算loss
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code 对输入图像数据进行正则化处理
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # 创建训练以及测试数据的迭代器
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=0,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.PIN_MEMORY
    )

    # 模型加载以及优化策略的相关配置
    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    # 循环迭代进行训练
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)


        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()

```

## 5 数据流分析
