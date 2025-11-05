import torch
import torch.nn as nn
import torch.nn.init as init
import random
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class CPMLEDModel(BaseModel):
    """Base SR model for joint low-light image enhancement and deblurring."""

    def __init__(self, opt):
        super(CPMLEDModel, self).__init__(opt)
        # Set random seed and deterministic
        # define network
        self.net_g = build_network(opt['network_g'])
        self.init_weights = self.opt['train'].get('init_weights', False)
        if self.init_weights:
            self.initialize_weights(self.net_g, 0.1)
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def initialize_weights(self, net_l, scale=0.1):
        if not isinstance(net_l, list):
            net_l = [net_l]
        for net in net_l:
            for n, m in net.named_modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale  # for residual block
                    if m.bias is not None:
                        m.bias.data.zero_()


    def init_training_settings(self):
        """初始化训练配置"""
        self.net_g.train()# 设置为训练模式
        train_opt = self.opt['train']
        # ===== 1. EMA配置 =====
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)# 创建EMA影子网络
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)# 加载预训练到EMA网络
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # 从当前网络复制权重
            self.net_g_ema.eval()  # EMA网络始终为eval模式

        # ===== 2. 损失函数配置 =====
        # 像素级损失（如L1/L2）
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):# 感知损失（如VGG-based）
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:# 必须至少有一种损失
            raise ValueError('Both pixel and perceptual losses are None.')

        if train_opt.get('edge_opt'):
            self.cri_edge = build_loss(train_opt['edge_opt']).to(self.device)
        else:
            self.cri_edge = None

        # ===== 3. 侧边输出配置 =====
        self.use_side_loss = train_opt.get('use_side_loss', True)
        self.side_loss_weight = train_opt.get('side_loss_weight', 0.8)

        # set up optimizers and schedulers# ===== 4. 优化器和调度器 =====
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        """配置优化器（仅网络G）"""
        train_opt = self.opt['train']
        optim_params = []
        # 收集所有需要梯度的参数
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # 创建优化器
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        """将数据移动到设备"""
        self.lq = data['lq'].to(self.device) # low-blurred image# 低质量/模糊输入
        self.gt = data['gt'].to(self.device) # ground truth# 真实高清图像

    def optimize_parameters(self, current_iter):
        """执行单次优化迭代"""
        self.optimizer_g.zero_grad()
        # ===== 1. 前向传播 =====
        # 使用侧边输出（根据参数use_side_loss=True）
        self.output, self.side_output = self.net_g(self.lq, side_loss=self.use_side_loss)
        if self.use_side_loss:# 为侧边输出准备下采样GT
            h,w = self.side_output.shape[2:]
            self.side_gt = torch.nn.functional.interpolate(self.gt, (h, w), mode='bicubic', align_corners=False)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            if self.use_side_loss:# 侧边输出像素损失（权重0.8）
                l_side_pix = self.cri_pix(self.side_output, self.side_gt) * self.side_loss_weight
                l_total += l_side_pix
                loss_dict['l_side_pix'] = l_side_pix

        # perceptual loss# 感知损失（主输出）
        if self.cri_perceptual:
            l_percep, _ = self.cri_perceptual(self.output, self.gt)
            l_total += l_percep
            loss_dict['l_percep'] = l_percep
            if self.use_side_loss:# 侧边输出感知损失（权重0.8）
                l_side_percep, _ = self.cri_perceptual(self.side_output, self.side_gt)
                l_side_percep = l_side_percep * self.side_loss_weight
                l_total += l_side_percep
                loss_dict['l_side_percep'] = l_side_percep

        # edge loss# 边缘损失
        if self.cri_edge:
            l_edge =self.cri_edge(self.output, self.gt)
            l_total += l_edge
            loss_dict['l_edge'] = l_edge
            if self.use_side_loss:
                l_side_edge = self.cri_edge(self.side_output,self.side_gt)
                l_side_edge = l_side_edge * self.side_loss_weight
                l_total += l_side_edge
                loss_dict['l_side_edge'] = l_side_edge

        # ===== 3. 反向传播和优化 =====
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)# 记录损失

        if self.ema_decay > 0:# EMA更新（如果启用）
            self.model_ema(decay=self.ema_decay)

    def test(self):
        """测试模式（使用EMA网络如果启用）"""
        # 如果启用了EMA（指数移动平均）
        if self.ema_decay > 0:
            self.net_g_ema.eval()  # 设置EMA模型为评估模式
            with torch.no_grad():  # 禁用梯度计算
                self.output = self.net_g_ema(self.lq)[0]  # 使用EMA模型推理
        else:
            self.net_g.eval()  # 设置原始模型为评估模式
            with torch.no_grad():  # 禁用梯度计算
                self.output = self.net_g(self.lq)[0]  # 使用原始模型推理
            self.net_g.train()  # 恢复训练模式

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """分布式验证（多GPU/多节点环境）"""
        # 只在主节点（rank 0）上执行验证
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """非分布式验证（单GPU环境）"""
        # 获取数据集名称
        dataset_name = dataloader.dataset.opt['name']

        # 检查是否启用了评估指标
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            # 初始化指标结果字典
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        # 创建进度条
        pbar = tqdm(total=len(dataloader), unit='image')

        # 遍历验证集
        for idx, val_data in enumerate(dataloader):
            # 获取图像名称（不含扩展名）
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            # 数据加载和推理
            self.feed_data(val_data)  # 加载数据到设备
            self.test()  # 执行推理

            # 获取可视化结果
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])  # 将张量转换为图像

            # 如果有真实图像（GT）
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])  # 转换GT为图像
                del self.gt  # 释放GT内存

            # 释放显存
            del self.lq
            del self.output
            torch.cuda.empty_cache()  # 清空CUDA缓存

            # 保存结果图像
            if save_img:
                if self.opt['is_train']:  # 训练时保存
                    save_img_path = osp.join(
                        self.opt['path']['visualization'],
                        img_name,
                        f'{img_name}_{current_iter}.png'
                    )
                else:  # 测试时保存
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'],
                            dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png'
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'],
                            dataset_name,
                            f'{img_name}_{self.opt["name"]}.png'
                        )
                imwrite(sr_img, save_img_path)  # 写入图像文件

            # 计算评估指标
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            # 更新进度条
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        # 关闭进度条
        pbar.close()

        # 计算平均指标值
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            # 记录指标结果
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        """记录验证指标到日志和TensorBoard"""
        log_str = f'Validation {dataset_name}\n'
        # 构建指标日志字符串
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'

        # 输出到日志文件
        logger = get_root_logger()
        logger.info(log_str)

        # 记录到TensorBoard
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        """获取当前批次的输入、输出和真实值（用于可视化）"""
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()  # 低质量输入
        out_dict['result'] = self.output.detach().cpu()  # 模型输出

        # 如果有真实图像（GT）
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()  # 真实高清图像

        return out_dict

    def save(self, epoch, current_iter):
        """保存模型和训练状态"""
        # 保存网络权重
        if self.ema_decay > 0:
            # 同时保存原始模型和EMA模型
            self.save_network(
                [self.net_g, self.net_g_ema],
                'net_g',
                current_iter,
                param_key=['params', 'params_ema']
            )
        else:
            # 只保存原始模型
            self.save_network(self.net_g, 'net_g', current_iter)

        # 保存训练状态（优化器、学习率调度器等）
        self.save_training_state(epoch, current_iter)


