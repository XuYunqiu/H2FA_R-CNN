# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
# Additionally modified by Yunqiu Xu for H2FA R-CNN
# ------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings

warnings.filterwarnings('ignore')

import paddle
import paddle.nn.functional as F
from paddle import amp

from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer, init_parallel_env, set_random_seed, init_fleet_env
from ppdet.slim import build_slim_model

import ppdet.utils.cli as cli
import ppdet.utils.check as check
from ppdet.utils.logger import setup_logger

import ppdet.utils.stats as stats
from ppdet.core.workspace import create
import time
from ppdet.utils import profiler
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

logger = setup_logger('train')


def parse_args():
    parser = cli.ArgsParser()
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")
    parser.add_argument(
        "-r", "--resume", default=None, help="weights path for resume")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--enable_ce",
        type=bool,
        default=False,
        help="If set True, enable continuous evaluation job."
             "This flag is only used for internal test.")
    parser.add_argument(
        "--amp",
        action='store_true',
        default=False,
        help="Enable auto mixed precision training.")
    parser.add_argument(
        "--fleet", action='store_true', default=False, help="Use fleet or not")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/scalar",
        help='VisualDL logging directory for scalar.')
    parser.add_argument(
        '--save_prediction_only',
        action='store_true',
        default=False,
        help='Whether to save the evaluation results only')
    parser.add_argument(
        '--profiler_options',
        type=str,
        default=None,
        help="The option of profiler, which should be in "
             "format \"key1=value1;key2=value2;key3=value3\"."
             "please see ppdet/utils/profiler.py for detail.")
    parser.add_argument(
        '--save_proposals',
        action='store_true',
        default=False,
        help='Whether to save the train proposals')
    parser.add_argument(
        '--proposals_path',
        type=str,
        default="sniper/proposals.json",
        help='Train proposals directory')

    args = parser.parse_args()
    return args


class CustomTrainer(Trainer):
    def __init__(self, cfg, mode='train'):
        super(CustomTrainer, self).__init__(cfg, mode)

        if self.mode == 'train':
            self.target_dataset = cfg['TargetTrainDataset']
            self.target_loader = create('InfiniteReader')(self.target_dataset, cfg.worker_num)

    def train(self, validate=False):
        assert self.mode == 'train', "Model not in 'train' mode"
        Init_mark = False

        sync_bn = (getattr(self.cfg, 'norm_type', None) == 'sync_bn' and
                   self.cfg.use_gpu and self._nranks > 1)
        if sync_bn:
            self.model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model)

        model = self.model
        if self.cfg.get('fleet', False):
            model = fleet.distributed_model(model)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
        elif self._nranks > 1:
            find_unused_parameters = self.cfg[
                'find_unused_parameters'] if 'find_unused_parameters' in self.cfg else False
            model = paddle.DataParallel(
                self.model, find_unused_parameters=find_unused_parameters)

        # enable auto mixed precision mode
        if self.cfg.get('amp', False):
            scaler = amp.GradScaler(
                enable=self.cfg.use_gpu or self.cfg.use_npu, init_loss_scaling=1024)

        self.status.update({
            'epoch_id': self.start_epoch,
            'step_id': 0,
            'steps_per_epoch': len(self.loader)
        })

        self.status['batch_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['data_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['training_staus'] = stats.TrainingStats(self.cfg.log_iter)

        if self.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg.worker_num)
            self._flops(flops_loader)
        profiler_options = self.cfg.get('profiler_options', None)

        self._compose_callback.on_train_begin(self.status)

        for epoch_id in range(self.start_epoch, self.cfg.epoch):
            self.status['mode'] = 'train'
            self.status['epoch_id'] = epoch_id
            self._compose_callback.on_epoch_begin(self.status)
            self.loader.dataset.set_epoch(epoch_id)
            self.target_loader.dataset.set_epoch(epoch_id)

            batch_idx = 0

            model.train()
            iter_tic = time.time()
            for step_id, data in enumerate(self.loader):
                target_data = next(self.target_loader)

                # TODO: better pre-processing implementation on CPU
                source_h, source_w = data['image'].shape[2:]
                target_h, target_w = target_data['image'].shape[2:]
                max_h = max(source_h, target_h)
                max_w = max(source_w, target_w)
                data['image'] = F.pad(data['image'], [0, max_w - source_w, 0, max_h - source_h])
                target_data['image'] = F.pad(target_data['image'], [0, max_w - target_w, 0, max_h - target_h])

                # merge_data: [source_data, target_data]
                data = {
                    k: data[k] + target_data[k] if type(data[k]) is list
                    else paddle.concat((data[k], target_data[k]))
                    for k in data.keys()
                }

                self.status['data_time'].update(time.time() - iter_tic)
                self.status['step_id'] = step_id
                profiler.add_profiler_step(profiler_options)
                self._compose_callback.on_step_begin(self.status)
                data['epoch_id'] = epoch_id

                if self.cfg.get('amp', False):
                    with amp.auto_cast(enable=self.cfg.use_gpu):
                        # model forward
                        outputs = model(data)
                        loss = outputs['loss']

                        loss /= 2

                    # model backward
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()

                    if (batch_idx + 1) % 2 == 0:
                        # in dygraph mode, optimizer.minimize is equal to optimizer.step
                        scaler.minimize(self.optimizer, scaled_loss)
                else:
                    # model forward
                    outputs = model(data)
                    loss = outputs['loss']
                    # model backward
                    loss.backward()
                    self.optimizer.step()
                curr_lr = self.optimizer.get_lr()
                self.lr.step()
                if self.cfg.get('unstructured_prune'):
                    self.pruner.step()
                if (batch_idx + 1) % 2 == 0:
                    self.optimizer.clear_grad()
                batch_idx += 1
                self.status['learning_rate'] = curr_lr

                if self._nranks < 2 or self._local_rank == 0:
                    self.status['training_staus'].update(outputs)

                self.status['batch_time'].update(time.time() - iter_tic)
                self._compose_callback.on_step_end(self.status)
                if self.use_ema:
                    self.ema.update()
                iter_tic = time.time()

            if self.cfg.get('unstructured_prune'):
                self.pruner.update_params()

            is_snapshot = (self._nranks < 2 or self._local_rank == 0) \
                          and ((epoch_id + 1) % self.cfg.snapshot_epoch == 0 or epoch_id == self.end_epoch - 1)
            if is_snapshot and self.use_ema:
                # apply ema weight on model
                weight = copy.deepcopy(self.model.state_dict())
                self.model.set_dict(self.ema.apply())
                self.status['weight'] = weight

            self._compose_callback.on_epoch_end(self.status)

            if validate and is_snapshot:
                if not hasattr(self, '_eval_loader'):
                    # build evaluation dataset and loader
                    self._eval_dataset = self.cfg.EvalDataset
                    self._eval_batch_sampler = \
                        paddle.io.BatchSampler(
                            self._eval_dataset,
                            batch_size=self.cfg.EvalReader['batch_size'])
                    # If metric is VOC, need to be set collate_batch=False.
                    if self.cfg.metric == 'VOC':
                        self.cfg['EvalReader']['collate_batch'] = False
                    self._eval_loader = create('EvalReader')(
                        self._eval_dataset,
                        self.cfg.worker_num,
                        batch_sampler=self._eval_batch_sampler)
                # if validation in training is enabled, metrics should be re-init
                # Init_mark makes sure this code will only execute once
                if validate and Init_mark == False:
                    Init_mark = True
                    self._init_metrics(validate=validate)
                    self._reset_metrics()

                with paddle.no_grad():
                    self.status['save_best_model'] = True
                    self._eval_with_loader(self._eval_loader)

            if is_snapshot and self.use_ema:
                # reset original weight
                self.model.set_dict(weight)
                self.status.pop('weight')

        self._compose_callback.on_train_end(self.status)


def run(FLAGS, cfg):
    # init fleet environment
    if cfg.fleet:
        init_fleet_env(cfg.get('find_unused_parameters', False))
    else:
        # init parallel environment if nranks > 1
        init_parallel_env()

    if FLAGS.enable_ce:
        set_random_seed(0)

    # build trainer
    trainer = CustomTrainer(cfg, mode='train')

    # load weights
    if FLAGS.resume is not None:
        trainer.resume_weights(FLAGS.resume)
    elif 'pretrain_weights' in cfg and cfg.pretrain_weights:
        trainer.load_weights(cfg.pretrain_weights)

    # training
    trainer.train(FLAGS.eval)


def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    cfg['amp'] = FLAGS.amp
    cfg['fleet'] = FLAGS.fleet
    cfg['use_vdl'] = FLAGS.use_vdl
    cfg['vdl_log_dir'] = FLAGS.vdl_log_dir
    cfg['save_prediction_only'] = FLAGS.save_prediction_only
    cfg['profiler_options'] = FLAGS.profiler_options
    cfg['save_proposals'] = FLAGS.save_proposals
    cfg['proposals_path'] = FLAGS.proposals_path
    merge_config(FLAGS.opt)

    # disable npu in config by default
    if 'use_npu' not in cfg:
        cfg.use_npu = False

    # disable xpu in config by default
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False

    if cfg.use_gpu:
        place = paddle.set_device('gpu')
    elif cfg.use_npu:
        place = paddle.set_device('npu')
    elif cfg.use_xpu:
        place = paddle.set_device('xpu')
    else:
        place = paddle.set_device('cpu')

    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn' and not cfg.use_gpu:
        cfg['norm_type'] = 'bn'

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config)

    # FIXME: Temporarily solve the priority problem of FLAGS.opt
    merge_config(FLAGS.opt)
    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_npu(cfg.use_npu)
    check.check_version()

    run(FLAGS, cfg)


if __name__ == "__main__":
    main()
