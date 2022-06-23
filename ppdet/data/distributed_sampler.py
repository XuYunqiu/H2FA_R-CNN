# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates.
# ------------------------------------------------------------------------
# Additionally modified by Yunqiu Xu for H2FA R-CNN
# ------------------------------------------------------------------------

import itertools
import math
from collections import defaultdict
import numpy as np

from paddle.io import Sampler


class InfiniteRandomSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=False, drop_last=False):
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, \
            "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
            "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
            "drop_last should be a boolean number"

        from paddle.fluid.dygraph.parallel import ParallelEnv

        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, \
                    "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, \
                    "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank

        self.drop_last = drop_last
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.nranks))
        self.total_size = self.num_samples * self.nranks
        self._size = self.num_samples

    def __iter__(self):
        start = self.local_rank * self.batch_size
        _sample_iter = itertools.islice(
            self._infinite_indices(), start*self.batch_size, None, self.nranks*self.batch_size
        )
        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []

    def _infinite_indices(self):
        while True:
            num_samples = len(self.dataset)
            indices = np.arange(num_samples).tolist()
            if self.shuffle:
                np.random.shuffle(indices)
            yield from indices
