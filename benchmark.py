import argparse
from collections import OrderedDict
from importlib import import_module
import pickle

import numpy as np

from time import time

import torch
# this turns on auto tuner which optimizes performance
torch.backends.cudnn.benchmark = True
import torchvision
print('GPU name=', torch.cuda.get_device_name())
print('cuda version=', torch.version.cuda)
print('cudnn version=', torch.backends.cudnn.version())

class pytorch_base:

    def __init__(self, model_name, precision, image_shape, batch_size):
        self.model = getattr(torchvision.models, model_name)().cuda() if precision == 'fp32' \
            else getattr(torchvision.models, model_name)().cuda().half()
        x = torch.rand(batch_size, 3, image_shape[0], image_shape[1]).cuda()
        self.eval_input = x if precision == 'fp32' else x.half()
        self.train_input = x if precision == 'fp32' else x.half()

    def eval(self, num_iterations, num_warmups):
        self.model.eval()
        durations = []
        for i in range(num_iterations + num_warmups):
            torch.cuda.synchronize()
            t1 = time()
            self.model(self.eval_input)
            torch.cuda.synchronize()
            t2 = time()
            if i >= num_warmups:
                durations.append(t2 - t1)
        return durations

    def train(self, num_iterations, num_warmups):
        self.model.train()
        durations = []
        for i in range(num_iterations + num_warmups):
            torch.cuda.synchronize()
            t1 = time()
            self.model.zero_grad()
            out = self.model(self.train_input)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            t2 = time()
            if i >= num_warmups:
                durations.append(t2 - t1)
        return durations

models = [
    'vgg16',
    'resnet152',
    'densenet161',
    'swin_b',
    'vit_b_32',
    'convnext_large',
    'resnext101_64x4d'
]

precisions = [
    'fp32',
    'fp16'
]
batchsizes = [
    16,
    32,
    64
]

class Benchmark():

 
    def benchmark_model(self, mode, framework, model, precision, image_shape=(224, 224), batch_size=16, num_iterations=10, num_warmups=20):
        framework_model = pytorch_base(model, precision, image_shape, batch_size)
        durations = framework_model.eval(num_iterations, num_warmups) if mode == 'eval' else framework_model.train(num_iterations, num_warmups)
        durations = np.array(durations)
        return durations.mean() * 1000

    def benchmark_framework(self,):
        results = OrderedDict()
        for precision in precisions:
            results[precision] = []
            for model in models:
                for batchsize in batchsizes:
                    eval_duration = self.benchmark_model('eval', 'pytorch', model, precision,batch_size=batchsize)
                    train_duration = self.benchmark_model('train', 'pytorch', model, precision,batch_size=batchsize)
                    print("{}'s batch-size {} {} eval at {}: {}ms avg, train at {}: {}ms avg".\
                            format('pytorch',batchsize, model, precision, round(eval_duration, 1),precision, round(train_duration, 1)))
                    results[precision].append(eval_duration)
                    results[precision].append(train_duration)

        return results

if __name__ == '__main__':
    results = Benchmark().benchmark_framework()
    pickle.dump(results, open('{}_results.pkl'.format(args.framework), 'wb'))

 




