import logging
import sys
import time
from os import makedirs
from os.path import dirname, exists
from cmreslogging.handlers import CMRESHandler
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import torch.backends.cudnn

line_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'saddlebrown', 'orange',
               'yellow', 'slateblue']


# loggers = {}
#
# LOG_ENABLED = True  # 是否开启日志
# LOG_TO_CONSOLE = True  # 是否输出到控制台
# LOG_TO_FILE = True  # 是否输出到文件
# LOG_TO_ES = True  # 是否输出到 Elasticsearch
#
# LOG_PATH = './SnowDisaster.log'  # 日志文件路径
# LOG_LEVEL = 'DEBUG'  # 日志级别
# LOG_FORMAT = '%(levelname)s - %(asctime)s - %(module)s - %(message)s'  # 每条日志输出格式
#
# ELASTIC_SEARCH_HOST = 'eshost'  # Elasticsearch Host
# ELASTIC_SEARCH_PORT = 9200  # Elasticsearch Port
# ELASTIC_SEARCH_INDEX = 'runtime'  # Elasticsearch Index Name
# APP_ENVIRONMENT = 'dev'  # 运行环境，如测试环境还是生产环境


class Mylogger():
    def __init__(self, logpath, loglevel, log_format):
        self.LOG_PATH = logpath
        self.LOG_LEVEL = loglevel
        self.LOG_FORMAT = log_format
        self.loggers = {}
        self.LOG_ENABLED = True  # 是否开启日志
        self.LOG_TO_CONSOLE = True  # 是否输出到控制台
        self.LOG_TO_FILE = True  # 是否输出到文件
        self.LOG_TO_ES = False  # 是否输出到 Elasticsearch
        self.ELASTIC_SEARCH_HOST = 'eshost'  # Elasticsearch Host
        self.ELASTIC_SEARCH_PORT = 9200  # Elasticsearch Port
        self.ELASTIC_SEARCH_INDEX = 'runtime'  # Elasticsearch Index Name
        self.APP_ENVIRONMENT = 'dev'  # 运行环境，如测试环境还是生产环境

    def get_logger(self, name=None):
        """
        get logger by name
        :param name: name of logger
        :return: logger
        """
        if not name:
            name = __name__

        if self.loggers.get(name):
            return self.loggers.get(name)

        logger = logging.getLogger(name)
        logger.setLevel(self.LOG_LEVEL)

        # 输出到控制台
        if self.LOG_ENABLED and self.LOG_TO_CONSOLE:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(level=self.LOG_LEVEL)
            formatter = logging.Formatter(self.LOG_FORMAT)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        # 输出到文件
        if self.LOG_ENABLED and self.LOG_TO_FILE:
            # 如果路径不存在，创建日志文件文件夹
            log_dir = dirname(self.LOG_PATH)
            if not exists(log_dir):
                makedirs(log_dir)
            # 添加 FileHandler
            file_handler = logging.FileHandler(self.LOG_PATH, encoding='utf-8')
            file_handler.setLevel(level=self.LOG_LEVEL)
            formatter = logging.Formatter(self.LOG_FORMAT)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # 输出到 Elasticsearch
        if self.LOG_ENABLED and self.LOG_TO_ES:
            # 添加 CMRESHandler
            es_handler = CMRESHandler(hosts=[{'host': self.ELASTIC_SEARCH_HOST, 'port': self.ELASTIC_SEARCH_PORT}],
                                      # 可以配置对应的认证权限
                                      auth_type=CMRESHandler.AuthType.NO_AUTH,
                                      es_index_name=self.ELASTIC_SEARCH_INDEX,
                                      # 一个月分一个 Index
                                      index_name_frequency=CMRESHandler.IndexNameFrequency.MONTHLY,
                                      # 额外增加环境标识
                                      es_additional_fields={'environment': self.APP_ENVIRONMENT}
                                      )
            es_handler.setLevel(level=self.LOG_LEVEL)
            formatter = logging.Formatter(self.LOG_FORMAT)
            es_handler.setFormatter(formatter)
            logger.addHandler(es_handler)

        # 保存到全局 loggers
        self.loggers[name] = logger
        return logger


def plot_loss_and_accuracy(graph_data, save_dir, title=None, show=False):
    save_dir = mkdir(save_dir)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax_ = ax.twinx()

    i = 0
    lines = []
    for key, value in graph_data.items():
        if key == 'loss':
            l, = ax_.plot(range(1, len(value) + 1), value, color=line_colors[i],
                          label=key, linewidth=0.8)
            ax_.set_ylabel('loss', fontsize=10)
            ax_.tick_params(labelsize=8)
        else:
            l, = ax.plot(range(1, len(value) + 1), [v for v in value],
                         color=line_colors[i],
                         label=key, linewidth=0.8)
        lines.append(l)
        i += 1
    ax.set_ylabel('accuracy(%)', fontsize=10)
    ax.set_xlabel('epoch', fontsize=10)
    ax.tick_params(labelsize=8)
    legend_label = [l.get_label() for l in lines]
    ax_.legend(lines, legend_label, loc='best', fontsize=8, framealpha=0.8)

    if title is not None:
        ax.set_title(title, fontsize=13)
    save_dir = save_dir + '/loss_and_accuracy.png'
    plt.savefig(save_dir)
    if show:
        plt.show()
    plt.close()


def mkdir(path):
    # 判断目录是否存在
    folder = os.path.exists(path)
    # 判断结果
    if not folder:
        # 如果不存在，则创建新目录
        os.makedirs(path)
    else:
        return path
    return path


def outputstr(savepath, best_record, confusion, classfication,  evaluate, each_acc):
    sign = 100
    blank1 = 5
    blank2 = sign // 2 - 10 // 2
    s1 = '\n%s\n' % \
         time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    s2 = '\n%s\n%s%s\n%s\n' % ('#' * sign, ' ' * blank1, 'Training Records',
                               '#' * sign)
    s3 = '\n%s%s%s%s\n' % (' ' * blank1, 'epoch:' + str(best_record['epoch']), ' ' * blank1 * 4,
                           'loss:' + str(best_record['loss']))
    s4 = '\n%s%s%s%s\n' % (' ' * blank1, 'train_acc:' + str(best_record['train_acc']), ' ' * blank1 * 4,
                           'val_acc:' + str(best_record['val_acc']))
    s5 = '\n%s\n%s%s\n%s\n' % ('#' * sign, ' ' * blank2, 'Accuracies',
                               '#' * sign)
    s6 = classfication
    s7 = '\n%s%s\n' % ('OA, AA, kappa:', str(evaluate))
    s8 = '\n%s%s\n' % ('each-acc:', str(each_acc))
    savepath = savepath + '/' + 'result.txt'
    with open(savepath, 'w') as f:
        f.write(s1)
        f.write(s2)
        f.write(s3)
        f.write(s4)
        f.write(s5)
        f.write(s6)
        f.write(s7)
        f.write(s8)
        f.write('\ntest_confusion_matrix: \n')
        np.savetxt(f, confusion, fmt='%5d', newline='\n')


def set_seed(seed=1):  # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
