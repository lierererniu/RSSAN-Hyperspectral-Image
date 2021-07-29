import argparse
import copy
import os
import time
import torch.backends.cudnn
from dataprocess import load_dataset
from train import *
import model.network as RSSAN
import torch
import numpy as np
from module import Mylogger, plot_loss_and_accuracy, outputstr
from torchsummary import summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RSSAN Training')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', default=0.0003, type=float, help='IN, UP: 0.0003, KSC: 0.0001')
    parser.add_argument('--path', default='./Dataset/', type=str, help='dataset location')
    parser.add_argument('--result', default='./result/', type=str, help='model and figure')
    parser.add_argument('--dataset', default='IN', type=str, help='dataset name : IN, PU, SA, KSC')
    parser.add_argument('--train_size', default=16, type=int, help='train batch_size')
    parser.add_argument('--test_size', default=16, type=int, help='test batch_size')
    parser.add_argument('--kernel_size', default=3, type=int, help='kernel_size ')
    parser.add_argument('--depth', default=32, type=int, help='IN、KSC:32 UP:8 ')
    parser.add_argument('--patchsize', default=17, type=int, help='spatial-spectral patch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--train_percent', default=0.2, type=float, help='samples of train set')
    parser.add_argument('--test_percent', default=0.675, type=float, help='samples of test set')
    parser.add_argument('--val_percent', default=0.125, type=float, help='samples of val set')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    # def output dir
    LOG_PATH = './train.log'  # 日志文件路径
    result_dir = args.result + args.dataset
    model_dir = os.path.join(args.result, args.dataset + '_weights.pkl')
    LOG_LEVEL = 'DEBUG'  # 日志级别
    LOG_FORMAT = '%(asctime)s - %(message)s'  # 每条日志输出格式
    logger = Mylogger(LOG_PATH, LOG_LEVEL, LOG_FORMAT).get_logger('train')

    logger.info('---------loading Dataset----------')
    train_loader, test_loader, val_loader, kinds, bands, datashape = load_dataset(args.path, args.dataset.upper(),
                                                                                  args.train_size, args.patchsize,
                                                                                  args.test_size)
    logger.info('---------initialize model----------')
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    model = RSSAN.RSSAN(kinds, bands, 3, args.depth, 1, 1, args.patchsize)
    model = model.cuda()
    summary(model, datashape, args.train_size)
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizer = torch.optim.RMSprop(model.parameters(), args.lr)
    logger.info('---------initialize parameter----------')
    best_acc = -1
    loss_list = []
    train_acc_list = []
    val_acc_list = []
    best_loss = 0.
    best_train = 0.
    best_epoch = 0
    best_model = None
    best_optimizer = None
    logger.info('---------begin to train %s dateset----------' %args.dataset)
    epoch = 0
    time_start = time.time()
    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch, args.lr)
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, use_cuda)
        val_loss, val_acc = test(val_loader, model, criterion, use_cuda)
        print("epoch:", epoch, "train loss:", train_loss, "train acc", train_acc,
              "loss:", val_loss, "val acc", val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_train = train_acc
            best_loss = train_loss
            best_epoch = epoch + 1
            best_model = copy.deepcopy(model)

        loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

    best_records = {'epoch': best_epoch, 'loss': best_loss,
                    'train_acc': best_train, 'val_acc': best_acc}

    time_end = time.time()
    train_time = time_end - time_start
    mean_time = train_time / args.epochs
    print('The training takes {} seconds.'.format(round(train_time, 4)))
    print('The average time is {} seconds.'.format(round(mean_time, 4)))
    # save model
    logger.info('---------save model----------')
    torch.save(best_model.state_dict(), model_dir)
    graph_data = {
        'loss': loss_list,
        'train accuracy': train_acc_list,
        'val accuracy': val_acc_list,
    }
    logger.info('---------plot loss and accuracy----------')
    plot_loss_and_accuracy(graph_data=graph_data, save_dir=result_dir,
                           title='Loss and Accuracy')
    logger.info('---------Log results----------')
    classification, confusion, evaluate, each_acc = reports(
        np.argmax(predict(test_loader, best_model, use_cuda), axis=1), np.array(test_loader.dataset.__labels__()),
        'IP')
    outputstr(result_dir, best_records, confusion, classification)
    print('OA, AA, kappa:', evaluate)
    print('each_acc', each_acc)


