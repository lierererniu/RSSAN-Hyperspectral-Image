import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import torch


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)  # 输出对角线上的元素
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))  # list_diag/list_raw_sum C内核
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(y_pred, y_test, name):
    classification = classification_report(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    evaluate = [oa, aa, kappa]
    each_acc = list(np.round(each_acc * 100, 2))  # 百分制，取两位有效
    return classification, confusion, evaluate, each_acc

def train(trainloader, model, criterion, optimizer, use_cuda):
    model.train()
    accs = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses[batch_idx] = loss.item()
        accs[batch_idx] = accuracy(outputs.data, targets.data)[0].item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.average(losses), np.average(accs)


def test(testloader, model, criterion, use_cuda):
    model.eval()
    accs = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs = model(inputs)
        losses[batch_idx] = criterion(outputs, targets).item()
        accs[batch_idx] = accuracy(outputs.data, targets.data, topk=(1,))[0].item()
    return np.average(losses), np.average(accs)


def predict(testloader, model, use_cuda):
    model.eval()
    predicted = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs = inputs.cuda()
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        [predicted.append(a) for a in model(inputs).data.cpu().numpy()]
    return np.array(predicted)


def adjust_learning_rate(optimizer, epoch, lr):
    """
    动态lr 每75次epoch调整一次
    :param optimizer: 优化器
    :param epoch: 迭代次数
    :param lr: 学习率
    :return: None
    """
    lr = lr * (0.1 ** (epoch // 75))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
