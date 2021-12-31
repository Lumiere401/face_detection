import torch


def accuracy(output, target, topk=[1]):
    """Computes the precision@k for the specified values of k"""
    # maxk = max(topk)
    batch_size = target.size(0)

    # _, pred = output.topk(maxk, 1, True, True)
    # pred = pred.t()
    pred = torch.round(torch.sigmoid(output))
    correct = pred.eq(target)
    correct = correct.squeeze()
    res = []
    num = 0
    for k in range(batch_size):
        if correct[k]==True:
            num +=1
    res = num/batch_size
    return res