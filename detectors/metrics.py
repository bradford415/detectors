import torch


@torch.no_grad()
def topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k;
    multiplies by 100 to get the percentage

    Args:
        output: TODO
        target:
        topk:
    """
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]

    maxk = max(topk)
    batch_size = target.size(0)

    # find the largest top-k prediction indices; the highest `k` is used and the
    # the smaller `k` values, if specified, are computed after
    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)

    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
