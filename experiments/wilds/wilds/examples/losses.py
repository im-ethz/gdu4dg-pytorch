import torch.nn as nn
from ..wilds.common.metrics.loss import ElementwiseLoss, Loss, MultiTaskLoss
from ..wilds.common.metrics.all_metrics import MSE, mse_loss
from .utils import cross_entropy_with_logits_loss

from .algorithms.gdu_pytorch.loss import LayerLoss

def initialize_loss(loss, config):

    if loss == 'gdu_loss':

        if config.dataset == 'ogb-molpcba':
            criterion = nn.BCEWithLogitsLoss(reduction='none')

        elif config.dataset == 'poverty':
            criterion = mse_loss

        else:
            criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

        layer_criterion = LayerLoss(
            name='loss',
            device=config.device,
            criterion= criterion,
            sigma=config.loss_kwargs['sigma'],
            lambda_OLS=config.loss_kwargs['lambda_OLS'],
            lambda_orth=config.loss_kwargs['lambda_orth'],
            lambda_sparse=config.loss_kwargs['lambda_sparse'],
            orthogonal_loss=config.loss_kwargs['orthogonal_loss'],
            sparse_coding=config.loss_kwargs['sparse_coding']
        )

        if config.dataset == 'ogb-molpcba':
            return MultiTaskLoss(loss_fn=layer_criterion, name='gdu_multitask_bce')
        elif config.dataset == 'poverty':
            return MSE(loss_fn=layer_criterion, name='gdu_mse')
        elif config.dataset == 'py150':
            return MultiTaskLoss(loss_fn=layer_criterion, name='gdu_lm_cross_entropy')
        else:
            return ElementwiseLoss(loss_fn=layer_criterion, name='gdu_cross_entropy')

    if loss == 'cross_entropy':
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))

    elif loss == 'lm_cross_entropy':
        return MultiTaskLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))

    elif loss == 'mse':
        return MSE(name='loss')

    elif loss == 'multitask_bce':
        return MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none'))

    elif loss == 'fasterrcnn_criterion':
        from models.detection.fasterrcnn import FasterRCNNLoss
        return ElementwiseLoss(loss_fn=FasterRCNNLoss(config.device))

    elif loss == 'cross_entropy_logits':
        return ElementwiseLoss(loss_fn=cross_entropy_with_logits_loss)

    else:
        raise ValueError(f'loss {loss} not recognized')
