import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau



def init_optimizer( model,
                    name ='Adam',
                    lr=1e-3,
                    weight_decay=0, #5e-4,
                    momentum=0.99,
                    sgd_dampening=0,
                    sgd_nesterov=False,
                    adam_beta1=0.9,
                    adam_beta2=0.999,
                    factor=1,
                    patience=0
                   ):

    if name == 'Adam':

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=patience, mode='min', verbose=True)
        return optimizer, scheduler


    elif name == 'Amsgrad':

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(adam_beta1, adam_beta2), amsgrad=True)
        scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=patience, mode='min', verbose=True)
        return optimizer, scheduler

    elif name == 'SGD':

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=sgd_dampening, nesterov=sgd_nesterov)
        scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=patience, mode='min', verbose=True)
        return optimizer, scheduler

    else:

        raise ValueError("Unsupported optimizer: {}".format(name))


