from torch import optim
import itertools

def get_optim_and_scheduler(network, epochs, lr, train_all, nesterov=False):
    params=[]
    if train_all:
        for net in network:
            params.append(net.parameters())
    else:
        params = network.get_params(lr)
    optimizer = optim.SGD(itertools.chain(*params), weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    # optimizer = optim.Adam(itertools.chain(*params), weight_decay=.0005*lr/.01, lr=lr)
    # optimizer = torch.optim.SGD((par for model in models for par in model.parameters()),
    #                             lr=0.01)

    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d" % step_size)
    return optimizer, scheduler
