from mAtt.utils import *
from mAtt import StiefelParameter

class MixOptimizer(object):
    """This is a meta optimizer which uses other optimizers for updating parameters
        and remap all StiefelParameter parameters to Stiefel space after they have been updated.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.state = {}

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        temp_dict = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    if id(p) not in self.state:
                        self.state[id(p)] = p.data.clone()
                    else:
                        self.state[id(p)].fill_(0).add_(p.data)
                    
                    p.data.fill_(0)
                    trans = orthogonal_projection(p.grad.data, p.data)
                    p.grad.data.fill_(0).add_(trans)

        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            
            for p in group['params']:
                
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    trans = retraction(p.data, self.state[id(p)])
                    p.data.fill_(0).add_(trans)
        return loss

