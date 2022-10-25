from optimizer import  Optimizer
def step(self):
    loss=None
    for group in self.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr=group['lr']
        v_old=0
        v_new=0

        for p in group['params']:
            if p.grad is None:
                continue
            d_p=p.grad#梯度，偏导数
            w=p.weight

            if weight_decay!=0:
                d_p=d_p+(weight_decay*w)
            if momentum!=0:
                if nesterov:
                    v_new=v_old*momentum+(1-dampening)*d_p
                    v_old=v_new
                    w=w-lr*v_new
                else:
                    v_new=d_p+v_old*momentum
                    v_old=v_new
                    w=w-lr*v_new
        return loss