class Optimizer(object):
    def __init__(self,params,defaults):

        self.defaults=defaults

        self.param_groups=[]

        param_groups=list(params)

        if not isinstance(param_groups[0],dict):
            param_groups=[{'params':param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self,param_group):
        assert isinstance(param_group,dict),"param group must be a dict"

        params=param_group['params']

        param_group['params']=list(params)

        for name,default in self.defaults.items():
            param_group.setdefault((name,default))

        params=param_group['params']
        param_set=set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        self.param_groups.append(param_group)

    def step(self):
        r"""Performs a single optimization step (parameter update).

                Arguments:
                    closure (callable): A closure that reevaluates the model and
                        returns the loss. Optional for most optimizers.

                .. note::
                    Unless otherwise specified, this function should not modify the
                    ``.grad`` field of the parameters.
                """
        raise NotImplementedError

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['parms']:
                if p.grad is not None:
                            p.grad=0
                            #p.grad.zero_()#梯度清零，考虑一下怎么写





