import torch
import lte


class MergeCondition():
    """
    LTE Merge scheduler
    
    Args:
        model (nn.Module): the model to merge
        merge_steps (int): the number of steps before merging
        method (str): the method to use for merging (default: 'step')
    
    Example::
        merge_scheduler = lte.misc.merge.MergeCondition(model, merge_steps=10, method='step')
        
        for ... in range(dataloader):
            # optimize model
            ...
            
            # step the merge scheduler (every 10 steps it will merge the model)
            merge_scheduler.step()
    """
    
    def __init__(
            self, 
            model, 
            merge_steps=1, 
            method='step', 
            ):
        
        self.model = model
        self.clock = 0
        self.merge_steps = merge_steps
        self.method = method
        self.reset_opt = True

        method_fn_map = {
            'step': self.step_condition,
        }
        self.cond_fn = method_fn_map[method]
        return    
    
    def peek(self):
        """ peeks whether model is planning to merge """
        return (self.clock + 1) % self.merge_steps == 0
    

    def step(self):
        """ increments step count """
        if self.model is None:
            raise RuntimeError('this merge condition is not registered to a model.')
                    
        self.clock += 1
        merged = self.cond_fn()
        return merged            
    
    @torch.no_grad()
    def merge(self):
        """ merges LTE and Replica layers """
        for m in self.model.modules():  
            
            if isinstance(m, lte.LTELayer):
                m.merge_parameters()
                        
            if isinstance(m, lte.ReplicaLayer):
                m.merge_parameters()        
        return
    
    def step_condition(self):
        """ simple step condition """
        if self.clock % int(self.merge_steps) == 0:
            self.merge()    
            self.clock = 0
            return True
        return False
    
    def register_model(self, model):
        """ registers model """
        self.model = model
    