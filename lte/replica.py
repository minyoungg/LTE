import torch
import torch.nn as nn
import copy
import lte.misc.distributed as D
from lte import ReplicaLayer


class MultiheadReplicaLayer(ReplicaLayer):
    """
    Replica layer. Creates N replicas of the module.
    
    Args:
        target_module (nn.Module): the module to replicate
        num_heads (int): the number of replicas to create
    
    NOTE: does not support module that outputs tuples
    """

    def __init__(self, target_module, num_heads, mode="ddp"):
        super().__init__()
        self.num_heads = num_heads
        self.replicas = []
        for i in range(num_heads):
            if mode == "ddp":                
                self.replicas.append(copy.deepcopy(target_module))
            else:
                device_id = f"cuda:{i % D.num_visible_devices()}"
                self.replicas.append(
                    copy.deepcopy(target_module).to(device=device_id)
                )
        self.replicas = nn.ModuleList(self.replicas)
        return

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): the input tensor
        Returns:
            outputs (torch.Tensor): the output tensor
        """
        if not self.training:
            replica_device = next(self.replicas[0].parameters()).device
            outputs = self.replicas[0](inputs.to(device=replica_device))
        else:
            xs = inputs.chunk(self.num_heads)
            outputs = []
            for x, replica in zip(xs, self.replicas):
                replica_device = next(replica.parameters()).device
                outputs.append(replica(x.to(device=replica_device)))
            outputs = torch.cat([x.to(device=inputs.device) for x in outputs])
        return outputs

    @torch.no_grad()
    def merge_parameters(self):
        """ compute average across N devices and then assign to all copies in replica """

        # compute average of the parameter
        avg_params = [torch.zeros_like(p) for p in self.replicas[0].parameters()]
        for replica in self.replicas:
            for p, avg_p in zip(replica.parameters(), avg_params):
                avg_p += p

        avg_params = [p / self.num_heads for p in avg_params]

        # assign to all replicas (clone and assign to correct device)
        for replica in self.replicas:
            for p, avg_p in zip(replica.parameters(), avg_params):
                p.data = avg_p.clone().to(device=p.device)
        return
