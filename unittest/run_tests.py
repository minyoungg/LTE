import unittest
import torch
import torch.nn as nn

from lte import LTEConfig, prepare_model_for_lte, MultiheadReplicaLayer


class TestModelParameters(unittest.TestCase):
    rtol = 1e-3
    atol = 1e-6
    lora_r = 8
    lora_alpha = 16
    num_heads = 4
    in_dim = 16
    out_dim = 32
    bias = True
    

    def setUp(self):
        torch.manual_seed(0)
        self.dmp_model = nn.Sequential(nn.Linear(self.in_dim, self.out_dim, bias=self.bias)).cuda()
        self.dmp_model = prepare_model_for_lte(
            self.dmp_model,
            LTEConfig.default(
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                num_heads=self.num_heads,
            ),
            mode='dmp',
            strict=True,
        ).double()

        torch.manual_seed(0)
        self.ddp_model = nn.Sequential(nn.Linear(self.in_dim, self.out_dim, bias=self.bias)).cuda()
        self.ddp_model = prepare_model_for_lte(
            self.ddp_model,
            LTEConfig.default(
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                num_heads=self.num_heads,
            ),
            mode='ddp',
            strict=True,
        ).double()
        
        for i in range(4):
            self.dmp_model[0].lora_A[i].weight.data = \
                self.ddp_model[0].lora_A_weight[i].data.clone().to(
                    device=self.dmp_model[0].lora_A[i].weight.device
                )
                
            if self.dmp_model[0].lora_bias:
                self.dmp_model[0].lora_A[i].bias.data = \
                    self.ddp_model[0].lora_A_bias[i].data.clone().to(
                        device=self.dmp_model[0].lora_A[i].bias.device
                    )
            
    def test_parameter_count_difference(self):
        # test if there is no parameter being tied
        ddp_param_count = sum(p.numel() for p in self.ddp_model.parameters())
        dmp_param_count = sum(p.numel() for p in self.dmp_model.parameters())
        self.assertNotEqual(ddp_param_count, dmp_param_count)

    def test_lora_parameter_synchronization(self):
        for i in range(4):
            dmp_w = self.dmp_model[0].layers[i].weight
            dmp_a = self.dmp_model[0].lora_A[i].weight
            dmp_b = self.dmp_model[0].lora_B[i].weight

            ddp_w = self.ddp_model[0].weight
            ddp_a = self.ddp_model[0].lora_A_weight[i]
            ddp_b = self.ddp_model[0].lora_B_weight[i]

            self.assertTrue(torch.allclose(ddp_w.cpu(), dmp_w.cpu()))
            self.assertTrue(torch.allclose(ddp_a.cpu(), dmp_a.cpu()))
            self.assertTrue(torch.allclose(ddp_b.cpu(), dmp_b.cpu()))
            
    def test_training_behavior(self):
        dmp_opt = torch.optim.AdamW(self.dmp_model.parameters(), lr=0.1)
        ddp_opt = torch.optim.AdamW(self.ddp_model.parameters(), lr=0.1)

        for _ in range(100):
            x = torch.randn(8, 128, self.in_dim).double().cuda()
            dmp_out = self.dmp_model(x)
            ddp_out = self.ddp_model(x)

            dmp_out.sum().backward()
            ddp_out.sum().backward()
            
            dmp_opt.step()
            ddp_opt.step()
                        
            dmp_opt.zero_grad()
            ddp_opt.zero_grad()

            self.assertTrue(torch.allclose(dmp_out, ddp_out, rtol=self.rtol, atol=self.atol))
        return self.dmp_model, self.ddp_model
        

    def test_training_behavior_and_merging_v1(self):
        self.test_training_behavior()

        # Test merging behavior
        self.dmp_model[0].merge_parameters()
        self.dmp_model[0].reset_lora_parameters()
        self.ddp_model[0].merge_parameters()
        self.ddp_model[0].reset_lora_parameters()

        x = torch.randn(8, 128, self.in_dim).double().cuda()

        dmp_out = self.dmp_model(x)
        ddp_out = self.ddp_model(x)

        self.assertTrue(torch.allclose(dmp_out, ddp_out, rtol=self.rtol, atol=self.atol))
    
    def test_training_behavior_and_merging_v2(self):
        self.test_training_behavior()

        # Test merging behavior (no resets)
        self.dmp_model[0].merge_parameters()
        # self.dmp_model[0].reset_lora_parameters()
        self.ddp_model[0].merge_parameters()
        # self.ddp_model[0].reset_lora_parameters()

        x = torch.randn(8, 128, self.in_dim).double().cuda()

        dmp_out = self.dmp_model(x)
        ddp_out = self.ddp_model(x)

        self.assertTrue(torch.allclose(dmp_out, ddp_out, rtol=self.rtol, atol=self.atol))
        
        
    def test_training_behavior_and_merging_without_resetting(self):
        self.test_training_behavior()

        # Test merging behavior
        self.dmp_model[0].merge_parameters()
        self.ddp_model[0].merge_parameters()

        x = torch.randn(8, 128, self.in_dim).double().cuda()

        dmp_out = self.dmp_model(x)
        ddp_out = self.ddp_model(x)
        self.assertTrue(torch.allclose(dmp_out, ddp_out, rtol=self.rtol, atol=self.atol))
                


    def test_dmp_ddp_equivalence(self):
        
        ddp_opt = torch.optim.SGD(self.ddp_model.parameters(), lr=0.01)
        dmp_opt = torch.optim.SGD(self.dmp_model.parameters(), lr=0.01)
        
        # merging every iteration should have equivalent behavior
        for _ in range(10):
            x = torch.randn(16, 128, self.in_dim).double().cuda()
            y = torch.randn(16, 128, self.out_dim).double().cuda()
            
            dmp_out = self.dmp_model(x)
            ddp_out = self.ddp_model(x)
            
            (dmp_out - y).mean().backward()
            (ddp_out - y).mean().backward()
            
            with torch.no_grad():
                self.assertTrue(torch.allclose(dmp_out, ddp_out, rtol=self.rtol, atol=self.atol))
                
                # for each lora param check identical gradient
                for i in range(self.num_heads):
                    ddp_grad_A = self.ddp_model[0].lora_A_weight.grad[i]
                    ddp_grad_B = self.ddp_model[0].lora_B_weight.grad[i]
                    dmp_grad_A = self.dmp_model[0].lora_A[i].weight.grad
                    dmp_grad_B = self.dmp_model[0].lora_B[i].weight.grad
                    
                    self.assertTrue(torch.allclose(ddp_grad_A, dmp_grad_A, rtol=self.rtol, atol=self.atol))
                    self.assertTrue(torch.allclose(ddp_grad_B, dmp_grad_B, rtol=self.rtol, atol=self.atol))
                    
                    if self.ddp_model[0].lora_bias:
                        ddp_grad_A = self.ddp_model[0].lora_A_bias.grad[i]
                        ddp_grad_B = self.ddp_model[0].lora_B_bias.grad[i]
                        dmp_grad_A = self.dmp_model[0].lora_A[i].bias.grad
                        dmp_grad_B = self.dmp_model[0].lora_B[i].bias.grad
                        
                        self.assertTrue(torch.allclose(ddp_grad_A, dmp_grad_A, rtol=self.rtol, atol=self.atol))
                        self.assertTrue(torch.allclose(ddp_grad_B, dmp_grad_B, rtol=self.rtol, atol=self.atol))
                    
            ddp_opt.step()
            dmp_opt.step()
            
            ddp_opt.zero_grad()
            dmp_opt.zero_grad()
            
            self.ddp_model[0].merge_parameters()
            self.dmp_model[0].merge_parameters()
            
                    
    def test_mhlora_dmp_lte_equivalence(self):
        
        mhlora = MultiheadLoRA(
            self.in_dim, 
            self.out_dim, 
            self.num_heads, 
            self.lora_r, 
            self.lora_alpha, 
            bias=self.bias
        ).cuda().double()
        
        self.dmp_model = self.dmp_model.double()
                
        # synchronize parameters 
        for i in range(4):
            mhlora.linear.weight.data = self.dmp_model[0].layers[i].weight.data.clone()
            mhlora.lora_A[i].weight.data = self.dmp_model[0].lora_A[i].weight.data.clone()
            mhlora.lora_B[i].weight.data = self.dmp_model[0].lora_B[i].weight.data.clone()
            
            if self.dmp_model[0].bias is not None:
                mhlora.linear.bias.data = self.dmp_model[0].layers[i].bias.data.clone()
            
            if self.dmp_model[0].lora_bias:
                mhlora.lora_A[i].bias.data = self.dmp_model[0].lora_A[i].bias.data.clone()
                mhlora.lora_B[i].bias.data = self.dmp_model[0].lora_B[i].bias.data.clone()
        
        mhlora_opt = torch.optim.SGD(mhlora.parameters(), lr=0.01)
        lte_opt = torch.optim.SGD(self.dmp_model.parameters(), lr=0.01)
        
        # merging every iteration should have equivalent behavior
        for _ in range(10):
            x1 = torch.randn(2, 128, self.in_dim).double().cuda()
            y1 = torch.randn(2, 128, self.out_dim).double().cuda()
            mhlora_out = mhlora(x1)
            
            x2 = x1.repeat(self.num_heads, 1, 1)
            y2 = y1.repeat(self.num_heads, 1, 1).unflatten(0, (self.num_heads, 2))

            lte_out = self.dmp_model(x2)
            lte_out = lte_out.unflatten(0, (self.num_heads, 2))

            (mhlora_out - y1).mean().backward()
            (lte_out - y2).mean().backward()
            
            with torch.no_grad():
                self.assertTrue(torch.allclose(mhlora_out, lte_out.mean(0), rtol=self.rtol, atol=self.atol))
                
                # for each lora param check identical gradient
                for i in range(self.num_heads):
                    mhlora_A_grad = mhlora.lora_A[i].weight.grad
                    mhlora_B_grad = mhlora.lora_B[i].weight.grad
                    lte_A_grad = self.dmp_model[0].lora_A[i].weight.grad
                    lte_B_grad = self.dmp_model[0].lora_B[i].weight.grad

                    self.assertTrue(torch.allclose(mhlora_A_grad, lte_A_grad, rtol=self.rtol, atol=self.atol))
                    self.assertTrue(torch.allclose(mhlora_B_grad, lte_B_grad, rtol=self.rtol, atol=self.atol))

                    if self.dmp_model[0].lora_bias:
                        mhlora_A_grad = mhlora.lora_A[i].bias.grad
                        mhlora_B_grad = mhlora.lora_B[i].bias.grad
                        lte_A_grad = self.dmp_model[0].lora_A[i].bias.grad
                        lte_B_grad = self.dmp_model[0].lora_B[i].bias.grad

                        self.assertTrue(torch.allclose(mhlora_A_grad, lte_A_grad, rtol=self.rtol, atol=self.atol))
                        self.assertTrue(torch.allclose(mhlora_B_grad, lte_B_grad, rtol=self.rtol, atol=self.atol))
            
            mhlora_opt.step()
            lte_opt.step()
            
            mhlora_opt.zero_grad()
            lte_opt.zero_grad()
            
            self.dmp_model[0].merge_parameters()
            # self.dmp_model[0].reset_lora_parameters()        
            
            
    def test_mhlora_ddp_lte_equivalence(self):
        
        mhlora = MultiheadLoRA(
            self.in_dim, 
            self.out_dim, 
            self.num_heads, 
            self.lora_r, 
            self.lora_alpha, 
            bias=self.bias
        ).cuda().double()
        
        self.ddp_model = self.ddp_model.double()
                
        # synchronize parameters 
        for i in range(4):
            mhlora.linear.weight.data = self.ddp_model[0].weight.data.clone()
            mhlora.lora_A[i].weight.data = self.ddp_model[0].lora_A_weight[i].data.clone()
            mhlora.lora_B[i].weight.data = self.ddp_model[0].lora_B_weight[i].data.clone()
            
            if self.ddp_model[0].bias is not None:
                mhlora.linear.bias.data = self.ddp_model[0].bias.data.clone()
            
            if self.ddp_model[0].lora_bias:
                mhlora.lora_A[i].bias.data = self.ddp_model[0].lora_A_bias[i].data.clone()
                mhlora.lora_B[i].bias.data = self.ddp_model[0].lora_B_bias[i].data.clone()
        
        mhlora_opt = torch.optim.SGD(mhlora.parameters(), lr=0.01)
        lte_opt = torch.optim.SGD(self.ddp_model.parameters(), lr=0.01)
        
        # merging every iteration should have equivalent behavior
        for _ in range(10):
            x1 = torch.randn(2, 128, self.in_dim).double().cuda()
            y1 = torch.randn(2, 128, self.out_dim).double().cuda()
            mhlora_out = mhlora(x1)
            
            x2 = x1.repeat(self.num_heads, 1, 1)
            y2 = y1.repeat(self.num_heads, 1, 1).unflatten(0, (self.num_heads, 2))

            lte_out = self.ddp_model(x2)
            lte_out = lte_out.unflatten(0, (self.num_heads, 2))

            (mhlora_out - y1).mean().backward()
            (lte_out - y2).mean().backward()
            
            with torch.no_grad():
                # print(mhlora_out, lte_out.mean(0))
                # import ipdb; ipdb.set_trace()
                self.assertTrue(torch.allclose(mhlora_out, lte_out.mean(0), rtol=self.rtol, atol=self.atol))
                
                # for each lora param check identical gradient
                for i in range(self.num_heads):
                    mhlora_A_grad = mhlora.lora_A[i].weight.grad
                    mhlora_B_grad = mhlora.lora_B[i].weight.grad
                    lte_A_grad = self.ddp_model[0].lora_A_weight.grad[i]
                    lte_B_grad = self.ddp_model[0].lora_B_weight.grad[i]

                    self.assertTrue(torch.allclose(mhlora_A_grad, lte_A_grad, rtol=self.rtol, atol=self.atol))
                    self.assertTrue(torch.allclose(mhlora_B_grad, lte_B_grad, rtol=self.rtol, atol=self.atol))

                    if self.ddp_model[0].lora_bias:
                        mhlora_A_grad = mhlora.lora_A[i].bias.grad
                        mhlora_B_grad = mhlora.lora_B[i].bias.grad
                        lte_A_grad = self.ddp_model[0].lora_A_bias.grad[i]
                        lte_B_grad = self.ddp_model[0].lora_B_bias.grad[i]

                        self.assertTrue(torch.allclose(mhlora_A_grad, lte_A_grad, rtol=self.rtol, atol=self.atol))
                        self.assertTrue(torch.allclose(mhlora_B_grad, lte_B_grad, rtol=self.rtol, atol=self.atol))
            
            mhlora_opt.step()
            lte_opt.step()
            
            mhlora_opt.zero_grad()
            lte_opt.zero_grad()
            
            self.ddp_model[0].merge_parameters()
            # self.ddp_model[0].reset_lora_parameters()

    def test_mhlora_ddp_lte_equivalence_v2(self):
        
        torch.manual_seed(0)
        lte_model = nn.Sequential(nn.Linear(self.in_dim, self.out_dim, bias=self.bias)).cuda()
        lte_model = prepare_model_for_lte(
            lte_model,
            LTEConfig.default(
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                num_heads=1,
            ),
            mode='ddp',
            strict=True,
        ).double()
        
        torch.manual_seed(0)
        mhlora_model = nn.Sequential(nn.Linear(self.in_dim, self.out_dim, bias=self.bias)).cuda()
        mhlora_model = prepare_model_for_lte(
            mhlora_model,
            LTEConfig.default(
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                num_heads=1,
            ),
            mode='mhlora',
            strict=True,
        ).double()
        
        # check if parameters have same dynamics throughout training and merging
        # LTE with merge=1 should be equivalent to mhlora without merge

        # lte_opt = torch.optim.SGD(lte_model.parameters(), lr=0.01)
        # mhlora_opt = torch.optim.SGD(mhlora_model.parameters(), lr=0.01)

        lte_opt = torch.optim.Adam(lte_model.parameters(), lr=0.01)
        mhlora_opt = torch.optim.Adam(mhlora_model.parameters(), lr=0.01)
        
        for i in range(10):
            # optimize both with the same input 
            x = torch.randn(16, 128, self.in_dim).double().cuda()
            y = torch.randn(16, 128, self.out_dim).double().cuda()
            
            lte_out = lte_model(x)
            mhlora_out = mhlora_model(x)
            
            lte_loss = (lte_out - y).mean()
            mhlora_loss = (mhlora_out - y).mean()
            self.assertTrue(torch.allclose(lte_out, mhlora_out, rtol=self.rtol, atol=self.atol))

            lte_loss.backward()
            mhlora_loss.backward()
                        
            lte_opt.step()
            mhlora_opt.step()
            
            lte_opt.zero_grad()
            mhlora_opt.zero_grad()
            
            lte_model[0].merge_parameters()
        return

            
    def test_replica_behavior(self):
        
        model = nn.Sequential(nn.Linear(self.in_dim, self.out_dim, bias=self.bias)).double().cuda()
        model = MultiheadReplicaLayer(model, self.num_heads)
        
        x = torch.randn(1, self.in_dim).double().cuda()        
        x = x.repeat(self.num_heads, 1)
        ys = model(x).chunk(self.num_heads)
        
        # check all equal
        for y in ys:
            self.assertTrue(torch.allclose(ys[0], y, rtol=self.rtol, atol=self.atol))
        
        # check if the parameters are tied (updated 1 layer and see if other layers are updated)
        model.replicas[0][0].weight.data += 1
        ys = model(x).chunk(self.num_heads)

        for y in ys[1:]:
            self.assertFalse(torch.allclose(ys[0], y, rtol=self.rtol, atol=self.atol))
        
        # make sure merge parameters works
        model = nn.Sequential(nn.Linear(self.in_dim, self.out_dim, bias=self.bias)).double().cuda()
        model = MultiheadReplicaLayer(model, self.num_heads)
        model.merge_parameters()
        ys = model(x).chunk(self.num_heads)

        # nothing should have changed
        for y in ys:
            self.assertTrue(torch.allclose(ys[0], y, rtol=self.rtol, atol=self.atol))
            
            
        
class MultiheadLoRA(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, lora_r, lora_alpha, bias=False, lora_bias=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.lora_A = nn.ModuleList(
            [nn.Linear(in_dim, lora_r, bias=lora_bias) for _ in range(num_heads)])
        self.lora_B = nn.ModuleList(
            [nn.Linear(lora_r, out_dim, bias=lora_bias) for _ in range(num_heads)])
        self.s = lora_alpha / lora_r
        self.num_heads = num_heads
        
        for p in self.linear.parameters():
            p.requires_grad_(False)
        return 
    
    def forward(self, x):
        out = self.linear(x)
        for a, b in zip(self.lora_A, self.lora_B):
            out += (self.s / self.num_heads * b(a(x)))
        return out
        

if __name__ == '__main__':
    unittest.main()
