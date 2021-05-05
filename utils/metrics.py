import torch
from pytorch_lightning.metrics import Metric
from skimage.measure import compare_ssim

class BatchPSNR(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, x: torch.Tensor, y: torch.Tensor):
        assert x.shape == y.shape
        with torch.no_grad(): losses = x.data.view(x.shape[0],-1).sub(y.data.view(x.shape[0],-1)).pow(2).mean(-1)
        self.correct += (1/losses).log10().mul(10).sum()
        self.total += x.shape[0]

    def compute(self):
        return self.correct.float() / self.total

def calc_batch_psnr(x,y):
    correct = 0
    total = 0
    with torch.no_grad(): losses = x.data.view(x.shape[0],-1).sub(y.data.view(x.shape[0],-1)).pow(2).mean(-1)
    correct += (1/losses).log10().mul(10).sum()
    total += x.shape[0]
    return correct,total

class BatchSSIM(Metric):
    def __init__(self, ddp_sync_on_step=False):
        super().__init__(ddp_sync_on_step=ddp_sync_on_step)

        self.add_state("correct", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, x: torch.Tensor, y: torch.Tensor):
        assert x.shape == y.shape
        with torch.no_grad(): 
            for (x_now,y_now) in zip(x,y):
                self.correct += compare_ssim(x_now.detach().cpu().numpy(),y_now.detach().cpu().numpy())
                self.total += 1

    def compute(self):
        return self.correct.float() / self.total


def calc_batch_ssim(x,y):
    correct = 0
    total = 0
    with torch.no_grad(): 
        for (x_now,y_now) in zip(x,y):
            correct += compare_ssim(x_now.detach().permute(1,2,0).cpu().numpy(),y_now.detach().permute(1,2,0).cpu().numpy(),multichannel=True)
            total += 1
    return (correct,total)            

    