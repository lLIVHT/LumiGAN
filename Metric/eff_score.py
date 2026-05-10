from thop import profile
import torch
from torchvision.models import resnet50
import time

def param_self_compute(model):
    parmas = 0
    for p in model.parameters():
        #print(p)
        parmas += p.numel()
    return parmas
def cal_eff_score(count = 100, use_cuda=True):

    # define input tensor
    from thop import profile
    import time
    inp_tensor = torch.rand(1, 3, 1200, 900).cuda() # NOTE: this is the shape for ACDC images
    # define model
    # model = resnet50().cuda()
    # get flops and params
    flops, params = profile(model, inputs=(inp_tensor, ))
    G_flops = flops * 1e-9
    M_params = params * 1e-6
    # get time
    start_time = time.time()
    count = 100
    for i in range(count):
        _ = model(inp_tensor)
    used_time = time.time() - start_time
    ave_time = used_time / count
    # print score
    print('FLOPs (G) = {:.4f}'.format(G_flops))
    print('Params (M) = {:.4f}'.format(M_params))
    print('Time (S) = {:.4f}'.format(ave_time))

if __name__ == "__main__":
    cal_eff_score()
