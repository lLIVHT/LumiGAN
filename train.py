import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import util.util as util
import random

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream,Loader=yaml.FullLoader)
def set_parse(opt):
    opt.dataroot = 'dataset/train'
    opt.name = 'modelname'
    # opt.name = 'lightweight'
    opt.batchSize = 2
    opt.patchD = False
    opt.patch_vgg = True
    opt.use_ragan = True
    opt.hybrid_loss = True
    opt.single = 0
    opt.colorloss = 1
    opt.addcolor = 1
    opt.addnoise = 1
    opt.fea = 1
    opt.pool = 1
    opt.input_2 = 1
    opt.realBf = 1
    opt.light = 0
    opt.net = 'network'
    opt.patchD_3 = 6
    opt.lr = 0.0001
    opt.lr_D = 0.0001
    opt.flag = 0
    opt.gan_opt = False
    opt.niter = 200
    opt.niter_decay = 200
    opt.gpu_ids = [0]
    # opt.dataset_mode = 'unaligned1'
    # opt.continue_train = 1
# seed = random.randint(1, 10000)
# seed = 42
# util.set_random_seed(seed)
opt = TrainOptions().parse()
set_parse(opt)
config = get_config(opt.config)


data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
# print('#random seed = %d' % seed)
model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0
if __name__=='__main__':
    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter = total_steps - dataset_size * (epoch - 1)
            model.set_input(data)
            model.optimize_parameters(epoch, i)

            if total_steps % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(epoch), epoch)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors(epoch)
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if epoch % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))


        if epoch > opt.niter:
            model.update_learning_rate()

    print('#random seed = %d' % seed)