
def create_model(opt):
    model = None
    # print(opt.model)
    if opt.model == 'single':
        # assert(opt.dataset_mode == 'unaligned')
        if opt.single == 1:
            from .single_model1 import SingleModel
        elif opt.single == 2:
            from .single_model2 import SingleModel
        elif opt.single == 3:
            from .single_model3 import SingleModel
        elif opt.single == 4:
            from .single_model4 import SingleModel
        elif opt.single == 11:
            from .single_model11 import SingleModel
        else:
            from .single_model import SingleModel
        model = SingleModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
