import yaml

def load_config(args):

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_data_loader(cfg, split):
    if cfg['dataset'] == "h36m":
        selected_dataset = None
    else:
        raise NotImplementedError('dataset not implemented')

    return 


def get_model(cfg):
    if cfg["model"] == "seq2seq":
        model = None #Seq2SeqModel(cfg)
    else:
        raise NotImplementedError('model not implemented')

    return model


def get_optimizer(cfg, model):
    """ Create an optimizer. """

    if cfg["optimizer"] == "adam":
        optimizer = None #optim.Adam(params=model.parameters(
        #), lr=cfg["lr"],  weight_decay=cfg["weight_decay"])
    else:
        raise NotImplementedError('optimizer not implemented')

    return optimizer