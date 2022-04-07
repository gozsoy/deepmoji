import os
import argparse

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, \
                                        ModelCheckpoint, TensorBoard

from utils import prepare_embeddings
from utils import get_data_loader, load_config
from utils import get_model, get_optimizer
from utils import set_device, set_seeds
from utils import F1Score


def train(cfg,lookup_layer,embedding_layer):
    
    train_ds = get_data_loader(cfg,lookup_layer,embedding_layer,split=0)
    print(f'train_ds ready')
    valid_ds = get_data_loader(cfg,lookup_layer,embedding_layer,split=1)
    print(f'valid_ds ready')
    test_ds = get_data_loader(cfg,lookup_layer,embedding_layer,split=2)
    print(f'test_ds ready')
    

    model = get_model(cfg)
    optimizer = get_optimizer(cfg)
    loss = BinaryCrossentropy(from_logits=True)

    main_dir,_ = os.path.split(cfg['data_dir'])
    checkpointer = ModelCheckpoint(filepath=os.path.join(main_dir,'checkpoints',cfg['dataset'],cfg['experiment_name']), monitor='val_loss', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=os.path.join(main_dir,'logs',cfg['dataset'],cfg['experiment_name']), write_images=True)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    

    f1_score = F1Score(from_logits=True,name='f1_score')

    model.compile(optimizer=optimizer, loss=loss, metrics=[f1_score])

    model.fit(x = train_ds, epochs=cfg['n_epochs'], verbose=1, shuffle=True,
              callbacks=[early_stopper,lr_scheduler,checkpointer,tensorboard], validation_data=valid_ds)

    model.load_weights(os.path.join(main_dir,'checkpoints',cfg['dataset'],cfg['experiment_name']))  # load the best model
    model.evaluate(x=test_ds)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file.")

    _args = parser.parse_args()

    cfg = load_config(_args)

    set_seeds(cfg)
    set_device(cfg)

    if cfg['model'] == 'BertBasedClassifier':
        train(cfg,None,None)

    else: # DeepMoji variants
        lookup_layer,embedding_layer = prepare_embeddings(cfg)
        train(cfg,lookup_layer,embedding_layer)



