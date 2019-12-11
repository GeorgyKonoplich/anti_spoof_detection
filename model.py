import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"]="1
import glob
import pandas as pd

from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.applications import DenseNet121, DenseNet201, MobileNetV2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from metrics import *
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split
from keras import Model
from keras.utils.training_utils import multi_gpu_model
from keras.models import Model

BATCH_SIZE = 2
img_rows = 400
img_cols = 400
nb_classes = 2
dir_overall = '/home/neuralbee/workspace/anti_spoof_detection/our_data/overall/'
weight_path='DensNet121_withaug_d5+d8+d9+d10+d11+d12_20_frames_0.3test_1e-7lr_400x400.h5'
number_epochs = 100

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
           serial-model holds references to the weights in the multi-gpu model.
           '''
        if 'load' in attrname or 'save' in attrname:
           return getattr(self._smodel, attrname)
        else:
           #return Model.__getattribute__(self, attrname)
           return super(ModelMGPU, self).__getattribute__(attrname)


def get_data(persons, dir_overall):
    X = []
    y = []
    for person in persons:
        if person in os.listdir(dir_overall):
            c = 0
            # for cls in ['real/',  'printed-color/']:#, 'printed-color-cut/']:
            for cls in ['real/', 'printed/', 'printed-color/', 'printed-color-cut/', 'replay/']:
                path = dir_overall + person + '/' + cls
                X = X + glob.glob(path + '/*.png')
                y = y + len(glob.glob(path + '/*.png')) * [c]
                c = 1
    return X, y


def exp_decay(epoch):
    initial_lrate = 0.0001
    k = 0.1
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate


def my_gen(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass


def show_loss(loss_history):
    epich = np.cumsum(np.concatenate(
        [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 10))
    _ = ax1.plot(epich,
                 np.concatenate([mh.history['loss'] for mh in loss_history]),
                 'b-')  # ,
    # epich, np.concatenate(
    # [mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training'])  # , 'Validation'])
    ax1.set_title('Loss')

    _ = ax2.plot(epich, np.concatenate(
        [mh.history['acc'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
            [mh.history['val_acc'] for mh in loss_history]),
                 'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('accuracy')

    _ = ax3.plot(epich, np.concatenate(
        [mh.history['auc'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
            [mh.history['val_auc'] for mh in loss_history]),
                 'r-')
    ax3.legend(['Training', 'Validation'])
    ax3.set_title('auc')

    _ = ax4.plot(epich, np.concatenate(
        [mh.history['f1'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
            [mh.history['val_f1'] for mh in loss_history]),
                 'r-')
    ax4.legend(['Training', 'Validation'])
    ax4.set_title('f1')


persons = list(set(os.listdir(dir_overall)))
persons_train, persons_test = train_test_split(persons, test_size=0.33, random_state=42)
X_train, y_train = get_data(persons_train, dir_overall)
X_test, y_test= get_data(persons_test, dir_overall)

data_size_train = 400
data_size_test = 400
train = pd.DataFrame()
train['path'] = X_train#[:data_size_train] + X_train[-data_size_train:]
train['label'] = y_train#[:data_size_train] + y_train[-data_size_train:]
test = pd.DataFrame()
test['path'] = X_test#[:data_size_test] + X_test[-data_size_test:]
test['label'] = y_test#[:data_size_test] + y_test[-data_size_test:]
train['label'] = train['label'].astype('str')
test['label'] = test['label'].astype('str')

test_datagen = ImageDataGenerator(rescale=1./255, fill_mode = 'nearest')
train_datagen = ImageDataGenerator(rescale=1./255, fill_mode = 'nearest', horizontal_flip=True, zoom_range=0.15, width_shift_range=0.15)

train_gen = train_datagen.flow_from_dataframe(
    train, directory=None, x_col='path',
    y_col='label', target_size=(img_rows, img_cols),
    color_mode='rgb',
    classes=None,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    interpolation='nearest',
    drop_duplicates=True)

val_gen = test_datagen.flow_from_dataframe(
    test, directory=None, x_col='path',
    y_col='label', target_size=(img_rows, img_cols),
    color_mode='rgb',
    classes=None,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    interpolation='nearest',
    drop_duplicates=True, shuffle=False)


model = DenseNet121(weights = "imagenet", include_top=False, input_shape = (img_rows, img_cols, 3))
x = model.output
x = Flatten()(x)
#x = Dense(100, activation="elu", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x)
#x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
#x = Dense(1024, activation="elu", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x)
#x = BatchNormalization()(x)
# x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x)
model_final = Model(input = model.input, output = predictions)
model_final = ModelMGPU(model_final, 2)
model_final.compile(loss = "binary_crossentropy", optimizer = Adam(lr=0.0000001, decay=0.0001), metrics=['accuracy',auc,f1, recall, minC])


checkpoint = ModelCheckpoint(weight_path, monitor='val_minC', verbose=2,
                             save_best_only=True, mode='min', save_weights_only=False)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
                                   verbose=2, mode='min', min_lr=0.0000001)

lrate = LearningRateScheduler(exp_decay)
callbacks_list = [checkpoint, reduceLROnPlat]

loss_history = [model_final.fit_generator(my_gen(train_gen),
                             steps_per_epoch = len(train)/BATCH_SIZE,
                             epochs = number_epochs,
                             validation_data = my_gen(val_gen),
                             validation_steps = len(test)/BATCH_SIZE,
                             callbacks = callbacks_list,
                             verbose=2)]

show_loss(loss_history)
