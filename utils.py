import cv2
import numpy as np
from sklearn.utils import shuffle
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, JpegCompression
)

def test_generator(x, batch_size=64):
    num_samples = len(x)  # x.shape[0]
    while 1:
        for i in range(0, num_samples, batch_size):
            x_data = [preprocess_image(im) for im in x[i:i + batch_size]]
            yield np.array(x_data)


def train_generator(x, y, batch_size=64):
    num_samples = len(x)  # x.shape[0]
    while 1:
        try:
            for i in range(0, num_samples, batch_size):
                x_data = [preprocess_and_aug_image(im) for im in x[i:i + batch_size]]
                y_data = y[i:i + batch_size]

                yield shuffle(np.array(x_data), np.array(y_data))
        except Exception as err:
            xxx = 1
            # print(err)


def val_generator(x, y, batch_size=64):
    num_samples = len(x)  # x.shape[0]
    while 1:
        try:
            for i in range(0, num_samples, batch_size):
                x_data = [preprocess_image(im) for im in x[i:i + batch_size]]
                y_data = y[i:i + batch_size]

                yield shuffle(np.array(x_data), np.array(y_data))
        except Exception as err:
            xxx = 1
            # print(err)


def face_aug(p=.5):
    return Compose([
        HorizontalFlip(p=0.5),
        OneOf([
            IAAAdditiveGaussianNoise(scale=(1, 3)),
            GaussNoise(var_limit=(1, 5)),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(alpha=(0.1, 0.2)),
            IAAEmboss(strength=(0.1, 0.3)),
            RandomContrast(limit=0.1),
            RandomBrightness(limit=0.15),
        ], p=0.3)
    ], p=p)


def weak_aug(p=.5):
    return Compose([
        HorizontalFlip(p=0.5),
        Transpose(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.3),
    ], p=p)


def strong_aug(p=.5):
    return Compose([
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


def augmentation(image):
    aug = strong_aug(p=1)
    image = aug(image=image)['image']
    return image


def preprocess_and_aug_image(image_path, size=224):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (size, size))
    img = augmentation(img)
    img = img / 255.0
    return img


def preprocess_image(image_path, size=224):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (size, size))
    img = img / 255.0
    return img