import numpy as np
import os
import cv2
import dlib
import glob

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
face_detector = dlib.get_frontal_face_detector()
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]


def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom


def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)


def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6


def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)


def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)


def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))


def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M


def crop_image(image, det):
    left, top, right, bottom = rect_to_tuple(det)
    return image[top:bottom, left:right]


def process(folder_in, folder_out):
    files = os.listdir(folder_in)
    for file in files:
        vidcap = cv2.VideoCapture(folder_in + file)
        success,image = vidcap.read()
        count = 0
        while success:
            success,image = vidcap.read()
            count += 1
            cv2.imwrite(folder_out + str(count) + '_' + file.split('.')[0] + '.jpg', image)


def process_file(file, folder_out):
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        count += 1
        cv2.imwrite(folder_out + str(count) + '_' + file.split('/')[-1].split('.')[0] + '.jpg', image)


def get_head(img):
    det = detector(img, 1)[0]
    height, width = img.shape[:2]

    lr = det.right() - det.left()
    tb = det.bottom() - det.top()
    det = dlib.rectangle(left=max(0, det.left() - int(1.5 * lr)),
                         top=max(0, det.top() - int(1 * tb)),
                         right=min(width, det.right() + int(1.5 * lr)),
                         bottom=min(height, det.bottom() + int(1.25 * tb)))
    cropped = crop_image(img, det)
    return cropped


def process_datasets(dataset_folders, input_foder, result_folder, frame_number=20):
    count = 0
    for folder in dataset_folders:
        for filename in glob.iglob(input_foder + folder + '/**/*.avi',
                                   recursive=True):
            print(filename)
            if not filename.split('/')[-2] in ['real', 'printed-color', 'printed', 'printed-color-cut', 'replay']:
                continue
            if not os.path.isdir(result_folder + filename.split('/')[-3]):
                os.mkdir(result_folder + filename.split('/')[-3])
            vidcap = cv2.VideoCapture(filename)
            amount_of_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            for i in range(0, frame_number):
                n_frame = int(
                    (amount_of_frames - 0.15 * amount_of_frames) / frame_number * i + amount_of_frames * 0.075)
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
                res, frame = vidcap.read()
                if res:
                    if not os.path.isdir(result_folder + filename.split('/')[-3] + '/' + filename.split('/')[-2] + '/'):
                        os.mkdir(result_folder + filename.split('/')[-3] + '/' + filename.split('/')[-2] + '/')
                    if not os.path.exists(
                            result_folder + filename.split('/')[-3] + '/' + filename.split('/')[-2] + '/' + str(
                                    count) + '_' + filename.split('/')[-1].split('.')[0] + '.png'):
                        try:
                            # frame = get_head(frame)
                            cv2.imwrite(
                                result_folder + filename.split('/')[-3] + '/' + filename.split('/')[-2] + '/' + str(
                                    count) + '_' + filename.split('/')[-1].split('.')[0] + '.png', frame)
                        except  Exception:
                            print('no head')
                    count = count + 1


folders = ['data_by_persons_d11', 'data_by_persons_d10', 'data_by_persons_d2', 'data_by_persons_d8', 'data_by_persons_d12', 'data_by_persons_d9', 'data_by_persons_d5']
result_folder = '/home/neuralbee/workspace/anti_spoof_detection/our_data/overall_test/'
input_foder = '/home/neuralbee/workspace/anti_spoof_detection/our_data/'

process_datasets(dataset_folders=folders,
                 input_foder=input_foder,
                 result_folder=result_folder)
