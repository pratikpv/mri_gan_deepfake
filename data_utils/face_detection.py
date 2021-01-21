from collections import OrderedDict
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from utils import *
from data_utils.utils import *
import json
import multiprocessing
from tqdm import tqdm

def get_face_detector_model(name='default'):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # default is mtcnn model from facenet_pytorch
    if name == 'default':
        name = 'mtcnn'

    if name == 'mtcnn':
        detector = MTCNN(margin=0, thresholds=[0.85, 0.95, 0.95], device=device)
    else:
        raise Exception("Unknown face detector model.")

    return detector


def locate_face_in_videofile(input_filepath=None, outfile_filepath=None):
    capture = cv2.VideoCapture(input_filepath)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    org_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    org_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    org_res = (org_width, org_height)
    org_fps = int(capture.get(cv2.CAP_PROP_FPS))

    detector = get_face_detector_model()
    frames = list()
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        face_box = list(detector.detect(frame, landmarks=False))[0]
        if face_box is not None:
            for f in range(len(face_box)):
                fc = list(face_box[f])
                cv2.rectangle(frame, (fc[0], fc[1]), (fc[2], fc[3]), (0, 255, 0), 4)
        frames.append(frame)
    create_video_from_images(frames, outfile_filepath, fps=org_fps, res=org_res)

def extract_landmarks_from_video(input_videofile, out_dir, batch_size=32, detector=None, overwrite=False):
    id = os.path.splitext(os.path.basename(input_videofile))[0]
    out_file = os.path.join(out_dir, "{}.json".format(id))

    if not overwrite and os.path.isfile(out_file):
        return

    capture = cv2.VideoCapture(input_videofile)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if detector is None:
        detector = get_face_detector_model()

    frames_dict = OrderedDict()
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames_dict[i] = frame

    result = OrderedDict()
    batches = list()
    frames = list(frames_dict.values())
    num_frames_detected = len(frames)
    for i in range(0, num_frames_detected, batch_size):
        end = i + batch_size
        if end > num_frames_detected:
            end = num_frames_detected
        batches.append((list(range(i, end)), frames[i:end]))

    for j, frames_list in enumerate(batches):
        frame_indices, frame_items = frames_list
        batch_boxes, prob, keypoints = detector.detect(frame_items, landmarks=True)
        batch_boxes = [b.tolist() if b is not None else None for b in batch_boxes]
        keypoints = [k.tolist() if k is not None else None for k in keypoints]

        result.update({i: b for i, b in zip(frame_indices, zip(batch_boxes, keypoints))})

    with open(out_file, "w") as f:
        json.dump(result, f)


def extract_landmarks_from_video_batch(input_filepath_list, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with multiprocessing.Pool(3) as pool:
        jobs = []
        results = []
        for input_filepath in input_filepath_list:
            jobs.append(pool.apply_async(extract_landmarks_from_video,
                                         (input_filepath, out_dir,),
                                         )
                        )

        for job in tqdm(jobs, desc="Extracting landmarks"):
            results.append(job.get())


def draw_landmarks_on_video(in_videofile, out_videofile, landmarks_file):
    capture = cv2.VideoCapture(in_videofile)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    org_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    org_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    org_res = (org_width, org_height)
    org_fps = int(capture.get(cv2.CAP_PROP_FPS))
    with open(landmarks_file, 'r') as jf:
        face_box_dict = json.load(jf)
    frames = list()
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        if str(i) not in face_box_dict.keys():
            continue
        face_box = face_box_dict[str(i)]
        if face_box is not None:
            faces = face_box[0]
            lm = face_box[1]
            for f in range(len(faces)):
                fc = list(map(int, faces[f]))
                cv2.rectangle(frame, (fc[0], fc[1]), (fc[2], fc[3]), (0, 255, 0), 4)
                keypoints = lm[f]
                cv2.circle(frame, tuple(map(int, keypoints[0])), 2, (0, 155, 255), 2)
                cv2.circle(frame, tuple(map(int, keypoints[1])), 2, (0, 155, 255), 2)
                cv2.circle(frame, tuple(map(int, keypoints[2])), 2, (0, 155, 255), 2)
                cv2.circle(frame, tuple(map(int, keypoints[3])), 2, (0, 155, 255), 2)
                cv2.circle(frame, tuple(map(int, keypoints[4])), 2, (0, 155, 255), 2)

        frames.append(frame)
    create_video_from_images(frames, out_videofile, fps=org_fps, res=org_res)


def crop_faces_from_video(in_videofile, landmarks_path, crop_faces_out_dir, overwrite=False, frame_hops=10, buf=0.10):
    id = os.path.splitext(os.path.basename(in_videofile))[0]
    json_file = os.path.join(landmarks_path, id + '.json')
    out_dir = os.path.join(crop_faces_out_dir, id)
    if not os.path.isfile(json_file):
        return
    if not overwrite and os.path.isdir(out_dir):
        return

    try:
        with open(json_file, 'r') as jf:
            face_box_dict = json.load(jf)
    except Exception as e:
        print(f'failed to parse {json_file}')
        print(e)
        raise e

    os.makedirs(out_dir, exist_ok=True)
    capture = cv2.VideoCapture(in_videofile)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frames_num):
        capture.grab()
        if i % frame_hops != 0:
            continue
        success, frame = capture.retrieve()
        if not success or str(i) not in face_box_dict:
            continue

        crops = []
        bboxes = face_box_dict[str(i)][0]
        if bboxes is None:
            continue
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = int(h * buf)
            p_w = int(w * buf)
            crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            crops.append(crop)

        for j, crop in enumerate(crops):
            cv2.imwrite(os.path.join(out_dir, "{}_{}.png".format(i, j)), crop)


def crop_faces_from_video_batch(input_filepath_list, landmarks_path, crop_faces_out_dir):
    os.makedirs(crop_faces_out_dir, exist_ok=True)
    with multiprocessing.Pool(3) as pool:
        jobs = []
        results = []
        for input_filepath in input_filepath_list:
            jobs.append(pool.apply_async(crop_faces_from_video,
                                         (input_filepath, landmarks_path, crop_faces_out_dir,),
                                         )
                        )

        for job in tqdm(jobs, desc="Cropping faces"):
            results.append(job.get())


def extract_landmarks_for_datasets():
    #
    # Celeb-V2 dataset
    #

    landmarks_path = ConfigParser.getInstance().get_celeb_df_v2_landmarks_path()

    print(f'Extracting landmarks from Celeb-df-v2 real data')
    data_path = ConfigParser.getInstance().get_celeb_df_v2_real_path()
    input_filepath_list = glob(data_path + '/*')
    extract_landmarks_from_video_batch(input_filepath_list, landmarks_path)

    print(f'Extracting landmarks from Celeb-df-v2 fake data')
    data_path = ConfigParser.getInstance().get_celeb_df_v2_fake_path()
    input_filepath_list = glob(data_path + '/*')
    extract_landmarks_from_video_batch(input_filepath_list, landmarks_path)

    #
    # DFDC dataset
    #
    print(f'Extracting landmarks from DFDC train data')
    data_path_root = ConfigParser.getInstance().get_dfdc_train_data_path()
    input_filepath_list = get_dfdc_training_video_filepaths(data_path_root)
    landmarks_path = ConfigParser.getInstance().get_dfdc_landmarks_train_path()
    extract_landmarks_from_video_batch(input_filepath_list, landmarks_path)

    print(f'Extracting landmarks from DFDC valid data')
    data_path = ConfigParser.getInstance().get_dfdc_valid_data_path()
    input_filepath_list = glob(data_path + '/*')
    landmarks_path = ConfigParser.getInstance().get_dfdc_landmarks_valid_path()
    extract_landmarks_from_video_batch(input_filepath_list, landmarks_path)

    print(f'Extracting landmarks from DFDC test data')
    data_path = ConfigParser.getInstance().get_dfdc_test_data_path()
    input_filepath_list = glob(data_path + '/*')
    landmarks_path = ConfigParser.getInstance().get_dfdc_landmarks_test_path()
    extract_landmarks_from_video_batch(input_filepath_list, landmarks_path)


def crop_faces_for_datasets():
    #
    # Celeb-df-v2
    #

    landmarks_path = ConfigParser.getInstance().get_celeb_df_v2_landmarks_path()
    crops_path = ConfigParser.getInstance().get_celeb_df_v2_crops_train_path()

    print(f'Cropping faces from Celeb-df-v2 real data')
    data_path = ConfigParser.getInstance().get_celeb_df_v2_real_path()
    input_filepath_list = glob(data_path + '/*')
    crop_faces_from_video_batch(input_filepath_list, landmarks_path, crops_path)

    print(f'Cropping faces from Celeb-df-v2 fake data')
    data_path = ConfigParser.getInstance().get_celeb_df_v2_fake_path()
    input_filepath_list = glob(data_path + '/*')
    crop_faces_from_video_batch(input_filepath_list, landmarks_path, crops_path)

    #
    # DFDC dataset
    #
    print(f'Cropping faces from DFDC train data')
    data_path_root = ConfigParser.getInstance().get_dfdc_train_data_path()
    input_filepath_list = get_dfdc_training_video_filepaths(data_path_root)
    landmarks_path = ConfigParser.getInstance().get_dfdc_landmarks_train_path()
    crops_path = ConfigParser.getInstance().get_dfdc_crops_train_path()
    crop_faces_from_video_batch(input_filepath_list, landmarks_path, crops_path)

    print(f'Cropping faces from DFDC valid data')
    data_path = ConfigParser.getInstance().get_dfdc_valid_data_path()
    input_filepath_list = glob(data_path + '/*')
    landmarks_path = ConfigParser.getInstance().get_dfdc_landmarks_valid_path()
    crops_path = ConfigParser.getInstance().get_dfdc_crops_valid_path()
    crop_faces_from_video_batch(input_filepath_list, landmarks_path, crops_path)

    print(f'Cropping faces from DFDC test data')
    data_path = ConfigParser.getInstance().get_dfdc_test_data_path()
    input_filepath_list = glob(data_path + '/*')
    landmarks_path = ConfigParser.getInstance().get_dfdc_landmarks_test_path()
    crops_path = ConfigParser.getInstance().get_dfdc_crops_test_path()
    crop_faces_from_video_batch(input_filepath_list, landmarks_path, crops_path)
