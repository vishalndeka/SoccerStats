import os
from inference import get_model
import supervision as sv
import cv2
from tqdm import tqdm

def install_dependencies():
    # works in a CUDA enabled GPU environment
    os.system("pip install -q gdown inference-gpu")
    os.system("pip install -q onnxruntime-gpu==1.18.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/")
    os.system("pip install -q git+https://github.com/roboflow/supervision.git")

def download_vids()->None:
    # downloads a few sample videos
    if not os.path.exists(os.path.join(os.getcwd(), 'vids')):
        os.mkdir(os.path.join(os.getcwd(), 'vids'))
    os.chdir(os.path.join(os.getcwd(), 'vids'))
    os.system('gdown -O "0bfacc_0.mp4" "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"')
    os.system('gdown -O "2e57b9_0.mp4" "https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf"')
    os.system('gdown -O "08fd33_0.mp4" "https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-"')
    os.system('gdown -O "573e61_0.mp4" "https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU"')
    os.system('gdown -O "121364_0.mp4" "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"')

def get_roboflow_model():
    ROBOFLOW_API_KEY = "DDHT1BYoK31ZfxKztjET" # keep this private
    PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
    PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    return PLAYER_DETECTION_MODEL

def supervision_utilities():
    # colors for classes: ball, gk, outfield, ref
    # box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(['#ffffff', '#00BFFF', '#FF1493', '#FFD700']), thickness=2)
    # label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#ffffff', '#00BFFF', '#FF1493', '#FFD700']), text_color=sv.Color.from_hex('#000000'))
    ellipse_annotator = sv.EllipseAnnotator(color=sv.Color.from_hex('#ffffff'), thickness=2)
    triangle_annotator = sv.TriangleAnnotator(color=sv.ColorPalette.from_hex(['#ffffff', '#00BFFF', '#FF1493', '#FFD700']), base=20, height=17)
    return ellipse_annotator, triangle_annotator

def full_vid_detection(source_video_path) -> None:
    SOURCE_VIDEO_PATH = source_video_path
    TARGET_VIDEO_PATH = 'ops\\full_obj_detection.mp4'
    BALL_ID = 0 # class id for ball class

    # supervision function that can extract info from video
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info) # supervision utility that helps in saving video

    with video_sink:
        # frame generator for looping over frames
        frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, start=300)
        # frame = next(frame_generator) # inference on the first frame

        player_detection_model = get_roboflow_model()
        # object detection on each frame
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            result = player_detection_model.infer(frame, confidence=0.3)[0]
            detections = sv.Detections.from_inference(result)

            box_annotator, label_annotator = supervision_utilities()
            labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(detections['class_name'], detections.confidence)]
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            video_sink.write_frame(annotated_frame)

def main():
    # os.environ['ONNXRUNTIME_EXECUTION_PROVIDERS'] = "[CUDAExecutionProvider]"
    # download_vids() # done
    # full_vid_detection('vids\\full_obj_detection.mp4')
    
    # obj detection with triangle and ellipse annotators
    SOURCE_VIDEO_PATH = 'vids\\0bfacc_0.mp4'
    TARGET_VIDEO_PATH = 'ops\\0bfacc_0.mp4'
    BALL_ID = 0 # class id for ball class

    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, start = 300)
    frame = next(frame_generator)
    player_detection_model = get_roboflow_model()
    result = player_detection_model.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)
    ball_detections = detections[detections.class_id == BALL_ID]
    all_detections = detections[detections.class_id != BALL_ID]

    ellipse_annotator, triangle_annotator = supervision_utilities()
    annotated_frame = frame.copy()
    annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=all_detections)
    annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=ball_detections)
    
    sv.plot_image(annotated_frame)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()