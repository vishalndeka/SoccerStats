import os
from inference import get_model
import supervision as sv
import cv2

def install_dependencies():
    # works in a CUDA enabled GPU environment
    os.system("pip install -q gdown inference-gpu")
    os.system("pip install -q onnxruntime-gpu==1.18.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/")
    os.system("pip install -q git+https://github.com/roboflow/supervision.git")

def download_vids()->None:
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
    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']), thickness=2)
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']), text_color=sv.Color.from_hex('#000000'))
    return box_annotator, label_annotator

def main():
    # download_vids() # done
    os.environ['ONNXRUNTIME_EXECUTION_PROVIDERS'] = "[CUDAExecutionProvider]"
    SOURCE_VIDEO_PATH = 'vids\\0bfacc_0.mp4'
    # frame generator for looping over frames
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, start=300)
    frame = next(frame_generator)

    player_detection_model = get_roboflow_model()
    result = player_detection_model.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)
    labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(detections['class_name'], detections.confidence)]
    box_annotator, label_annotator = supervision_utilities()
    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    sv.plot_image(annotated_frame)
    cv2.imwrite("frame", frame)

if __name__=='__main__':
    main()