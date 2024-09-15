import os
import inference
import supervision as sv
import cv2
from tqdm import tqdm
import torch
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import numpy as np

def install_dependencies()->None:
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
    PLAYER_DETECTION_MODEL = inference.get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    return PLAYER_DETECTION_MODEL

def supervision_utilities():
    # colors for classes: ball, gk, outfield, ref
    # box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(['#ffffff', '#00BFFF', '#FF1493', '#FFD700']), thickness=2)
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']), text_color=sv.Color.from_hex('#000000'), text_position=sv.Position.BOTTOM_CENTER)
    ellipse_annotator = sv.EllipseAnnotator(color=sv.Color.from_hex('#ffffff'), thickness=2)
    triangle_annotator = sv.TriangleAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']), base=20, height=17)
    return ellipse_annotator, triangle_annotator, label_annotator

def full_vid_detection(source_video_path) -> None:
    SOURCE_VIDEO_PATH = source_video_path
    TARGET_VIDEO_PATH = 'ops\\full_obj_detection.mp4'

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

def get_tracker():
    tracker = sv.ByteTrack()
    tracker.reset()
    return tracker

def extract_crops(source_video_path:str, player_detection_model)->list:
    STRIDE = 30
    PLAYER_ID = 2
    frame_generator = sv.get_video_frames_generator(source_video_path, stride = STRIDE)
    crops = []
    for frame in tqdm(frame_generator, desc="skipping frames and colecting crops"):
        result = player_detection_model.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic = True)
        detections = detections[detections.class_id == PLAYER_ID]
        crops += [
            sv.crop_image(frame, xyxy) for xyxy in detections.xyxy
        ]
    
    return crops

def get_siglip():
    # pulling siglip from huggingface and loading into memory
    SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(DEVICE)
    EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
    return EMBEDDINGS_MODEL, EMBEDDINGS_PROCESSOR

def main():
    os.environ['ONNXRUNTIME_EXECUTION_PROVIDERS'] = "[AzureExecutionProvider]"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # download_vids() # done
    # full_vid_detection('vids\\full_obj_detection.mp4')
    
    # obj detection with triangle and ellipse annotators
    SOURCE_VIDEO_PATH = 'vids\\0bfacc_0.mp4'
    TARGET_VIDEO_PATH = 'ops\\0bfacc_0.mp4'
    BALL_ID = 0 # class id for ball class

    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, start = 200)
    frame = next(frame_generator)
    
    # init tracker
    tracker = get_tracker()
    # init obj detection model
    player_detection_model = get_roboflow_model()
    # result = player_detection_model.infer(frame, confidence=0.3)[0]
    # detections = sv.Detections.from_inference(result)
    # ball_detections = detections[detections.class_id == BALL_ID]
    # to expand a bounding box by some pixels:
    # ball_detections.xyxy = sv.pad_boxes(xyxy = ball_detections.xyxy, px=10)
    # all_detections = detections[detections.class_id != BALL_ID]
    # all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    # all_detections.class_id = all_detections.class_id-1 # since class 0 is taken care of by the ellipse annotator
    # all_detections = tracker.update_with_detections(all_detections)

    # labels = [f"{tracker_id}" for tracker_id in all_detections.tracker_id]

    # ellipse_annotator, triangle_annotator, label_annotator = supervision_utilities()
    # # annotation
    # annotated_frame = frame.copy()
    # annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=all_detections)
    # annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=ball_detections)
    # annotated_frame = label_annotator.annotate(annotated_frame, all_detections, labels)
    # sv.plot_image(annotated_frame)

    crops = extract_crops(SOURCE_VIDEO_PATH, player_detection_model)
    # sv.plot_images_grid(crops[:100], grid_size=(10, 10))

    # getting embeddings from crops using siglip
    # sv uses opencv as its engine while siglip uses pillow, so we'll convert crops into a format that is suitable
    crops = [sv.cv2_to_pillow(crop) for crop in crops]
    BATCH_SIZE = 32
    batches = chunked(crops, BATCH_SIZE) # to split pillow crops to batches
    data = []
    EMBEDDINGS_MODEL, EMBEDDINGS_PROCESSOR = get_siglip() # fetches siglip from hugging face, comes with a model and a processor for data pre and post processing, as is the case with huggingface models
    with torch.no_grad():
        for batch in tqdm(batches, desc='embeddings extraction'):
            inputs = EMBEDDINGS_PROCESSOR(images = batch, return_tensors = 'pt').to(DEVICE)
            outputs = EMBEDDINGS_MODEL(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy() # embeddings are present at the last hidden state of the model
            data.append(embeddings)
    data = np.concatenate(data)
    print(data.shape)
    print(len(data))


if __name__=='__main__':
    main()