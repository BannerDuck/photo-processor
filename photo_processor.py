import os
import re
import time
import yaml
import logging
from datetime import datetime
from PIL import Image, ImageFilter
import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ================== Suppress Mediapipe/TensorFlow logs ==================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ================== YAML Environment Variable Support ==================
def env_var_constructor(loader, node):
    """Expands environment variables inside YAML values (e.g. ${VAR:-default})."""
    value = loader.construct_scalar(node)
    pattern = re.compile(r'\${([^:}]+)(:-([^}]*))?}')
    
    def replacer(match):
        env_var = match.group(1)
        default = match.group(3) if match.group(3) else ''
        return os.environ.get(env_var, default)
    
    return pattern.sub(lambda m: replacer(m), value)

yaml.SafeLoader.add_implicit_resolver('!env_var', re.compile(r'\${[^}^{]+}'))
yaml.SafeLoader.add_constructor('!env_var', env_var_constructor)

# ================== Load Config File ==================
CONFIG_FILE = os.environ.get("CONFIG_FILE", "config.yaml")
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

PATHS = config.get("paths", {})

TEMP_DIR = PATHS.get("temp_dir", "/app/temp")
PROCESSED_DIR = PATHS.get("processed_dir", "/app/processed")
PROCESSED_DIR_REMBG = PATHS.get("processed_dir_rembg", "/app/processed_rembg")
ORIGINALS_DIR = PATHS.get("originals_dir", "/app/originals")
LOG_FILE = PATHS.get("log_file", "/app/photo_processor.log")

# ================== Ensure folders exist ==================
for folder in [TEMP_DIR, PROCESSED_DIR, PROCESSED_DIR_REMBG, ORIGINALS_DIR]:
    os.makedirs(folder, exist_ok=True)

# ================== Logger Setup ==================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log(message):
    logging.info(message)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# ================== Mediapipe Setup ==================
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ================== Image Processing Function ==================
def process_image(file_path):
    try:
        log(f"Processing {file_path}...")
        img = cv2.imread(file_path)
        if img is None:
            log(f"❌ Failed to read {file_path}")
            return
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect face landmarks
        results = mp_face_mesh.process(rgb_img)
        if not results.multi_face_landmarks:
            log(f"⚠ No face detected in {file_path}")
            return

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = img.shape

        # Get left and right eye coordinates
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        left_eye_coord = (int(left_eye.x * w), int(left_eye.y * h))
        right_eye_coord = (int(right_eye.x * w), int(right_eye.y * h))

        # Calculate rotation angle
        delta_y = right_eye_coord[1] - left_eye_coord[1]
        delta_x = right_eye_coord[0] - left_eye_coord[0]
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        # Rotate image
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rgb_img = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

        # Face detection for crop
        face_results = mp_face_detection.process(rgb_img)
        if not face_results.detections:
            log(f"⚠ No face detected after rotation in {file_path}")
            return

        bbox = face_results.detections[0].location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)

        head_height = y2 - y1
        y1_new = max(0, y1 - int(head_height / 1.75))
        y2_new = min(h, y2 + head_height // 2)
        x_center = (x1 + x2) // 2
        crop_size = max(x2 - x1, y2_new - y1_new)
        x1_new = max(0, x_center - crop_size // 2)
        x2_new = min(w, x_center + crop_size // 2)

        cropped = rotated[y1_new:y2_new, x1_new:x2_new]

        # Selfie segmentation
        seg_results = mp_selfie_segmentation.process(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        mask = seg_results.segmentation_mask
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask_3c = np.stack((mask,) * 3, axis=-1)
        white_bg = np.ones_like(cropped, dtype=np.uint8) * 255
        blended = (cropped * mask_3c + white_bg * (1 - mask_3c)).astype(np.uint8)

        fname = os.path.basename(file_path)
        processed_path = os.path.join(PROCESSED_DIR, fname)
        Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)).save(processed_path)
        log(f"✔ Saved selfie segmentation image to {processed_path}")

        # Rembg processing
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        no_bg = remove(pil_img)
        alpha = no_bg.split()[3]
        for _ in range(2):
            alpha = alpha.filter(ImageFilter.MinFilter(5))
        white_bg_rembg = Image.new("RGB", no_bg.size, (255, 255, 255))
        white_bg_rembg.paste(no_bg, mask=alpha)
        processed_path_rembg = os.path.join(PROCESSED_DIR_REMBG, fname)
        white_bg_rembg.save(processed_path_rembg)
        log(f"✔ Saved rembg image to {processed_path_rembg}")

        # Move original file
        original_path = os.path.join(ORIGINALS_DIR, fname)
        os.rename(file_path, original_path)
        log(f"↪ Moved original to {original_path}")

    except Exception as e:
        log(f"❌ Error processing {file_path}: {e}")

# ================== Process Existing Images ==================
for fname in os.listdir(TEMP_DIR):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        process_image(os.path.join(TEMP_DIR, fname))

# ================== Watchdog for New Images ==================
class PhotoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith((".jpg", ".jpeg", ".png")):
            time.sleep(1)
            process_image(event.src_path)

observer = Observer()
observer.schedule(PhotoHandler(), path=TEMP_DIR, recursive=False)

try:
    observer.start()
    log("📸 Watching folder for new images... Press Ctrl+C to exit.")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    log("🛑 Keyboard interrupt received. Stopping...")
    observer.stop()
except Exception as e:
    log(f"❌ Observer error: {e}")

observer.join()
log("✅ Stopped watching folder.")
