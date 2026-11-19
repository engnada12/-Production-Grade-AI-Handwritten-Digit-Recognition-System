"""
Handwritten Digit Recognizer - Model Handler & Preprocessing Engine
Ensures local directory structures are explicitly created and populated:
- evaluation/
- joblib_model/
- plots/
- saved_model_directory_v8/
"""

import os
import io
import time
import base64
import numpy as np
from PIL import Image
import mlflow
from datetime import datetime
from pathlib import Path

# 1. إعداد خادم وتتبع MLflow على المنفذ 5001
mlflow.set_tracking_uri("http://localhost:5002")
EXPERIMENT_NAME = "handwritten_digit_inference_experiments"
mlflow.set_experiment(EXPERIMENT_NAME)

# 2. تحديد مجلد المشروع الأساسي وضمان إنشاء المجلدات الأربعة فوراً
BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "evaluation"
JOBLIB_DIR = BASE_DIR / "joblib_model"
PLOTS_DIR = BASE_DIR / "plots"
SAVED_MODEL_DIR = BASE_DIR / "saved_model_directory_v8"

for folder in [EVAL_DIR, JOBLIB_DIR, PLOTS_DIR, SAVED_MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Global references for cached models
_MODEL_INSTANCE = None
_MODEL_TYPE = None


def get_model_path():
    """Detects available trained models in priority order and populates target directories."""
    models_dir = BASE_DIR / "models"
    
    # أولوية 1: البحث في saved_model_directory_v8 للـ CNN
    v8_keras = SAVED_MODEL_DIR / "digit_cnn_model.keras"
    if v8_keras.exists():
        return str(v8_keras), "CNN"
        
    # إذا كان النموذج موجوداً في المجلد القديم، نقوم بنسخه/حفظه إلى saved_model_directory_v8 ليظهر هناك
    old_cnn = models_dir / "digit_cnn_model.keras"
    if old_cnn.exists():
        import tensorflow as tf
        temp_model = tf.keras.models.load_model(str(old_cnn))
        temp_model.save(str(v8_keras))
        print(f"[ModelHandler] Migrated CNN model to: {v8_keras}")
        return str(v8_keras), "CNN"
        
    # أولوية 2: البحث في joblib_model للـ Random Forest
    rf_path = JOBLIB_DIR / "mnist_random_forest_model.pkl"
    if rf_path.exists():
        return str(rf_path), "RandomForest"
        
    parent_rf = BASE_DIR.parent / "mnist_random_forest_model.pkl"
    if parent_rf.exists():
        import joblib
        temp_rf = joblib.load(str(parent_rf))
        joblib.dump(temp_rf, str(rf_path))
        print(f"[ModelHandler] Migrated RF model to: {rf_path}")
        return str(rf_path), "RandomForest"
        
    return None, None


def load_model():
    """Loads and caches the model in memory."""
    global _MODEL_INSTANCE, _MODEL_TYPE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE, _MODEL_TYPE
        
    path, mtype = get_model_path()
    if not path:
        raise FileNotFoundError("No trained model found!")
        
    print(f"[ModelHandler] Loading {mtype} model from: {path}")
    if mtype == "CNN":
        import tensorflow as tf
        _MODEL_INSTANCE = tf.keras.models.load_model(path)
        _MODEL_TYPE = "CNN"
        _MODEL_INSTANCE(np.zeros((1, 28, 28, 1), dtype=np.float32), training=False)
    else:
        import joblib
        _MODEL_INSTANCE = joblib.load(path)
        _MODEL_TYPE = "RandomForest"
        
    print(f"[ModelHandler] {mtype} model loaded successfully.")
    return _MODEL_INSTANCE, _MODEL_TYPE


def preprocess_image(image_input):
    """Preprocessing pipeline and canvas preparation."""
    if isinstance(image_input, str):
        if os.path.isfile(image_input):
            img = Image.open(image_input).convert("L")
        else:
            if "," in image_input:
                image_input = image_input.split(",", 1)[1]
            img_bytes = base64.b64decode(image_input)
            img = Image.open(io.BytesIO(img_bytes)).convert("L")
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
    img_arr = np.array(img, dtype=np.float32)
    h, w = img_arr.shape
    border_pixels = np.concatenate([img_arr[0, :], img_arr[-1, :], img_arr[:, 0], img_arr[:, -1]])
    
    if np.median(border_pixels) > 127.0:
        img_arr = 255.0 - img_arr
        
    bg_level = np.percentile(img_arr, 15)
    img_arr = np.clip(img_arr - bg_level, 0, 255)
    
    max_val = np.max(img_arr)
    if max_val > 10:
        img_arr = (img_arr / max_val) * 255.0
        
    binary_mask = img_arr > 35
    coords = np.argwhere(binary_mask)
    
    if len(coords) < 10:
        canvas = Image.new("L", (28, 28), 0)
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        pad = 2
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        y_max = min(h - 1, y_max + pad)
        x_max = min(w - 1, x_max + pad)
        
        cropped = Image.fromarray(img_arr[y_min:y_max+1, x_min:x_max+1].astype(np.uint8))
        cw, ch = cropped.size
        if cw > ch:
            new_w = 20
            new_h = max(1, int(round(ch * 20.0 / cw)))
        else:
            new_h = 20
            new_w = max(1, int(round(cw * 20.0 / ch)))
            
        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(resized, ((28 - new_w) // 2, (28 - new_h) // 2))
        
    final_arr = np.array(canvas, dtype=np.float32) / 255.0
    flat_array = final_arr.reshape(1, 784)
    spatial_array = final_arr.reshape(1, 28, 28, 1)
    
    buffer = io.BytesIO()
    preview_img = canvas.resize((112, 112), Image.Resampling.NEAREST)
    preview_img.save(buffer, format="PNG")
    preview_base64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return flat_array, spatial_array, preview_base64, canvas


def predict_digit(image_input, target_digits_only=True):
    """Inference engine with structured file storage and MLflow tracking."""
    start_time = time.perf_counter()
    
    model, model_type = load_model()
    flat_arr, spatial_arr, preview_b64, canvas = preprocess_image(image_input)
    
    prep_time_ms = (time.perf_counter() - start_time) * 1000.0
    infer_start = time.perf_counter()
    
    if model_type == "CNN":
        raw_probs = model(spatial_arr, training=False)[0].numpy()
    else:
        raw_probs = model.predict_proba(flat_arr)[0]
        
    infer_time_ms = (time.perf_counter() - infer_start) * 1000.0
    total_time_ms = (time.perf_counter() - start_time) * 1000.0
    
    full_probs = {int(i): float(raw_probs[i]) for i in range(len(raw_probs))}
    
    if target_digits_only:
        digits_1_9 = [d for d in range(1, 10) if d in full_probs]
        sum_1_9 = sum(full_probs[d] for d in digits_1_9)
        normalized_1_9 = {d: round(full_probs[d] / sum_1_9, 4) for d in digits_1_9} if sum_1_9 > 1e-6 else {d: round(1.0 / 9.0, 4) for d in digits_1_9}
        best_digit = max(normalized_1_9, key=normalized_1_9.get)
        confidence = normalized_1_9[best_digit]
        prob_dist = normalized_1_9
    else:
        best_digit = int(np.argmax(raw_probs))
        confidence = round(float(raw_probs[best_digit]), 4)
        prob_dist = {d: round(float(p), 4) for d, p in full_probs.items()}
        
    top_3 = sorted(
        [{"digit": d, "confidence": prob_dist[d]} for d in prob_dist],
        key=lambda x: x["confidence"],
        reverse=True
    )[:3]
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # حفظ المعاينات والتقارير في مجلداتها
    plot_path = PLOTS_DIR / f"preprocessed_digit_{timestamp}.png"
    canvas.resize((112, 112), Image.Resampling.NEAREST).save(plot_path)
    
    eval_path = EVAL_DIR / f"metrics_{timestamp}.txt"
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {timestamp}\nModel Type: {model_type}\nPredicted Digit: {best_digit}\nConfidence: {confidence}\nLatency (ms): {total_time_ms}\n")

    # تسجيل الـ Run في MLflow
    run_name = f"inference_{model_type}_{timestamp}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"model_family": model_type, "task": "digit_recognition"})
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("predicted_digit", float(best_digit))
        mlflow.log_metric("confidence", float(confidence))
        mlflow.log_metric("total_latency_ms", float(total_time_ms))
        
        mlflow.log_artifact(str(plot_path), artifact_path="plots")
        mlflow.log_artifact(str(eval_path), artifact_path="evaluation")

    return {
        "status": "success",
        "prediction": int(best_digit),
        "confidence": float(confidence),
        "probabilities": prob_dist,
        "top_3": top_3,
        "model_type": model_type,
        "timing": {
            "preprocess_ms": round(prep_time_ms, 2),
            "inference_ms": round(infer_time_ms, 2),
            "total_latency_ms": round(total_time_ms, 2)
        },
        "preprocessed_preview": preview_b64
    }


if __name__ == "__main__":
    print("--- Testing Model Handler & Structured Folders ---")
    test_img = Image.new("L", (28, 28), color=255)
    try:
        result = predict_digit(test_img)
        print("Prediction Result:", result)
        print("All folders and model files are successfully structured!")
    except Exception as e:
        print("Error:", e)