"""

app.py



Flask backend for the AI-IoT crop disease detection web application.



Current pipeline:

Image -> crop validation -> leaf validation -> crop-specific CNN

-> severity -> treatment knowledge base -> rule-based decision -> JSON

"""



from __future__ import annotations



import uuid

from pathlib import Path



import joblib
import pandas as pd

import numpy as np

from flask import Flask, jsonify, request, render_template, send_from_directory

from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from flask import render_template, session, redirect, url_for


from core.potato_cnn import predict_potato_disease as predict_potato_cnn

from core.tomato_cnn import predict_tomato_disease as predict_tomato_cnn

from core.image_analyzer import analyze_image

from core.severity import estimate_severity

from core.knowledge_base import lookup_treatment

from core.decision_engine import decide





# ============================================================

# APP SETUP

# ============================================================



app = Flask(__name__)



BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"

MODEL_DIR = BASE_DIR / "models"

# ============================================================
# SIMULATED IoT SENSOR DATA
# ============================================================

IOT_DATA_PATH = BASE_DIR / "data" / "Smart_Farming_Crop_Yield_2024.csv"
iot_data = None
iot_index = 0

# Current displayed sensor state. We smooth the historical Kaggle rows
# so the demo behaves like a plausible sensor stream rather than jumping
# between unrelated farm records every few seconds.
iot_current = {
    "temperature_c": None,
    "humidity_pct": None,
    "soil_moisture_pct": None,
}

if IOT_DATA_PATH.exists():
    try:
        iot_data = pd.read_csv(IOT_DATA_PATH)
        required_iot_columns = {
            "soil_moisture_%",
            "temperature_C",
            "humidity_%",
        }
        if not required_iot_columns.issubset(iot_data.columns):
            print("WARNING: IoT CSV is missing required sensor columns.")
            iot_data = None
        else:
            # Start the simulated stream from the first valid dataset row.
            first_row = iot_data.iloc[0]
            iot_current["temperature_c"] = float(first_row["temperature_C"])
            iot_current["humidity_pct"] = float(first_row["humidity_%"])
            iot_current["soil_moisture_pct"] = float(first_row["soil_moisture_%"])
            print(f"IoT sensor replay loaded: {len(iot_data)} rows.")
    except Exception as exc:
        print("WARNING: Could not load IoT sensor data:", exc)
else:
    print("WARNING: IoT sensor CSV not found.")




UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_DIR.mkdir(exist_ok=True)



app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024





# ============================================================

# SETTINGS

# ============================================================



ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}



ALLOWED_CROPS = {

    "potato",

    "tomato",

}


#load  leaf validator

# Leaf vs non-leaf validator
LEAF_VALIDATOR_PATH = MODEL_DIR / "leaf_validator.keras"
leaf_validator = None

if LEAF_VALIDATOR_PATH.exists():
    try:
        leaf_validator = tf.keras.models.load_model(
            LEAF_VALIDATOR_PATH,
            compile=False,
        )
        print("Leaf validator loaded successfully.")
    except Exception as exc:
        print("WARNING: Could not load leaf validator:", exc)
else:
    print("WARNING: Leaf validator model not found:", LEAF_VALIDATOR_PATH)

# Model probability above this value means NON-LEAF.
# Initial threshold; validate it against more real images.
NON_LEAF_THRESHOLD = 0.95




# ============================================================

# LOAD CROP CLASSIFIER

# ============================================================

CROP_MODEL_PATH = MODEL_DIR / "crop_classifier_cnn.keras"

crop_model = None

if CROP_MODEL_PATH.exists():
    try:
        crop_model = tf.keras.models.load_model(
            CROP_MODEL_PATH,
            compile=False
        )
        print("Crop classification CNN loaded successfully.")
    except Exception as exc:
        print("WARNING: Could not load crop CNN:", exc)
else:
    print("WARNING: Crop CNN model not found:", CROP_MODEL_PATH)






# ============================================================

# MODEL STATUS

# ============================================================



print("Potato MobileNetV2 CNN loaded through core.potato_cnn.")

print("Tomato full-image MobileNetV2 CNN loaded through core.tomato_cnn.")





# ============================================================

# HELPER FUNCTIONS

# ============================================================



def allowed_file(filename: str) -> bool:

    return (

        "." in filename

        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

    )





def parse_float(value, default=None):

    if value is None or value == "":

        return default



    try:

        return float(value)

    except (TypeError, ValueError):

        return default





# ============================================================

# IMAGE -> CROP CLASSIFIER VECTOR

# ============================================================



def image_to_vector(image: Image.Image) -> np.ndarray:

    img = image.convert("RGB")

    img = img.resize((64, 64))

    arr = np.asarray(img, dtype=np.float32) / 255.0

    return arr.flatten()





# ============================================================

# CROP CLASSIFICATION

# ============================================================

def predict_crop(image: Image.Image):
    """Classify an image as Potato or Tomato using MobileNetV2 CNN."""

    if crop_model is None:
        return {
            "crop": "unknown",
            "confidence": 0.0,
            "method": "model-not-loaded",
        }

    try:
        # Match training preprocessing exactly.
        img = image.convert("RGB").resize((224, 224))

        arr = np.asarray(img, dtype=np.float32)

        arr = tf.keras.applications.mobilenet_v2.preprocess_input(
            arr
        )

        batch = np.expand_dims(arr, axis=0)

        # Model output: sigmoid probability for Tomato (class 1).
        probability = float(
            np.asarray(crop_model.predict(batch, verbose=0)).reshape(-1)[0]
        )

        if not np.isfinite(probability):
            raise ValueError("CNN returned an invalid probability.")

        detected_crop = "tomato" if probability >= 0.5 else "potato"

        confidence = (
            probability if detected_crop == "tomato"
            else 1.0 - probability
        )

        return {
            "crop": detected_crop,
            "confidence": round(confidence, 4),
            "method": "mobilenetv2-crop-cnn",
        }

    except Exception as exc:
        print("Crop CNN prediction error:", exc)

        return {
            "crop": "unknown",
            "confidence": 0.0,
            "method": "prediction-error",
        }

  #prediction function 
  


def predict_leaf_status(image: Image.Image):
    if leaf_validator is None:
        raise RuntimeError("Leaf validator model is not loaded.")

    img = image.convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)

    non_leaf_probability = float(
        np.asarray(
            leaf_validator.predict(batch, verbose=0)
        ).reshape(-1)[0]
    )

    if not np.isfinite(non_leaf_probability):
        raise ValueError("Invalid leaf prediction probability.")

    non_leaf_probability = float(
        np.clip(non_leaf_probability, 0.0, 1.0)
    )

    return {
        "is_leaf": non_leaf_probability < NON_LEAF_THRESHOLD,
        "leaf_probability": round(1.0 - non_leaf_probability, 4),
        "non_leaf_probability": round(non_leaf_probability, 4),
        "threshold": NON_LEAF_THRESHOLD,
    }


# ============================================================

# POTATO DISEASE PREDICTION

# ============================================================



def predict_potato_disease(image: Image.Image):

    """

    Potato MobileNetV2 prediction.



    Supports both:

      - dictionary return from core.potato_cnn

      - older (label, confidence) tuple return

    """



    temp_path = UPLOAD_DIR / "_potato_cnn_temp.jpg"



    try:

        image.convert("RGB").save(temp_path)



        result = predict_potato_cnn(temp_path)



        if isinstance(result, dict):

            disease_label = result.get("label", "unknown")

            confidence = result.get("confidence", 0.0)



            return {

                "label": disease_label,

                "confidence": round(float(confidence), 3),

                "method": result.get(

                    "method",

                    "mobilenetv2-cnn"

                ),

                "raw_class": result.get(

                    "raw_class",

                    disease_label

                ),

            }



        if isinstance(result, (tuple, list)) and len(result) >= 2:

            disease_label = result[0]

            confidence = result[1]



            label_mapping = {

                "Potato___Early_blight": "early_blight",

                "Potato___Late_blight": "late_blight",

                "Potato___healthy": "healthy",

            }



            return {

                "label": label_mapping.get(

                    disease_label,

                    str(disease_label).lower()

                ),

                "confidence": round(float(confidence), 3),

                "method": "mobilenetv2-cnn",

                "raw_class": disease_label,

            }



        raise ValueError(

            "Unexpected Potato CNN prediction format."

        )



    finally:

        if temp_path.exists():

            try:

                temp_path.unlink()

            except OSError:

                pass





# ============================================================

# TOMATO DISEASE PREDICTION

# ============================================================



def predict_tomato_disease(image: Image.Image):

    """

    New 4-class Tomato MobileNetV2 model:



      0 -> Bacterial Spot

      1 -> Early Blight

      2 -> Late Blight

      3 -> Healthy

    """



    temp_path = UPLOAD_DIR / "_tomato_cnn_temp.jpg"



    try:

        image.convert("RGB").save(temp_path)



        result = predict_tomato_cnn(temp_path)



        if not isinstance(result, dict):

            raise ValueError(

                "Unexpected Tomato CNN prediction format."

            )



        return result



    finally:

        if temp_path.exists():

            try:

                temp_path.unlink()

            except OSError:

                pass





# ============================================================

# HOME PAGE

# ============================================================



@app.route("/")
def login_page():
    return render_template("login.html")


@app.route("/upload")
def upload_page():
    return render_template("index.html")


@app.route("/result")
def result_page():
    return render_template("result.html")


@app.route("/treatment")
def treatment_page():
    return render_template("treatment.html")





# ============================================================

# UPLOADED IMAGE

# ============================================================



@app.route("/uploads/<path:filename>")

def uploaded_file(filename):

    return send_from_directory(UPLOAD_DIR, filename)





# ============================================================

# HEALTH CHECK

# ============================================================



@app.route("/health", methods=["GET"])

def health():

    return jsonify({

        "status": "ok",

        "crop_classifier_loaded": crop_model is not None,

        "potato_cnn_loaded": True,

        "tomato_cnn_loaded": True,

    })





# ============================================================

# ANALYZE IMAGE

# ============================================================



# ============================================================
# SIMULATED REAL-TIME SENSOR DATA
# ============================================================

@app.route("/sensor-data", methods=["GET"])
def sensor_data():
    """Return a smoothed simulated IoT reading based on Kaggle data.

    The CSV contains historical/simulated farm observations rather than
    consecutive readings from one physical sensor. Therefore, the demo
    moves gradually toward each dataset observation instead of displaying
    the raw next row every five seconds.
    """
    global iot_index, iot_current

    if iot_data is None or iot_data.empty:
        return jsonify({"error": "IoT sensor dataset is not available."}), 503

    row = iot_data.iloc[iot_index]
    current_index = iot_index
    iot_index = (iot_index + 1) % len(iot_data)

    targets = {
        "temperature_c": float(row["temperature_C"]),
        "humidity_pct": float(row["humidity_%"]),
        "soil_moisture_pct": float(row["soil_moisture_%"]),
    }

    # Small per-update movement keeps the 5-second dashboard refresh
    # visually plausible while still following the Kaggle data over time.
    max_change = {
        "temperature_c": 0.25,
        "humidity_pct": 0.50,
        "soil_moisture_pct": 0.10,
    }

    for key, target in targets.items():
        current = iot_current[key]
        if current is None:
            iot_current[key] = target
            continue

        delta = target - current
        if delta > max_change[key]:
            delta = max_change[key]
        elif delta < -max_change[key]:
            delta = -max_change[key]

        iot_current[key] = current + delta

    return jsonify({
        "source": "Kaggle Smart_Farming_Crop_Yield_2024.csv",
        "mode": "simulated_realtime_smoothed_replay",
        "row_index": int(current_index),
        "temperature_c": round(iot_current["temperature_c"], 2),
        "humidity_pct": round(iot_current["humidity_pct"], 2),
        "soil_moisture_pct": round(iot_current["soil_moisture_pct"], 2),
    })


@app.route("/analyze", methods=["POST"])

def analyze():



    # --------------------------------------------------------

    # 1. CHECK IMAGE

    # --------------------------------------------------------



    if "image" not in request.files:

        return jsonify({

            "error": "No image uploaded."

        }), 400



    file = request.files["image"]



    if file.filename == "":

        return jsonify({

            "error": "No image selected."

        }), 400



    if not allowed_file(file.filename):

        return jsonify({

            "error": (

                "Unsupported image format. "

                "Use JPG, PNG or WEBP."

            )

        }), 400





    # --------------------------------------------------------

    # 2. SELECTED CROP

    # --------------------------------------------------------



    crop = request.form.get(

        "crop",

        ""

    ).strip().lower()



    if crop == "other":
     return jsonify({
        "error": "Unsupported crop.",
        "isValidationError": True,
        "message": (
            "This crop is not supported yet. "
            "Please upload a clear Tomato or Potato leaf image "
            "and select the correct crop."
        ),
        "reason": "unsupported_crop",
        "selected_crop": "other",
    }), 400

    if crop not in ALLOWED_CROPS:
      return jsonify({
        "error": "Unsupported crop.",
        "isValidationError": True,
        "message": (
            "Please select Tomato or Potato to continue."
        ),
        "allowed_crops": sorted(ALLOWED_CROPS),
    }), 400





    # --------------------------------------------------------

    # 3. OPTIONAL FIELD CONDITIONS

    # --------------------------------------------------------



    temperature_c = parse_float(

        request.form.get("temperature_c")

    )



    humidity_pct = parse_float(

        request.form.get("humidity_pct")

    )



    soil_moisture_pct = parse_float(

        request.form.get("soil_moisture_pct")

    )





    # --------------------------------------------------------

    # 4. OPEN IMAGE

    # --------------------------------------------------------



    try:

        image = Image.open(

            file.stream

        ).convert("RGB")



    except (

        UnidentifiedImageError,

        OSError,

        Exception,

    ) as exc:

        return jsonify({

            "error": "Invalid image file.",

            "details": str(exc),

        }), 400





    # --------------------------------------------------------

    # 5. SAVE IMAGE

    # --------------------------------------------------------



    extension = Path(

        file.filename

    ).suffix.lower()



    if extension == "":

        extension = ".jpg"



    saved_name = (

        f"{uuid.uuid4().hex}"

        f"{extension}"

    )



    saved_path = UPLOAD_DIR / saved_name



    try:

        image.save(saved_path)

    except Exception as exc:

        return jsonify({

            "error": "Could not save uploaded image.",

            "details": str(exc),

        }), 500





    # --------------------------------------------------------

    # 6. CROP CLASSIFICATION

    # --------------------------------------------------------



    crop_prediction = predict_crop(image)

    detected_crop = crop_prediction["crop"]



    crop_warning = None



    if (

        detected_crop != "unknown"

        and detected_crop != crop

    ):

        crop_warning = (

            f"The crop validation model suggested "

            f"{detected_crop}, but the selected crop "

            f"{crop} will be used for disease analysis."

        )



        print(

            "Crop validation warning:",

            crop_warning

        )





    # --------------------------------------------------------

    # 7. 

    # --------------------------------------------------------

    #

    # Tomato:

    # Do NOT hard-block because the old crop classifier can

    # confuse Tomato leaves with Potato.

    #

    # Potato:

    # Keep the old hard mismatch protection so a clearly

    # Tomato-looking image is not sent to the Potato CNN.

    #



    # if crop == "potato" and (

    #     detected_crop != "unknown"

    #     and detected_crop != crop

    # ):
    

        # return jsonify({

        #     "error": "Crop mismatch.",

        #     "message": (

        #         "You selected potato, but the crop "

        #         f"validation model suggested {detected_crop}."

        #     ),

        #     "selected_crop": crop,

        #     "detected_crop": detected_crop,

        #     "crop_validation": {

        #         "valid": False,

        #         "method": crop_prediction["method"],

        #         "distances": crop_prediction["distances"],

        #     },

        # }), 400

#     if (
#     crop == "potato"
#     and detected_crop != "unknown"
#     and detected_crop != crop
# ):
#      return jsonify({
#         "error": "Crop mismatch",
#         "message": (
#             f"You selected Potato, but the crop "
#             f"classifier detected {detected_crop.title()}. "
#             "Please upload a Potato leaf image."
#         ),
#         "selected_crop": crop,
#         "detected_crop": detected_crop,
#         "crop_validation": {
#             "valid": False,
#             "method": crop_prediction["method"],
#             "distances": crop_prediction.get("distances",{})
#         },
#     }), 400

    # 7. CROP VALIDATION — check both directions
    if (
        detected_crop in ALLOWED_CROPS
        and detected_crop != crop
        and crop_prediction.get("confidence", 0) >= 0.70
    ):
        return jsonify({
            "error": "Crop mismatch",
            "message": (
                f"You selected {crop.title()}, but the "
                f"classifier detected {detected_crop.title()}. "
                f"Please upload a {crop.title()} leaf image."
            ),
            "selected_crop": crop,
            "detected_crop": detected_crop,
            "crop_validation": {
                "valid": False,
                "method": crop_prediction.get("method", "unknown"),
                "confidence": crop_prediction.get("confidence", 0),
                "distances": crop_prediction.get("distances", {}),
            },
        }), 400

    print("SELECTED CROP:", crop)
    print("CROP CLASSIFIER RESULT:", crop_prediction)


    
    # 8. LEAF VALIDATION
    try:
        leaf_result = predict_leaf_status(image)
    except Exception as exc:
        print("Leaf validator error:", exc)
        return jsonify({
            "error": "Leaf validation unavailable.",
            "message": (
                "The image could not be validated. "
                "Please check the leaf validator model."
            ),
        }), 503

    print("LEAF VALIDATOR RESULT:", leaf_result)

    if not leaf_result["is_leaf"]:
        return jsonify({
            "error": "Invalid leaf image.",
            "message": (
                "This image does not appear to contain a "
                "supported crop leaf. Please upload a clear "
                "Potato or Tomato leaf image."
            ),
            "selected_crop": crop,
            "reason": "leaf_validation_failed",
            "leaf_validation": leaf_result,
        }), 400


    # 8. IMAGE / LEAF ANALYSIS

    # -------------------------------------------------------


    analysis = analyze_image(

        image,
        crop

    )





    # --------------------------------------------------------

    # 9. INVALID / NON-CROP IMAGE

    # --------------------------------------------------------



    if analysis.status == "not_crop":

        return jsonify({

            "error": "Invalid crop image.",

            "message": (

                "The uploaded image does not appear to "

                "contain a supported crop leaf. Please "

                "upload a clear crop leaf image."

            ),

            "selected_crop": crop,

            "reason": "leaf_validation_failed",

            "crop_validation": {

                "valid": False,

                "status": "not_crop",

            },

            "image_quality": {

                "leaf_area_ratio":

                    analysis.leaf_area_ratio,

                "green_area_ratio":

                    analysis.green_area_ratio,

            },

        }), 400





    # --------------------------------------------------------

    # 10. DISEASE MODEL

    # --------------------------------------------------------



    potato_prediction = None

    tomato_prediction = None





    # --------------------------------------------------------

    # POTATO -> MobileNetV2 CNN

    # --------------------------------------------------------



    if crop == "potato":



        try:

            potato_prediction = predict_potato_disease(

                image

            )



        except Exception as exc:

            print(

                "Potato CNN error:",

                exc

            )



            return jsonify({

                "error": (

                    "Potato model prediction failed."

                ),

                "details": str(exc),

            }), 500



        analysis.disease_label = (

            potato_prediction["label"]

        )



        analysis.confidence = (

            potato_prediction["confidence"]

        )



        analysis.method = (

            potato_prediction["method"]

        )





    # --------------------------------------------------------

    # TOMATO -> New 4-class MobileNetV2 CNN

    # --------------------------------------------------------



    elif crop == "tomato":



        try:

            tomato_prediction = predict_tomato_disease(

                image

            )



        except Exception as exc:

            print(

                "Tomato CNN error:",

                exc

            )



            return jsonify({

                "error": (

                    "Tomato model prediction failed."

                ),

                "details": str(exc),

            }), 500



        analysis.disease_label = (

            tomato_prediction["label"]

        )



        analysis.confidence = (

            tomato_prediction["confidence"]

        )



        analysis.method = (

            tomato_prediction["method"]

        )





    # --------------------------------------------------------

    # 11. SEVERITY

    # --------------------------------------------------------

    #

    # If the trained CNN identifies the leaf as healthy,

    # do not report CV colour variation as disease severity.

    # This prevents contradictory results such as:

    # "Healthy" + "High severity".

    #



    if analysis.disease_label == "healthy":

        analysis.affected_area_ratio = 0.0



    severity = estimate_severity(

        analysis.affected_area_ratio

    )





    # --------------------------------------------------------

    # 12. KNOWLEDGE BASE

    # --------------------------------------------------------



    treatment = lookup_treatment(

        crop,

        analysis.disease_label

    )





    # --------------------------------------------------------

    # 13. DECISION ENGINE

    # --------------------------------------------------------



    try:

        decision = decide(

            severity,

            analysis.disease_label,

            treatment,

            temperature_c=temperature_c,

            humidity_pct=humidity_pct,

            soil_moisture_pct=soil_moisture_pct,

        )



    except TypeError:

        decision = decide(

            severity,

            analysis.disease_label,

            treatment,

        )





    # --------------------------------------------------------

    # 14. DECISION -> JSON

    # --------------------------------------------------------



    decision_response = {

        "treat": decision.treat,

        "reason": decision.reason,

        "risk_note": decision.risk_note,

        "spray_duration_s": decision.spray_duration_s,

        "spray_volume_ml": decision.spray_volume_ml,

        "recheck_days": decision.recheck_days,
        "leaf_validation": leaf_result,

    }





    # --------------------------------------------------------

    # 15. RESPONSE

    # --------------------------------------------------------



    response = {

        "timestamp": __import__(

            "datetime"

        ).datetime.now(

            __import__("datetime").timezone.utc

        ).isoformat(),



        "image_url": f"/uploads/{saved_name}",



        "crop": crop,



        "crop_validation": {

            "valid": True,

            "selected_crop": crop,

            "detected_crop": detected_crop,

            "warning": crop_warning,

            "method": crop_prediction["method"],

            "distances": crop_prediction.get("distances",{})

        },



        "disease": {

            "label": analysis.disease_label,

            "display_name": (

                analysis.disease_label

                .replace("_", " ")

                .title()

            ),

            "confidence": analysis.confidence,

            "method": analysis.method,

        },



        "image_analysis": {

            "status": analysis.status,

            "leaf_area_ratio":

                analysis.leaf_area_ratio,

            "green_area_ratio":

                analysis.green_area_ratio,

            "affected_area_ratio":

                analysis.affected_area_ratio,

        },



        "severity": {

            "score": severity.score,

            "severity_class":

                severity.severity_class,

            "affected_area_pct": round(

                analysis.affected_area_ratio * 100,

                1

            ),

            "leaf_detected_pct": round(

                analysis.leaf_area_ratio * 100,

                1

            ),

        },



        "treatment": treatment,



        "decision": decision_response,



        "env_used": {

            "temperature_c": temperature_c,

            "humidity_pct": humidity_pct,

            "soil_moisture_pct": soil_moisture_pct,

        },

    }





    # --------------------------------------------------------

    # 16. MODEL DETAILS

    # --------------------------------------------------------



    if potato_prediction is not None:

        response["potato_model"] = {

            "raw_class":

                potato_prediction.get("raw_class"),

            "confidence":

                potato_prediction.get("confidence"),

            "method":

                potato_prediction.get("method"),

        }





    if tomato_prediction is not None:

        response["tomato_model"] = {

            "raw_class":

                tomato_prediction.get("raw_class"),

            "confidence":

                tomato_prediction.get("confidence"),

            "method":

                tomato_prediction.get("method"),

            "model_file":

                tomato_prediction.get("model_file"),

        }





    # --------------------------------------------------------

    # 17. RETURN

    # --------------------------------------------------------



    return jsonify(response)





# ============================================================

# FILE TOO LARGE

# ============================================================



@app.errorhandler(413)

def too_large(error):

    return jsonify({

        "error": (

            "Image is too large. "

            "Maximum size is 8 MB."

        )

    }), 413





# ============================================================

# RUN APPLICATION

# ============================================================



if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )
