import json
import numpy as np
import tensorflow as tf
from PIL import Image

# ==========================================
# FILE PATHS
# ==========================================
MODEL_PATH = "plant_disease_resnet50.keras"
CLASS_NAMES_PATH = "class_names.json"
TREATMENTS_PATH = "treatments_all_plant_diseases.json"


# ==========================================
# LOAD MODEL
# ==========================================
print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# LOAD CLASS NAMES
# ==========================================
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

index_to_class = {int(v): k for k, v in class_names.items()}


# ==========================================
# LOAD TREATMENTS
# ==========================================
with open(TREATMENTS_PATH, "r", encoding="utf-8") as f:
    treatments = json.load(f)

print("Treatment entries:", len(treatments))


# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_disease(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index]) * 100

    disease = index_to_class[predicted_index]

    return disease, confidence


# ==========================================
# DISPLAY LIST
# ==========================================
def print_list(items):

    if not items:
        print("• Not available")
        return

    for item in items:
        print("•", item)


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    image_path = input("Enter image path: ").strip()

    try:

        disease, confidence = predict_disease(image_path)

        print("\n")
        print("=" * 50)
        print("       🌱 PLANT DISEASE PREDICTION")
        print("=" * 50)

        print(f"\nDisease    : {disease}")
        print(f"Confidence : {confidence:.2f}%")

        treatment = treatments.get(disease)

        if treatment:

            # QUICK EXPLANATION
            print("\n" + "-" * 50)
            print("📋 QUICK EXPLANATION")
            print("-" * 50)
            print(treatment.get(
                "quick_summary",
                "Not available"
            ))

            # SYMPTOMS
            print("\n" + "-" * 50)
            print("🔍 SYMPTOMS")
            print("-" * 50)
            print_list(treatment.get("symptoms", []))

            # IMMEDIATE ACTION
            print("\n" + "-" * 50)
            print("🚨 WHAT TO DO FIRST")
            print("-" * 50)
            print_list(treatment.get("immediate_action", []))

            # FUNGICIDE / CONTROL
            print("\n" + "-" * 50)
            print("🧪 FUNGICIDE / CONTROL")
            print("-" * 50)
            print(treatment.get(
                "fungicide_or_control",
                "Not available"
            ))

            # HOME TREATMENT
            print("\n" + "-" * 50)
            print("🏡 HOME TREATMENT")
            print("-" * 50)
            print_list(treatment.get("home_treatment", []))

            # PREVENTION
            print("\n" + "-" * 50)
            print("🛡️ PREVENTION")
            print("-" * 50)
            print_list(treatment.get("prevention", []))

            # WARNING
            print("\n" + "-" * 50)
            print("⚠️ IMPORTANT")
            print("-" * 50)
            print(treatment.get(
                "important_warning",
                "Not available"
            ))

        else:

            print("\n⚠️ Treatment information not found.")

        print("\n" + "=" * 50)

    except FileNotFoundError:
        print("\n❌ Image file not found.")
        print("Check the image path.")

    except Exception as e:
        print("\n❌ Error:", e)