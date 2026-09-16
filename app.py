import os
import json
import uuid
import re
import difflib
import threading
from datetime import datetime
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session
)
from werkzeug.utils import secure_filename
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "plant_disease_resnet50.keras"
)

TFLITE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "plant_disease_resnet50.tflite"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)

TREATMENT_PATH = os.path.join(
    BASE_DIR,
    "treatments_all_plant_diseases.json"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

DATA_FOLDER = os.path.join(
    BASE_DIR,
    "data"
)

COLLECTION_FILE = os.path.join(
    DATA_FOLDER,
    "upload_records.json"
)

CHAT_FILE = os.path.join(
    DATA_FOLDER,
    "chat_records.json"
)

USERS_FILE = os.path.join(
    DATA_FOLDER,
    "users.json"
)

BEFORE_SAVED_FILE = os.path.join(
    DATA_FOLDER,
    "saved_before_records.json"
)

PLANT_PROFILES_FILE = os.path.join(
    DATA_FOLDER,
    "plant_profiles.json"
)

TREATMENT_RECORDS_FILE = os.path.join(
    DATA_FOLDER,
    "treatment_records.json"
)

CARE_REMINDERS_FILE = os.path.join(
    DATA_FOLDER,
    "care_reminders.json"
)

IMAGE_SIZE = (224, 224)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}


# ============================================================
# SUPPORTED PLANTS
# ============================================================

SUPPORTED_PLANTS = {

    "hibiscus": {
        "display": "🌺 Hibiscus"
    },

    "chilli": {
        "display": "🌶️ Chilli"
    },

    "tomato": {
        "display": "🍅 Tomato"
    }
}


# ============================================================
# PLANT CARE CONTENT
# Loaded from plant_care_data.json so the JSON file is the
# single source of truth for the Plant Care page.
# ============================================================

PLANT_CARE_PATH = os.path.join(
    BASE_DIR,
    "plant_care_data.json"
)


def _build_plant_care_display(raw_data):
    """Convert the JSON care structure into the format used by
    plant_care.html and result.html, without losing the source data.
    """

    display_data = {}

    if not isinstance(raw_data, dict):
        return display_data

    for raw_plant_name, raw_plant in raw_data.items():

        if raw_plant_name.lower() == "metadata":
            continue

        if not isinstance(raw_plant, dict):
            continue

        plant_key = str(raw_plant_name).strip().lower()

        sections = []

        section_order = [
            "watering",
            "sunlight",
            "temperature",
            "soil",
            "fertilizer",
            "pruning"
        ]

        for section_key in section_order:

            section = raw_plant.get(section_key)

            if not isinstance(section, dict):
                continue

            title = section.get("title") or section_key.replace("_", " ").title()
            summary = section.get("summary", "")

            bullets = []

            # The JSON uses both "guidance" and, in some versions,
            # "bullets". Support both without changing the source JSON.
            guidance = section.get("guidance")
            if isinstance(guidance, list):
                bullets.extend(str(x) for x in guidance if x is not None)
            elif guidance:
                bullets.append(str(guidance))

            source_bullets = section.get("bullets")
            if isinstance(source_bullets, list):
                bullets.extend(str(x) for x in source_bullets if x is not None)
            elif source_bullets:
                bullets.append(str(source_bullets))

            extra = []

            if section.get("frequency"):
                extra.append([
                    "Frequency",
                    section.get("frequency")
                ])

            if section.get("ideal_range"):
                extra.append([
                    "Ideal Range",
                    section.get("ideal_range")
                ])

            if section.get("recommended"):
                extra.append([
                    "Recommended",
                    section.get("recommended")
                ])

            if section.get("recommended_mix"):
                extra.append([
                    "Recommended Mix",
                    section.get("recommended_mix")
                ])

            schedule = section.get("schedule")
            if schedule:
                extra.append([
                    "Schedule",
                    schedule
                ])

            card = {
                "icon": "",
                "title": title,
                "summary": summary,
                "bullets": bullets,
                "extra": extra,
                "warning": section.get("warning", "")
            }

            # Keep a familiar icon even if a source title has no emoji.
            icon_map = {
                "watering": "💧",
                "sunlight": "☀️",
                "temperature": "🌡️",
                "soil": "🪴",
                "fertilizer": "🌱",
                "pruning": "🍃"
            }
            card["icon"] = icon_map.get(section_key, "🌿")

            sections.append(card)

        # Preserve the source's prevention/warning information in the
        # normalized structure used by the existing templates.
        important_warning = raw_plant.get("important_warning", [])
        safety_care = raw_plant.get("safety_care", [])

        prevention_bullets = []
        if isinstance(important_warning, list):
            prevention_bullets.extend(
                str(x) for x in important_warning if x is not None
            )

        display_data[plant_key] = {
            "plant": raw_plant.get(
                "display_name",
                f"🌿 {raw_plant_name}"
            ),
            "sections": sections,
            "prevention": {
                "summary": "Follow the important plant-care precautions and inspect the plant regularly.",
                "bullets": prevention_bullets,
                "warning": "Continue to follow disease-specific treatment guidance separately when a disease has been detected."
            },
            "calendar": raw_plant.get("calendar", {}),
            "safety_care": safety_care,
            "important_warning": important_warning,
            # Keep the complete original plant JSON available for any
            # other page/feature that needs a field not mapped above.
            "source_data": raw_plant
        }

    return display_data


print()
print("🌿 Loading plant care database...")

try:

    with open(
        PLANT_CARE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        _raw_plant_care_data = json.load(file)

    PLANT_CARE_DATA = _build_plant_care_display(
        _raw_plant_care_data
    )

    print(
        "✅ Plant care entries:",
        len(PLANT_CARE_DATA)
    )

except Exception as error:

    print("⚠️ Plant care database loading failed")
    print(error)

    # Keep the application from crashing if the JSON is temporarily
    # missing. The page will simply receive an empty care database.
    PLANT_CARE_DATA = {}


# ============================================================
# KNOWN DISEASES
# ============================================================

KNOWN_DISEASES = [

    "Chilli_Cercospora",
    "Chilli_Mites_and_Thrips",
    "Chilli_Nutritional_Deficiency",
    "Healthy_Chilli",

    "botrytis_blightgray_mold",
    "Healthy_Hibiscus",
    "leaf_spot",
    "mealybug_infestation",
    "powdery_mildew",
    "rust",
    "sooty_mold",
    "whitefly_infestation",

    "Bacterial_Spot",
    "Early_Blight",
    "Healthy_Tomato",
    "Late_Blight",
    "Leaf_Mold",
    "Septoria_Leaf_Spot",
    "Target_Spot",
    "Tomato_Mosaic_Virus",
    "Tomato_Yellow_Leaf_Curl_Virus"
]


# ============================================================
# CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    DATA_FOLDER,
    exist_ok=True
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.secret_key = "plant-health-local-secret"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# ============================================================
# ☁️ SUPABASE PERSISTENT STORAGE
# ============================================================
# Local JSON files remain as a safe fallback for Windows/offline
# development. When SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY are
# present (Render), Supabase becomes the persistent source for user
# records and Supabase Storage becomes the persistent image store.
# The service-role key must NEVER be placed in source code.
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get(
    "SUPABASE_SERVICE_ROLE_KEY", ""
).strip()
SUPABASE_BUCKET = os.environ.get(
    "SUPABASE_STORAGE_BUCKET",
    "plant-images"
).strip() or "plant-images"

SUPABASE_ENABLED = bool(
    SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY
)

print(
    "☁️ Supabase persistence:",
    "ENABLED" if SUPABASE_ENABLED else "DISABLED (local JSON only)",
    flush=True
)

SUPABASE_TABLES = {
    "users.json": "users",
    "upload_records.json": "upload_records",
    "chat_records.json": "chat_records",
    "saved_before_records.json": "saved_before_records",
    "plant_profiles.json": "plant_profiles",
    "treatment_records.json": "treatment_records",
    "care_reminders.json": "care_reminders",
    "before_after_records.json": "before_after_records",
}


def _supabase_headers(prefer=None, content_type="application/json"):
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": "Bearer " + SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": content_type,
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _supabase_request(method, path, payload=None, query=None, headers=None):
    if not SUPABASE_ENABLED:
        return None

    url = SUPABASE_URL + path
    if query:
        url += "?" + query

    body = None
    request_headers = _supabase_headers()
    if headers:
        request_headers.update(headers)

    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = Request(
        url,
        data=body,
        headers=request_headers,
        method=method.upper()
    )

    try:
        with urlopen(request, timeout=20) as response:
            raw = response.read()
            if not raw:
                return None
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return raw
    except HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(error)
        raise RuntimeError(
            f"Supabase {method} {path} failed ({error.code}): {detail[:500]}"
        ) from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Supabase connection failed: {error}"
        ) from error


def _supabase_table_for_file(file_path):
    return SUPABASE_TABLES.get(
        os.path.basename(str(file_path))
    )


def _supabase_get_rows(table):
    rows = _supabase_request(
        "GET",
        f"/rest/v1/{table}",
        query="select=*"
    )
    return rows if isinstance(rows, list) else []


def _supabase_row_to_record(row):
    if not isinstance(row, dict):
        return {}

    data = row.get("data")
    if isinstance(data, dict):
        record = dict(data)
        if row.get("id") is not None:
            record.setdefault("id", row.get("id"))
        if row.get("user_id") is not None:
            record.setdefault("user_id", row.get("user_id"))
        return record

    return {
        key: value
        for key, value in row.items()
        if key != "data"
    }


def _supabase_load_records(table):
    return [
        record
        for record in (
            _supabase_row_to_record(row)
            for row in _supabase_get_rows(table)
        )
        if record
    ]


def _supabase_upsert_record(table, record):
    if not record:
        return

    record = dict(record)
    record_id = record.get("id") or uuid.uuid4().hex
    record["id"] = record_id

    # Store the complete original Flask record in data JSONB. This
    # means existing application fields do not need to be redesigned.
    row = {
        "id": record_id,
        "data": record
    }
    if record.get("user_id") is not None:
        row["user_id"] = record.get("user_id")

    try:
        _supabase_request(
            "POST",
            f"/rest/v1/{table}",
            payload=[row],
            headers={
                "Prefer": "resolution=merge-duplicates,return=minimal"
            }
        )
        return
    except Exception as first_error:
        # Compatibility fallback for tables that were created with
        # individual columns and do not have a data JSONB column.
        try:
            _supabase_request(
                "POST",
                f"/rest/v1/{table}",
                payload=[record],
                headers={
                    "Prefer": "resolution=merge-duplicates,return=minimal"
                }
            )
            return
        except Exception:
            raise first_error


def _supabase_delete_record(table, record_id):
    if not record_id:
        return

    _supabase_request(
        "DELETE",
        f"/rest/v1/{table}",
        query="id=eq." + quote(str(record_id), safe="")
    )


def _supabase_save_records(table, records):
    records = [
        dict(record)
        for record in records
        if isinstance(record, dict)
    ]

    existing = _supabase_load_records(table)
    desired_ids = set()

    for record in records:
        record_id = record.get("id") or uuid.uuid4().hex
        record["id"] = record_id
        desired_ids.add(str(record_id))
        _supabase_upsert_record(table, record)

    # Keep deletes performed by the existing app synchronized.
    for old in existing:
        old_id = old.get("id")
        if old_id is not None and str(old_id) not in desired_ids:
            _supabase_delete_record(table, old_id)

    return True


def _storage_public_url(storage_path):
    return (
        SUPABASE_URL
        + "/storage/v1/object/public/"
        + quote(SUPABASE_BUCKET, safe="")
        + "/"
        + quote(storage_path.lstrip("/"), safe="/")
    )


def _upload_image_to_supabase(local_path, storage_path):
    if not SUPABASE_ENABLED:
        return None

    if not local_path or not os.path.isfile(local_path):
        return None

    extension = os.path.splitext(local_path)[1].lower()
    content_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(extension, "application/octet-stream")

    with open(local_path, "rb") as image_file:
        image_bytes = image_file.read()

    url = (
        SUPABASE_URL
        + "/storage/v1/object/"
        + quote(SUPABASE_BUCKET, safe="")
        + "/"
        + quote(storage_path.lstrip("/"), safe="/")
    )

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": "Bearer " + SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": content_type,
        "x-upsert": "true",
    }

    for method in ("POST", "PUT"):
        try:
            request = Request(
                url,
                data=image_bytes,
                headers=headers,
                method=method
            )
            with urlopen(request, timeout=30):
                pass
            return _storage_public_url(storage_path)
        except Exception as error:
            if method == "PUT":
                print("⚠️ Supabase image upload failed:", error)

    return None


def _download_storage_image(public_url, local_path):
    if not public_url or not str(public_url).startswith(("http://", "https://")):
        return False

    try:
        request = Request(
            public_url,
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": "Bearer " + SUPABASE_SERVICE_ROLE_KEY,
            }
        )
        with urlopen(request, timeout=30) as response:
            image_bytes = response.read()

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as image_file:
            image_file.write(image_bytes)

        return True

    except Exception as error:
        print("⚠️ Supabase image download failed:", error)
        return False


def _delete_storage_image(storage_path):
    if not SUPABASE_ENABLED or not storage_path:
        return

    url = (
        SUPABASE_URL
        + "/storage/v1/object/"
        + quote(SUPABASE_BUCKET, safe="")
        + "/"
        + quote(str(storage_path).lstrip("/"), safe="/")
    )

    try:
        request = Request(
            url,
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": "Bearer " + SUPABASE_SERVICE_ROLE_KEY,
            },
            method="DELETE"
        )
        with urlopen(request, timeout=20):
            pass
    except Exception as error:
        print("⚠️ Supabase image delete failed:", error)

def _load_json_list(file_path):
    """Load a JSON list from local storage safely."""
    try:
        if not os.path.exists(file_path):
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, list) else []

    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        print(f"⚠️ Local JSON load error: {file_path} -> {e}", flush=True)
        return []
def _persistent_load_records(file_path):
    table = _supabase_table_for_file(file_path)

    if SUPABASE_ENABLED and table:
        try:
            records = _supabase_load_records(table)

            if records:
                return records

            # First run: migrate existing local records automatically.
            local_records = _local_load_json_list(file_path)

            if local_records:
                _supabase_save_records(
                    table,
                    local_records
                )
                return local_records

            return []

        except Exception as error:
            print(
                "⚠️ Supabase read failed; using local JSON:",
                error
            )

    return _local_load_json_list(file_path)


def _persistent_save_records(file_path, records):
    local_ok = _save_json_list(
        file_path,
        records
    )

    table = _supabase_table_for_file(file_path)

    if SUPABASE_ENABLED and table:
        try:
            _supabase_save_records(
                table,
                records
            )

            print(
                "☁️ Supabase records saved:",
                table
            )

            return True

        except Exception as error:
            print(
                "❌ Supabase record save failed:",
                error
            )

            return local_ok

    return local_ok


# ============================================================
# 📱 MOBILE APP STATE / NAVIGATION PERSISTENCE
# ============================================================

MOBILE_STATE_SCRIPT = r"""
<script>
(function () {
    "use strict";
    const HISTORY = "plant_ai_visited_pages_v1";
    const LAST = "plant_ai_last_page_v1";
    const SCROLL = "plant_ai_scroll_positions_v1";
    const HOME = "plant_ai_explicit_home_v1";
    const current = window.location.pathname + window.location.search;
    const isRoot = window.location.pathname === "/";
    const launchHome = new URLSearchParams(window.location.search).get("launch") === "home";

    function read(key, fallback) {
        try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; }
        catch (e) { return fallback; }
    }
    function write(key, value) {
        try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) {}
    }

    if (isRoot && !launchHome && !sessionStorage.getItem(HOME)) {
        const last = localStorage.getItem(LAST);
        // Only restore real pages. Do NOT restore form/action/API endpoints.
        // /predict accepts POST only, so restoring it with a normal page
        // navigation causes "405 Method Not Allowed".
        const blocked = [
            "/logout",
            "/login",
            "/register",
            "/predict",
            "/api/",
            "/static/"
        ];

        if (last && last !== "/" && !blocked.some(x => last.indexOf(x) === 0)) {
            window.location.replace(last);
            return;
        }

        // If an old version saved /predict as the last page, clear it.
        if (last && blocked.some(x => last.indexOf(x) === 0)) {
            try { localStorage.removeItem(LAST); } catch (e) {}
        }
    }

    if (current && !current.startsWith("/static/")) {
        let pages = read(HISTORY, []);
        pages = pages.filter(item => item && item.path !== current);
        pages.unshift({path: current, title: document.title || "Plant Disease AI", visitedAt: new Date().toISOString()});
        write(HISTORY, pages.slice(0, 100));
        if (!isRoot) localStorage.setItem(LAST, current);
    }

    const positions = read(SCROLL, {});
    if (positions[current] !== undefined) {
        setTimeout(() => window.scrollTo(0, Number(positions[current]) || 0), 80);
    }

    let timer;
    window.addEventListener("scroll", function () {
        clearTimeout(timer);
        timer = setTimeout(function () {
            const p = read(SCROLL, {}); p[current] = window.scrollY || 0; write(SCROLL, p);
        }, 250);
    }, {passive: true});

    document.addEventListener("click", function (event) {
        const link = event.target.closest("a[href]");
        if (!link) return;
        const href = link.getAttribute("href") || "";
        if (href === "/" || href === window.location.origin + "/") {
            try { sessionStorage.setItem(HOME, "1"); } catch (e) {}
        }
    }, true);

    document.addEventListener("visibilitychange", function () {
        if (document.visibilityState === "hidden") {
            const p = read(SCROLL, {}); p[current] = window.scrollY || 0; write(SCROLL, p);
        }
    });
})();
</script>
"""

# ============================================================
# LOAD MODEL
# ============================================================
# Local Windows development keeps using the existing Keras model.
# Render/Linux uses the lightweight TFLite model so the app can
# run within Render Free's memory limit without importing TensorFlow.
# ============================================================

print()
print("=" * 60)
print("🌱 LOADING PLANT DISEASE MODEL")
print("=" * 60)

USE_TFLITE = (os.name != "nt") or os.environ.get("USE_TFLITE", "").lower() == "true"

model = None
interpreter = None
input_details = None
output_details = None
MODEL_LOCK = threading.Lock()

try:

    if USE_TFLITE:

        if not os.path.isfile(TFLITE_MODEL_PATH):
            raise FileNotFoundError(
                "TFLite model not found: " + TFLITE_MODEL_PATH
            )

        from tflite_runtime.interpreter import Interpreter

        interpreter = Interpreter(
            model_path=TFLITE_MODEL_PATH,
            num_threads=1
        )
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("✅ TFLite model loaded successfully")
        print("Input shape :", input_details[0]["shape"])
        print("Input dtype :", input_details[0]["dtype"])
        print("Output shape:", output_details[0]["shape"])
        print("Output dtype:", output_details[0]["dtype"])

    else:

        import tensorflow as tf

        # Inference only: do not restore the saved optimizer state.
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print("✅ Keras model loaded successfully")
        print("Input shape :", model.input_shape)
        print("Output shape:", model.output_shape)

except Exception as error:

    print("❌ Model loading failed")
    print(error)

    raise


# ============================================================
# LOAD CLASS NAMES
# ============================================================

print()
print("📚 Loading class names...")

try:

    with open(
        CLASS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(
            file
        )

except Exception as error:

    print("❌ class_names.json loading failed")
    print(error)

    raise


if isinstance(
    class_names,
    dict
):

    index_to_class = {}

    for key, value in class_names.items():

        try:

            if isinstance(
                value,
                int
            ):

                index_to_class[int(value)] = key

            elif str(value).isdigit():

                index_to_class[int(value)] = key

            elif str(key).isdigit():

                index_to_class[int(key)] = value

        except (
            TypeError,
            ValueError
        ):

            continue

elif isinstance(
    class_names,
    list
):

    index_to_class = {
        index: name
        for index, name in enumerate(
            class_names
        )
    }

else:

    raise ValueError(
        "Unsupported class_names.json format."
    )


print(
    "✅ Number of classes:",
    len(index_to_class)
)

print(
    "📚 Classes:",
    list(index_to_class.values())
)


# ============================================================
# LOAD TREATMENT DATABASE
# ============================================================

print()
print("📋 Loading treatment database...")

try:

    with open(
        TREATMENT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        treatments = json.load(
            file
        )

    if not isinstance(
        treatments,
        dict
    ):

        treatments = {}

    treatments.pop(
        "metadata",
        None
    )

    print(
        "✅ Treatment entries:",
        len(treatments)
    )

except Exception as error:

    print(
        "⚠️ Treatment database loading failed"
    )

    print(error)

    treatments = {}


# ============================================================
# LOAD CHAT KNOWLEDGE BASE
#
# A broader, hand-curated set of pattern -> response rules
# (greetings, weather, watering, farmer/student framing,
# before/after, etc.) that supplements - but never replaces -
# the detailed per-class disease answers built from the
# treatment database above. See match_chat_knowledge() and its
# use inside generate_chat_response() further down.
# ============================================================

# Prefer the expanded project chat knowledge database.
# If it is not present, fall back to the original filename so
# the existing app still works without changing any other logic.
CHAT_KNOWLEDGE_CANDIDATES = [
    os.path.join(BASE_DIR, "chat_knowledge_expanded.json"),
    os.path.join(BASE_DIR, "chat_knowledge.json")
]

CHAT_KNOWLEDGE_PATH = next(
    (path for path in CHAT_KNOWLEDGE_CANDIDATES if os.path.isfile(path)),
    CHAT_KNOWLEDGE_CANDIDATES[-1]
)

print()
print("🧠 Loading chat knowledge base...")
print("📄 Chat knowledge file:", os.path.basename(CHAT_KNOWLEDGE_PATH))

try:

    with open(
        CHAT_KNOWLEDGE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        chat_knowledge = json.load(
            file
        )

    if not isinstance(
        chat_knowledge,
        dict
    ):

        chat_knowledge = {}

    CHAT_KNOWLEDGE_INTENTS = chat_knowledge.get(
        "intents",
        []
    )

    if not isinstance(
        CHAT_KNOWLEDGE_INTENTS,
        list
    ):

        CHAT_KNOWLEDGE_INTENTS = []

    print(
        "✅ Chat knowledge intents:",
        len(CHAT_KNOWLEDGE_INTENTS)
    )

except Exception as error:

    print(
        "⚠️ Chat knowledge base loading failed"
    )

    print(error)

    chat_knowledge = {}

    CHAT_KNOWLEDGE_INTENTS = []


# Pre-normalize patterns once at startup instead of on every
# chat message, and skip intents that already have a dedicated,
# more detailed hand-built answer elsewhere in this file (the
# 21 disease classes, greeting, and the home/app-info intents),
# so this knowledge base only ever *adds* coverage instead of
# replacing a richer existing answer.

CHAT_KNOWLEDGE_SKIP_INTENTS = {

    "greeting",

    "app_help",

    "supported_plants",

    "confidence",

    "treatment_general",

    "what_to_do_first",

    "before_after"
}

CHAT_KNOWLEDGE_INDEX = []

for _entry in CHAT_KNOWLEDGE_INTENTS:

    if not isinstance(_entry, dict):
        continue

    if _entry.get("intent") in CHAT_KNOWLEDGE_SKIP_INTENTS:
        continue

    _patterns = _entry.get("patterns", [])

    if not isinstance(_patterns, list):
        continue

    _normalized_patterns = [
        re.sub(
            r"[^a-z0-9\s]",
            "",
            str(p).strip().lower()
        )
        for p in _patterns
        if str(p).strip()
    ]

    _normalized_patterns = [
        p for p in _normalized_patterns if p
    ]

    if not _normalized_patterns:
        continue

    CHAT_KNOWLEDGE_INDEX.append({
        "intent": _entry.get("intent", ""),
        "patterns": _normalized_patterns,
        "base_response": _entry.get(
            "base_response",
            _entry.get("response", "")
        ),
        "mentality_responses": _entry.get(
            "mentality_responses",
            {}
        )
    })


def match_chat_knowledge(message, mentality="friendly"):

    """
    Matches a user message against the chat_knowledge.json
    pattern list. Returns the best-matching entry's response
    text for the given mentality, or None if nothing matches
    well enough to answer with.

    "Best" = the single longest matching pattern across all
    intents, so a more specific phrase ("overwatering yellow
    leaves") wins over a shorter generic one ("watering") when
    both happen to appear in the message.
    """

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        str(message or "").strip().lower()
    )

    if not text:
        return None

    best_entry = None
    best_length = 0

    for entry in CHAT_KNOWLEDGE_INDEX:

        for pattern in entry["patterns"]:

            if (
                pattern in text
                and
                len(pattern) > best_length
            ):

                best_entry = entry
                best_length = len(pattern)

    if best_entry is None:
        return None

    responses = best_entry.get(
        "mentality_responses",
        {}
    )

    return (
        responses.get(mentality)
        or best_entry.get("base_response")
        or None
    )


# ============================================================
# CHAT MODES
# ============================================================

MENTALITIES = {

    "friendly": {
        "name": "🙂 Friendly",
        "style": "warm, simple and encouraging"
    },

    "simple": {
        "name": "🏡 Simple",
        "style": "very plain, step-by-step and easy to follow"
    },

    "quick": {
        "name": "⚡ Quick",
        "style": "short, straight-to-the-point answers only"
    },

    "detailed": {
        "name": "🔬 Detailed",
        "style": "thorough, technical and structured"
    }
}


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text):

    text = str(
        text or ""
    ).lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def compact_text(text):

    return re.sub(
        r"[^a-z0-9]",
        "",
        normalize_text(text)
    )


def clean_label(text):

    text = str(
        text or ""
    )

    text = text.replace(
        "_",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def value_to_text(value):

    if value is None:
        return ""

    if isinstance(
        value,
        str
    ):

        return value.strip()

    if isinstance(
        value,
        (int, float, bool)
    ):

        return str(value)

    if isinstance(
        value,
        list
    ):

        output = []

        for item in value:

            if isinstance(
                item,
                dict
            ):

                for key, val in item.items():

                    key_text = clean_label(
                        key
                    )

                    val_text = value_to_text(
                        val
                    )

                    if val_text:

                        output.append(
                            f"{key_text}: {val_text}"
                        )

            else:

                item_text = value_to_text(
                    item
                )

                if item_text:

                    output.append(
                        f"• {item_text}"
                    )

        return "\n".join(
            output
        )

    if isinstance(
        value,
        dict
    ):

        output = []

        for key, val in value.items():

            key_text = clean_label(
                key
            )

            val_text = value_to_text(
                val
            )

            if val_text:

                output.append(
                    f"{key_text}: {val_text}"
                )

        return "\n".join(
            output
        )

    return str(value)


def first_available(
    data,
    keys
):

    if not isinstance(
        data,
        dict
    ):

        return ""

    for key in keys:

        if key not in data:
            continue

        value = data.get(
            key
        )

        text = value_to_text(
            value
        )

        if text:
            return text

    return ""


# ============================================================
# PLANT DETECTION FROM MESSAGE
# ============================================================

def detect_plant_from_message(message):

    text = normalize_text(
        message
    )

    compact = compact_text(
        message
    )

    # --------------------------------------------------------
    # HIBISCUS
    # --------------------------------------------------------

    hibiscus_words = [
        "hibiscus",
        "hibisucs",
        "hibiscusplant",
        "shoe flower"
    ]

    for word in hibiscus_words:

        if word in text or word in compact:

            return "hibiscus"


    # --------------------------------------------------------
    # CHILLI
    # --------------------------------------------------------

    chilli_words = [
        "chilli",
        "chili",
        "chilliplant",
        "greenchilli",
        "green chili",
        "mirchi"
    ]

    for word in chilli_words:

        if word in text or word in compact:

            return "chilli"


    # --------------------------------------------------------
    # TOMATO
    # --------------------------------------------------------

    tomato_words = [
        "tomato",
        "tomatoplant",
        "tomato plant"
    ]

    for word in tomato_words:

        if word in text or word in compact:

            return "tomato"


    return None


# ============================================================
# PLANT DETECTION FROM DISEASE
# ============================================================

def detect_plant_from_disease(disease):

    raw = str(
        disease or ""
    ).strip()

    if not raw:
        return None


    # --------------------------------------------------------
    # EXACT-CASE CHECKS FIRST
    #
    # "healthy" (Hibiscus) and "Healthy" (Tomato) are two
    # different classes in the treatment database that only
    # differ by capitalization. Lowercasing too early made them
    # indistinguishable, so both silently failed to resolve to
    # a plant. Check the exact original casing before any
    # lowercasing happens below.
    # --------------------------------------------------------

    if raw == "healthy":
        return "hibiscus"

    if raw == "Healthy":
        return "tomato"

    if raw == "Healthy_Chilli":
        return "chilli"


    disease = raw.lower()


    # --------------------------------------------------------
    # CHILLI
    # --------------------------------------------------------

    if (
        disease.startswith("chilli_")
        or
        disease == "healthy_chili"
    ):

        return "chilli"


    # --------------------------------------------------------
    # HIBISCUS
    # --------------------------------------------------------

    if (
        disease == "healthy_hibiscus"
        or
        disease in [
            "botrytis_blightgray_mold",
            "leaf_spot",
            "mealybug_infestation",
            "powdery_mildew",
            "rust",
            "sooty_mold",
            "whitefly_infestation"
        ]
    ):

        return "hibiscus"


    # --------------------------------------------------------
    # TOMATO
    # --------------------------------------------------------

    if (
        disease == "healthy_tomato"
        or
        disease in [
            "bacterial_spot",
            "early_blight",
            "late_blight",
            "leaf_mold",
            "septoria_leaf_spot",
            "target_spot",
            "tomato_mosaic_virus",
            "tomato_yellow_leaf_curl_virus"
        ]
    ):

        return "tomato"


    return None


# ============================================================
# DISEASE DETECTION FROM MESSAGE
# ============================================================

def detect_disease_from_message(message):

    text = normalize_text(
        message
    )

    compact = compact_text(
        message
    )


    # --------------------------------------------------------
    # DATABASE DISEASE NAMES
    #
    # Checked longest-name-first: a short generic name like
    # "leaf_spot" is a substring of a more specific compound
    # name like "Septoria_Leaf_Spot", so checking in arbitrary
    # dict order let the generic entry win even when the user
    # clearly named the specific one. Longest match first fixes
    # that without needing to touch dict ordering.
    # --------------------------------------------------------

    sorted_diseases = sorted(
        treatments.keys(),
        key=lambda d: len(normalize_text(d)),
        reverse=True
    )

    for disease in sorted_diseases:

        disease_text = normalize_text(
            disease
        )

        disease_compact = compact_text(
            disease
        )

        if (
            disease_text in text
            or
            disease_compact in compact
        ):

            # "healthy" (Hibiscus) and "Healthy" (Tomato) are two
            # different classes that normalize to the same text.
            # If the message names a specific plant, prefer the
            # matching one instead of whichever happens to be
            # checked first.
            if disease_text == "healthy":

                if "tomato" in text:
                    return "Healthy"

                if "hibiscus" in text:
                    return "healthy"

            return disease


    # --------------------------------------------------------
    # COMMON ALIASES
    # --------------------------------------------------------

    aliases = {

        "late blight":
            "Late_Blight",

        "early blight":
            "Early_Blight",

        "bacterial spot":
            "Bacterial_Spot",

        "leaf mold":
            "Leaf_Mold",

        "leaf spot":
            "leaf_spot",

        "septoria":
            "Septoria_Leaf_Spot",

        "septoria leaf spot":
            "Septoria_Leaf_Spot",

        "target spot":
            "Target_Spot",

        "mosaic virus":
            "Tomato_Mosaic_Virus",

        "tomato mosaic":
            "Tomato_Mosaic_Virus",

        "yellow leaf curl":
            "Tomato_Yellow_Leaf_Curl_Virus",

        "yellow leaf curl virus":
            "Tomato_Yellow_Leaf_Curl_Virus",

        "powdery mildew":
            "powdery_mildew",

        "rust":
            "rust",

        "sooty mold":
            "sooty_mold",

        "sooty mould":
            "sooty_mold",

        "mealybug":
            "mealybug_infestation",

        "mealbug":
            "mealybug_infestation",

        "melbug":
            "mealybug_infestation",

        "whitefly":
            "whitefly_infestation",

        "mites":
            "Chilli_Mites_and_Thrips",

        "thrips":
            "Chilli_Mites_and_Thrips",

        "cercospora":
            "Chilli_Cercospora",

        "nutritional deficiency":
            "Chilli_Nutritional_Deficiency",

        "healthy chilli":
            "Healthy_Chilli",

        "healthy chili":
            "Healthy_Chilli",

        "powdery":
            "powdery_mildew",

        "soot mold":
            "sooty_mold",

        "soot mould":
            "sooty_mold",

        "gray mold":
            "botrytis_blightgray_mold",

        "grey mold":
            "botrytis_blightgray_mold",

        "gray mould":
            "botrytis_blightgray_mold",

        "grey mould":
            "botrytis_blightgray_mold",

        "botrytis":
            "botrytis_blightgray_mold",

        "botrytis blight":
            "botrytis_blightgray_mold",

        "blight gray mold":
            "botrytis_blightgray_mold",

        "thrip":
            "Chilli_Mites_and_Thrips",

        "mite":
            "Chilli_Mites_and_Thrips",

        "mealybugs":
            "mealybug_infestation",

        "whiteflies":
            "whitefly_infestation",

        "white fly":
            "whitefly_infestation",

        "white flies":
            "whitefly_infestation",

        "sooty":
            "sooty_mold",

        "mosaic":
            "Tomato_Mosaic_Virus",

        "curl virus":
            "Tomato_Yellow_Leaf_Curl_Virus",

        "leaf curl":
            "Tomato_Yellow_Leaf_Curl_Virus"
    }


    for alias, disease in aliases.items():

        if alias in text:

            return disease


    return None


# ============================================================
# TREATMENT LOOKUP
# ============================================================

def find_treatment(disease):

    disease = str(
        disease or ""
    ).strip()

    if not disease:
        return {}


    if disease in treatments:

        value = treatments[disease]

        return (
            value
            if isinstance(
                value,
                dict
            )
            else {}
        )


    disease_lower = disease.lower()


    for key, value in treatments.items():

        if str(key).lower() == disease_lower:

            return (
                value
                if isinstance(
                    value,
                    dict
                )
                else {}
            )


    normalized_requested = re.sub(
        r"[^a-z0-9]+",
        "",
        disease_lower
    )
    # Treat US/UK spelling variants as the same disease/class.
    normalized_requested = normalized_requested.replace(
        "chili",
        "chilli"
    )


    for key, value in treatments.items():

        candidates = [
            str(key)
        ]


        if isinstance(
            value,
            dict
        ):

            candidates.append(
                str(
                    value.get(
                        "display_name",
                        ""
                    )
                )
            )


        for candidate in candidates:

            normalized_candidate = re.sub(
                r"[^a-z0-9]+",
                "",
                candidate.lower()
            )
            normalized_candidate = normalized_candidate.replace(
                "chili",
                "chilli"
            )

            if (
                normalized_candidate
                ==
                normalized_requested
            ):

                return (
                    value
                    if isinstance(
                        value,
                        dict
                    )
                    else {}
                )


    return {}


# ============================================================
# DISPLAY NAME
# ============================================================

def get_display_name(
    disease,
    treatment
):

    if isinstance(
        treatment,
        dict
    ):

        display_name = treatment.get(
            "display_name"
        )

        if display_name:

            return str(
                display_name
            )


    return clean_label(
        disease or "Unknown"
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================

def allowed_file(filename):

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


def validate_image_file(file):

    try:

        if not file or not allowed_file(file.filename or ""):
            return False

        # Read/verify without consuming the upload permanently.
        position = file.stream.tell()

        with Image.open(file.stream) as image:
            image.verify()

        file.stream.seek(position)
        return True

    except Exception:

        try:
            file.stream.seek(0)
        except Exception:
            pass

        return False


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image_path):

    with Image.open(
        image_path
    ) as image:

        image = image.convert(
            "RGB"
        )

        image = image.resize(
            IMAGE_SIZE
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        )


    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# CONFIDENCE
# ============================================================

def get_confidence_level(
    confidence
):

    if confidence >= 85:

        return "HIGH CONFIDENCE"

    if confidence >= 60:

        return "MODERATE CONFIDENCE"

    return "LOW CONFIDENCE"


# ============================================================
# DISEASE PREDICTION
# ============================================================

def predict_disease(
    image_path
):

    image_array = prepare_image(
        image_path
    )

    if USE_TFLITE:

        input_info = input_details[0]
        output_info = output_details[0]

        input_data = image_array.astype(
            input_info["dtype"],
            copy=False
        )

        # Support quantized TFLite models too, although the current
        # conversion is float32 and therefore follows the normal path.
        input_scale, input_zero = input_info.get("quantization", (0.0, 0))
        if input_scale:
            input_data = np.round(
                image_array / input_scale + input_zero
            ).astype(input_info["dtype"])

        with MODEL_LOCK:
            interpreter.set_tensor(
                input_info["index"],
                input_data
            )
            interpreter.invoke()
            predictions = interpreter.get_tensor(
                output_info["index"]
            )

        output_scale, output_zero = output_info.get("quantization", (0.0, 0))
        if output_scale:
            predictions = (
                predictions.astype(np.float32) - output_zero
            ) * output_scale

    else:

        predictions = model.predict(
            image_array,
            verbose=0
        )

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )


    if predictions.ndim == 2:

        predictions = predictions[0]


    if predictions.ndim != 1:

        raise ValueError(
            "Unexpected model prediction shape."
        )


    predicted_index = int(
        np.argmax(
            predictions
        )
    )


    disease = index_to_class.get(
        predicted_index,
        "Unknown"
    )


    confidence = float(
        predictions[predicted_index]
    ) * 100.0


    confidence = round(
        confidence,
        2
    )


    confidence_level = get_confidence_level(
        confidence
    )


    return (
        disease,
        confidence,
        confidence_level
    )


# ============================================================
# TERMINAL RECORD LOGGING
# ============================================================
# This is ONLY for local terminal visibility. It does not change
# how records are stored, filtered, or displayed in the app.
# ============================================================

def _terminal_record_log(title, record=None, extra=None):
    print()
    print("=" * 60, flush=True)
    print(title, flush=True)
    print("=" * 60, flush=True)

    if isinstance(record, dict):
        for key, value in record.items():
            print(f"{key:<20}: {value}", flush=True)

    if isinstance(extra, dict):
        for key, value in extra.items():
            print(f"{key:<20}: {value}", flush=True)

    print("=" * 60, flush=True)


def _terminal_records_count(title, records, user_id=None):
    print(
        f"📋 {title}: {len(records)} record(s)"
        + (f" | user_id={user_id}" if user_id else ""),
        flush=True
    )


# ============================================================
# RECORD STORAGE
# ============================================================

def load_collection_records(user_id=None):

    records = _persistent_load_records(
        COLLECTION_FILE
    )

    _terminal_records_count(
        "Upload records loaded",
        records,
        user_id
    )

    if user_id is None:
        return records

    return [
        record
        for record in records
        if str(record.get("user_id", "")) == str(user_id)
    ]


def save_collection_record(
    original_filename,
    saved_filename,
    disease,
    confidence,
    confidence_level
):

    user_id = session.get("user_id")
    records = load_collection_records()

    image_path = (
        "static/uploads/" + saved_filename
    )

    local_image = os.path.join(
        UPLOAD_FOLDER,
        saved_filename
    )

    storage_url = _upload_image_to_supabase(
        local_image,
        "uploads/" + saved_filename
    )

    if storage_url:
        image_path = storage_url

    record = {
        "id": uuid.uuid4().hex,
        "user_id": user_id,
        "original_filename": original_filename,
        "saved_filename": saved_filename,
        "image_path": image_path,
        "predicted_disease": disease,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "review_status": "pending",
        "verified_disease": None,
        "uploaded_at": datetime.now().isoformat(
            timespec="seconds"
        )
    }

    records.append(record)

    if _persistent_save_records(
        COLLECTION_FILE,
        records
    ):
        print("💾 Upload record saved", flush=True)
        _terminal_record_log(
            "📋 UPLOAD RECORD SAVED",
            record,
            {
                "Total records": len(records),
                "Storage": "Supabase + local JSON"
                if SUPABASE_ENABLED
                else "Local JSON"
            }
        )
    else:
        print("⚠️ Upload record save error", flush=True)

    return record


# ============================================================
# USER ACCOUNTS (name + phone number login)
# ============================================================

def load_users():
    return _persistent_load_records(
        USERS_FILE
    )


def save_users(users):
    return _persistent_save_records(
        USERS_FILE,
        users
    )


def normalize_phone(phone):

    # Keep digits only (strips spaces, dashes, parentheses, +country code punctuation).
    return re.sub(
        r"[^0-9]",
        "",
        str(phone or "")
    )


def is_valid_phone(phone):

    digits = normalize_phone(
        phone
    )

    return 7 <= len(digits) <= 15


def find_user_by_phone(phone):

    digits = normalize_phone(
        phone
    )

    for user in load_users():

        if user.get("phone") == digits:

            return user

    return None


def find_user_by_id(user_id):

    for user in load_users():

        if user.get("id") == user_id:

            return user

    return None


def get_or_create_user(name, phone):

    """
    Looks up an account by phone number. If one already exists,
    its name is left unchanged (phone number is the identity) and
    the account is returned as-is. Otherwise a new account is
    created with the given name + phone.
    """

    digits = normalize_phone(
        phone
    )

    existing = find_user_by_phone(
        digits
    )

    if existing:

        return existing


    users = load_users()

    new_user = {

        "id":
            uuid.uuid4().hex,

        "name":
            name.strip(),

        "phone":
            digits,

        "created_at":
            datetime.utcnow().isoformat()
    }

    users.append(
        new_user
    )

    save_users(
        users
    )

    return new_user


def get_current_user():

    user_id = session.get(
        "user_id"
    )

    if not user_id:
        return None

    return find_user_by_id(
        user_id
    )


# ============================================================
# CHAT RECORDS
# ============================================================

def load_chat_records(user_id=None):

    records = _persistent_load_records(
        CHAT_FILE
    )

    _terminal_records_count(
        "Chat records loaded",
        records,
        user_id
    )

    if user_id is None:
        return records

    return [
        record
        for record in records
        if str(record.get("user_id", "")) == str(user_id)
    ]


def save_chat_message(
    message,
    response,
    mentality,
    disease
):

    user_id = session.get("user_id")
    records = load_chat_records()

    records.append({
        "id": uuid.uuid4().hex,
        "user_id": user_id,
        "message": message,
        "response": response,
        "mentality": mentality,
        "disease": disease,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        )
    })

    chat_record = records[-1]

    if _persistent_save_records(
        CHAT_FILE,
        records
    ):
        print("💬 Chat record saved", flush=True)
        _terminal_record_log(
            "💬 CHAT RECORD SAVED",
            chat_record,
            {
                "Total records": len(records),
                "Storage": "Supabase + local JSON"
                if SUPABASE_ENABLED
                else "Local JSON"
            }
        )
    else:
        print("⚠️ Chat record save error", flush=True)


# ============================================================
# BUILD DISEASE CONTEXT
# ============================================================

def build_context(
    disease,
    confidence,
    treatment
):

    treatment = (
        treatment
        if isinstance(
            treatment,
            dict
        )
        else {}
    )


    return {

        "disease":
            get_display_name(
                disease,
                treatment
            ),

        "raw_disease":
            disease,

        "confidence":
            round(
                float(
                    confidence or 0
                ),
                2
            ),

        "description":
            first_available(
                treatment,
                [
                    "description",
                    "what_is_this",
                    "what_is_it"
                ]
            ),

        "why":
            first_available(
                treatment,
                [
                    "why_does_it_happen",
                    "why",
                    "cause",
                    "causes",
                    "reason"
                ]
            ),

        "symptoms":
            first_available(
                treatment,
                [
                    "symptoms_to_look_for",
                    "symptoms",
                    "signs"
                ]
            ),

        "first_steps":
            first_available(
                treatment,
                [
                    "what_to_do_first",
                    "first_steps",
                    "first_action"
                ]
            ),

        "treatment":
            first_available(
                treatment,
                [
                    "professional_treatment_guidance",
                    "treatment",
                    "treatments",
                    "control",
                    "final_recommendation"
                ]
            ),

        "home_care":
            first_available(
                treatment,
                [
                    "home_care",
                    "home_remedy",
                    "natural_remedy"
                ]
            ),

        "prevention":
            first_available(
                treatment,
                [
                    "prevention",
                    "prevent"
                ]
            ),

        "medicine":
            first_available(
                treatment,
                [
                    "example_products",
                    "medicine",
                    "medicines",
                    "fungicide",
                    "pesticide",
                    "product"
                ]
            )
            + (
                "\n\n" + str(treatment.get("example_products_disclaimer"))
                if treatment.get("example_products_disclaimer")
                else ""
            ),

        "safety":
            first_available(
                treatment,
                [
                    "treatment_safety",
                    "safety",
                    "safety_note",
                    "warning"
                ]
            ),

        "expert_help":
            first_available(
                treatment,
                [
                    "when_to_get_expert_help",
                    "expert_help",
                    "professional_help"
                ]
            )
    }


# ============================================================
# HOME CHAT INTENT DETECTION
# ============================================================

def detect_home_intent(text, has_disease_context=False):

    text = normalize_text(
        text
    )


    # ========================================================
    # GUARD: is a specific disease or plant already in play?
    #
    # The broad keyword fallbacks below (further down this
    # function) are only allowed to fire when there is NO
    # disease/plant context at all - neither named in this
    # message NOR already active from a scan/URL (passed in via
    # has_disease_context). That way a real question like "how
    # do I treat this?" on the treatment page (disease already
    # active, nothing named in the message itself) still goes to
    # the specific disease answer, and only genuinely general
    # questions ("how does treatment work", "tell me about plant
    # care", "how does tracking work") get answered here.
    # ========================================================

    has_specific_context = (
        has_disease_context
        or bool(detect_disease_from_message(text))
        or bool(detect_plant_from_message(text))
    )


    # ========================================================
    # 0. ABOUT THE APP
    # ========================================================

    about_app_patterns = [

        "what is this app",

        "what is this app for",

        "what does this app do",

        "what is this for",

        "what do you do",

        "what can you do",

        "what can this app do",

        "how does this app work",

        "how does this work",

        "who are you",

        "what are you",

        "tell me about this app",

        "tell me about yourself",

        "what is plant disease ai",

        "what is this website",

        "how do i use this app",

        "how to use this app"
    ]


    for pattern in about_app_patterns:

        if pattern in text:

            return "about_app"


    # ========================================================
    # 1. SUPPORTED DISEASES
    # ========================================================

    supported_patterns = [

        "what diseases can this app identify",

        "what diseases can this app diagnose",

        "what diseases can you identify",

        "what diseases can you diagnose",

        "which diseases can this app identify",

        "which diseases can this app diagnose",

        "which plant can i diagnose",

        "which plants can i diagnose",

        "which plants are supported",

        "what plants are supported",

        "what plants does this app support",

        "what plants can this app identify",

        "what plants can i diagnose",

        "what plant can i diagnose",

        "supported plants",

        "what can i diagnose",

        "what can this app diagnose",

        "what can this app identify"
    ]


    for pattern in supported_patterns:

        if pattern in text:

            return "supported_diseases"


    # ========================================================
    # 2. GENERAL PREVENTION
    #
    # Gated by has_specific_context: with an active/named disease,
    # "how do I prevent this" should answer for THAT disease, not
    # give the app-wide generic prevention tips.
    # ========================================================

    if not has_specific_context:

        prevention_patterns = [

            "how can i prevent plant diseases",

            "how do i prevent plant diseases",

            "how to prevent plant diseases",

            "how can plant diseases be prevented",

            "general disease prevention",

            "prevent plant disease",

            "plant disease prevention",

            "how can i protect my plants",

            "how do i protect my plants",

            "ways to prevent plant disease"
        ]


        for pattern in prevention_patterns:

            if pattern in text:

                return "general_prevention"


    # ========================================================
    # 3. GENERAL PLANT CARE
    #
    # Same gating as above - an active disease means "care advice"
    # is almost always about THAT disease.
    # ========================================================

    if not has_specific_context:

        care_patterns = [

            "give me general plant care advice",

            "give me plant care advice",

            "general plant care",

            "plant care advice",

            "how do i care for my plant",

            "how should i care for my plant",

            "how to care for plants",

            "general care advice",

            "basic plant care",

            "plant care tips"
        ]


        for pattern in care_patterns:

            if pattern in text:

                return "general_care"


    # ========================================================
    # 4. GENERAL FIRST ACTION
    #
    # Same gating - "what should I do first" on a page with an
    # active diagnosis means first-aid for THAT diagnosis, not
    # the generic app-wide checklist.
    # ========================================================

    if not has_specific_context:

        first_patterns = [

            "what should i do first if my plant is infected",

            "what should i do first when my plant is infected",

            "what should i do if my plant is infected",

            "what do i do if my plant is infected",

            "what should i do first",

            "what do i do first",

            "what can i do first",

            "first thing to do if my plant is infected",

            "first thing to do if plant is infected",

            "first step if plant is infected",

            "plant is infected what should i do"
        ]


        for pattern in first_patterns:

            if pattern in text:

                return "general_first_action"


    # ========================================================
    # 5. BROAD FALLBACKS (guarded)
    #
    # The categories above only match a fixed list of exact
    # phrases, so plenty of natural real-world phrasings ("tell
    # me about plant care", "how does treatment work in this
    # app", "how do I prevent diseases in general") were falling
    # through to the disease-specific flow and getting bounced
    # with a "scan a leaf first" message even though no disease
    # was named at all. These broader keyword checks catch those
    # cases - but only when the message does NOT name a specific
    # disease or plant (has_specific_context), so a real question
    # like "how to treat early blight" is completely unaffected
    # and still goes to the specific disease answer.
    # ========================================================

    if not has_specific_context:

        # ---- before/after recovery tracking ----
        #
        # Uses fuzzy word matching instead of a strict regex, so
        # small typos ("before affter", "beofre after") still
        # match instead of silently falling through to the
        # generic assistant intro.

        words = text.split()

        has_before_word = any(
            difflib.SequenceMatcher(None, w, "before").ratio() >= 0.8
            for w in words
        )

        has_after_word = any(
            difflib.SequenceMatcher(None, w, "after").ratio() >= 0.75
            for w in words
        )

        if (
            (has_before_word and has_after_word)
            or
            re.search(
                r"\btrack(ing)?\b|\bprogress\b|"
                r"\bcompar(e|ison|ing)\b|"
                r"\brecover(y|ing|ed)?\b",
                text
            )
        ):

            return "general_tracking"


        # ---- general treatment / medicine question ----

        if re.search(
            r"\btreat(ment|ing)?\b|\bmedicine\b|\bfungicide\b|"
            r"\bpesticide\b|\bspray\b|\bcure\b",
            text
        ):

            return "general_treatment_info"


        # ---- general plant care question ----

        if re.search(
            r"\bcare\b|\bwatering\b|\bfertiliz",
            text
        ):

            return "general_care"


        # ---- general prevention question ----

        if re.search(
            r"\bprevent(ion)?\b|\bavoid\b",
            text
        ):

            return "general_prevention"


        # ---- general "what do I do" question ----

        if re.search(
            r"what (should|do|can) i do|first step|what now|what next",
            text
        ):

            return "general_first_action"


    return None


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(text, has_disease_context=False):

    text = normalize_text(
        text
    )


    # --------------------------------------------------------
    # HOME QUESTIONS FIRST
    # --------------------------------------------------------

    home_intent = detect_home_intent(
        text,
        has_disease_context=has_disease_context
    )


    if home_intent:

        return home_intent


    # --------------------------------------------------------
    # DISEASE / PLANT
    # --------------------------------------------------------

    disease_from_message = detect_disease_from_message(
        text
    )

    plant_from_message = detect_plant_from_message(
        text
    )


    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    if any(
        re.search(r"\b" + re.escape(x) + r"\b", text)
        for x in [
            "hello",
            "hi",
            "hey",
            "vanakkam",
            "good morning",
            "good afternoon",
            "good evening"
        ]
    ):

        return "greeting"


    # --------------------------------------------------------
    # MENTALITY
    # --------------------------------------------------------

    if any(
        x in text
        for x in [
            "friendly mode",
            "simple mode",
            "quick mode",
            "detailed mode",
            "chat style",
            "chat mode",
            "mentality",
            "change style",
            "what mode are you",
            "what mode are you in",
            "current mode",
            "which mode"
        ]
    ):

        return "mentality"


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if any(
        x in text
        for x in [
            "confidence",
            "accurate",
            "accuracy",
            "correct",
            "sure",
            "reliable",
            "trust",
            "certain"
        ]
    ):

        return "confidence"


    # --------------------------------------------------------
    # WORRIED
    # --------------------------------------------------------

    if any(
        x in text
        for x in [
            "i am worried",
            "i'm worried",
            "worried",
            "scared",
            "afraid",
            "panic",
            "please help",
            "is my plant dying",
            "will my plant die",
            "my plant is dying"
        ]
    ):

        return "reassurance"


    # --------------------------------------------------------
    # HOME CARE
    # --------------------------------------------------------

    if any(
        x in text
        for x in [
            "home remedy",
            "home care",
            "natural remedy",
            "natural treatment",
            "at home",
            "home treatment",
            "without medicine",
            "without chemical"
        ]
    ):

        return "home_care"


    # --------------------------------------------------------
    # TREATMENT
    # --------------------------------------------------------

    if any(
        x in text
        for x in [
            "treatment",
            "medicine",
            "fungicide",
            "pesticide",
            "insecticide",
            "product",
            "products",
            "spray",
            "chemical",
            "control",
            "cure",
            "kill",
            "get rid of",
            "remedy",
            "solution",
            "recommend",
            "suggest",
            "what should i use",
            "what can i use",
            "what to use",
            "apply",
            "how to treat",
            "how can i treat",
            "treat"
        ]
    ):

        return "treatment"


    # --------------------------------------------------------
    # PREVENTION
    # --------------------------------------------------------

    if any(
        x in text
        for x in [
            "prevent",
            "prevention",
            "avoid",
            "protect",
            "stop it",
            "stop this",
            "how to prevent"
        ]
    ):

        return "prevention"


    # --------------------------------------------------------
    # SYMPTOMS
    # --------------------------------------------------------

    if any(
        x in text
        for x in [
            "symptom",
            "symptoms",
            "signs",
            "sign of",
            "look like",
            "recognize",
            "how to know",
            "how does it look"
        ]
    ):

        return "symptoms"


    # --------------------------------------------------------
    # FIRST ACTION
    # --------------------------------------------------------

    if any(
        x in text
        for x in [
            "what should i do",
            "what do i do",
            "what can i do",
            "what now",
            "what next",
            "do first",
            "first step",
            "start with",
            "save my plant",
            "help my plant",
            "right now",
            "immediate action"
        ]
    ):

        return "first_action"


    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    if any(
        x in text
        for x in [
            "serious",
            "dangerous",
            "severity",
            "severe",
            "will it spread",
            "can it spread",
            "spread to",
            "kill my plant",
            "plant die",
            "how bad"
        ]
    ):

        return "severity"


    # --------------------------------------------------------
    # EXPERT
    # --------------------------------------------------------

    if any(
        x in text
        for x in [
            "expert",
            "agriculture officer",
            "agricultural officer",
            "specialist",
            "professional",
            "agronomist"
        ]
    ):

        return "expert"


    # --------------------------------------------------------
    # CAUSE
    # --------------------------------------------------------

    if any(
        x in text
        for x in [
            "why",
            "cause",
            "caused",
            "happen",
            "reason"
        ]
    ):

        return "cause"


    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    if any(
        x in text
        for x in [
            "what is this",
            "what is it",
            "what disease",
            "what problem",
            "meaning",
            "explain this",
            "tell me about",
            "explain"
        ]
    ):

        return "information"


    # --------------------------------------------------------
    # BARE DISEASE / PLANT NAME (no question words at all)
    #
    # If the user just types a disease name on its own - "Early
    # Blight", "powdery mildew", "gray mold" - none of the
    # keyword checks above match anything, so this used to fall
    # through to "general" and show the generic app intro
    # instead of actually answering about that disease. If a
    # disease was named in the message, give a full overview of
    # it instead.
    # --------------------------------------------------------

    if disease_from_message:

        return "information"


    return "general"


# ============================================================
# AUTOMATIC MENTALITY
# ============================================================

def detect_message_mentality(text):

    text = normalize_text(
        text
    )


    if any(
        x in text
        for x in [
            "quick answer",
            "quickly",
            "in short",
            "short answer",
            "briefly",
            "tl;dr",
            "tldr",
            "just tell me quick",
            "fast answer"
        ]
    ):

        return "quick"


    if any(
        x in text
        for x in [
            "detailed",
            "in detail",
            "in depth",
            "indepth",
            "technical",
            "thorough",
            "research",
            "pathogen",
            "mechanism",
            "scientific",
            "explain fully"
        ]
    ):

        return "detailed"


    if any(
        x in text
        for x in [
            "simple",
            "simply",
            "easy way",
            "in simple words",
            "beginner",
            "easy explanation",
            "plain language"
        ]
    ):

        return "simple"


    return None


# ============================================================
# HOME RESPONSE: ABOUT THE APP
# ============================================================

def response_about_app():

    return (
        "🌿 **Plant Health Assistant**\n\n"

        "I'm the assistant built into Plant Disease AI. "
        "Here's what this app does, end to end:\n\n"

        "📷 **Scan** - upload a clear leaf photo and an AI model "
        "screens it for disease across Tomato, Chilli and "
        "Hibiscus.\n\n"

        "🔍 **Diagnose** - you get the predicted disease with a "
        "confidence score.\n\n"

        "💊 **Treatment** - tailored treatment/control guidance, "
        "medicine info and safety notes for the result.\n\n"

        "💬 **Chat (me!)** - ask about symptoms, causes, "
        "treatment, prevention or plant care anytime, with or "
        "without a scan.\n\n"

        "🌱 **Plant Care** - everyday watering, sunlight, soil, "
        "fertilizer and pruning guidance, disease aside.\n\n"

        "📊 **Track Recovery** - save a BEFORE scan and later compare "
        "the AFTER scan using the same ResNet50 model.\n\n"

        "📋 **My Records** - review your past scans anytime.\n\n"

        "Just ask me anything about plant diseases, treatment or "
        "care - I'll do my best to answer directly."
    )


# ============================================================
# HOME RESPONSE: SUPPORTED DISEASES
# ============================================================

def response_supported_diseases():

    disease_names = list(
        index_to_class.values()
    )


    chilli = []
    hibiscus = []
    tomato = []


    for disease in disease_names:

        clean = clean_label(
            disease
        )

        lower = clean.lower()


        if "chilli" in lower:

            chilli.append(
                clean
            )

        elif (
            "hibiscus" in lower
            or
            lower in [
                "leaf spot",
                "mealybug infestation",
                "powdery mildew",
                "rust",
                "sooty mold",
                "whitefly infestation",
                "botrytis blightgray mold"
            ]
        ):

            hibiscus.append(
                clean
            )

        elif (
            "tomato" in lower
            or
            lower in [
                "bacterial spot",
                "early blight",
                "late blight",
                "leaf mold",
                "septoria leaf spot",
                "target spot"
            ]
        ):

            tomato.append(
                clean
            )


    if not chilli:

        chilli = [
            "Chilli Cercospora",
            "Chilli Mites and Thrips",
            "Chilli Nutritional Deficiency",
            "Healthy Chilli"
        ]


    if not hibiscus:

        hibiscus = [
            "Botrytis Blight / Gray Mold",
            "Healthy Hibiscus",
            "Leaf Spot",
            "Mealybug Infestation",
            "Powdery Mildew",
            "Rust",
            "Sooty Mold",
            "Whitefly Infestation"
        ]


    if not tomato:

        tomato = [
            "Bacterial Spot",
            "Early Blight",
            "Healthy Tomato",
            "Late Blight",
            "Leaf Mold",
            "Septoria Leaf Spot",
            "Target Spot",
            "Tomato Mosaic Virus",
            "Tomato Yellow Leaf Curl Virus"
        ]


    response = (
        "🌿 **Your Plant Disease AI can diagnose these supported plants:**\n\n"

        "🌺 **Hibiscus**\n"
    )


    for disease in hibiscus:

        response += (
            f"• {disease}\n"
        )


    response += (
        "\n🌶️ **Chilli**\n"
    )


    for disease in chilli:

        response += (
            f"• {disease}\n"
        )


    response += (
        "\n🍅 **Tomato**\n"
    )


    for disease in tomato:

        response += (
            f"• {disease}\n"
        )


    response += (
        "\n📷 **How to diagnose:**\n"
        "Upload a clear photo of the affected leaf or plant. "
        "The trained disease model will analyze the image and "
        "show the predicted result, confidence and treatment guidance.\n\n"

        "💡 For the best result, use a clear, well-lit image "
        "with the affected leaf visible."
    )


    return response


# ============================================================
# HOME RESPONSE: GENERAL PREVENTION
# ============================================================

def response_general_prevention():

    return (
        "🛡️ **How to prevent plant diseases**\n\n"

        "🌱 **1. Inspect plants regularly**\n"
        "Check leaves, stems and new growth for spots, insects, "
        "discoloration or unusual growth.\n\n"

        "💧 **2. Water correctly**\n"
        "Avoid excessive watering and avoid keeping the soil "
        "constantly waterlogged.\n\n"

        "☀️ **3. Give suitable sunlight**\n"
        "Make sure the plant receives the amount of light it needs "
        "for healthy growth.\n\n"

        "🍃 **4. Maintain good airflow**\n"
        "Avoid overcrowding plants. Good air circulation helps "
        "reduce conditions that favour some diseases.\n\n"

        "🧹 **5. Remove infected material**\n"
        "Remove severely affected leaves or plant debris where "
        "appropriate and keep the growing area clean.\n\n"

        "🪴 **6. Keep tools clean**\n"
        "Clean pruning and gardening tools regularly, especially "
        "after working on an affected plant.\n\n"

        "🐛 **7. Watch for pests**\n"
        "Check regularly for mealybugs, whiteflies, mites, thrips "
        "and other pests.\n\n"

        "🌿 **8. Avoid unnecessary chemical use**\n"
        "Use plant-protection products only when appropriate and "
        "always follow the product label.\n\n"

        "🔎 **Best practice:** Early detection is one of the most "
        "important parts of plant disease management."
    )


# ============================================================
# HOME RESPONSE: GENERAL CARE
# ============================================================

def response_general_care():

    return (
        "🌱 **General Plant Care Guide**\n\n"

        "💧 **Watering**\n"
        "Water according to the plant's needs and growing conditions. "
        "Avoid both severe drying and prolonged waterlogging.\n\n"

        "☀️ **Sunlight**\n"
        "Give the plant suitable light for healthy growth. "
        "Avoid sudden changes from shade to very strong sunlight.\n\n"

        "🪴 **Soil**\n"
        "Use well-draining soil appropriate for the plant. "
        "Good drainage helps prevent root-related problems.\n\n"

        "🌱 **Fertilizer**\n"
        "Use a suitable balanced fertilizer according to the plant's "
        "growth stage. Avoid excessive fertilizer application.\n\n"

        "🍃 **Pruning**\n"
        "Remove dead, damaged or severely affected plant material "
        "with clean tools where appropriate.\n\n"

        "🐛 **Pest checking**\n"
        "Inspect the underside of leaves and young shoots regularly "
        "for insects and other pests.\n\n"

        "🧹 **Clean surroundings**\n"
        "Remove fallen infected leaves and plant debris and keep "
        "the growing area clean.\n\n"

        "📅 **Regular monitoring**\n"
        "A quick inspection every few days can help you notice "
        "problems before they become severe.\n\n"

        "🌿 Supported plants in this app: "
        "**Hibiscus, Chilli and Tomato.**"
    )


# ============================================================
# HOME RESPONSE: FIRST ACTION
# ============================================================

def response_general_first_action():

    return (
        "🚨 **What should you do first if your plant is infected?**\n\n"

        "1️⃣ **Inspect the plant carefully**\n"
        "Look at the affected leaves, stems and nearby growth.\n\n"

        "2️⃣ **Separate the affected plant if possible**\n"
        "Keep it away from healthy plants when practical.\n\n"

        "3️⃣ **Remove severely affected material**\n"
        "Remove badly damaged leaves or plant parts where appropriate "
        "and dispose of them safely.\n\n"

        "4️⃣ **Check nearby plants**\n"
        "Look for the same symptoms on surrounding plants.\n\n"

        "5️⃣ **Avoid random treatments**\n"
        "Do not immediately mix several pesticides or fungicides.\n\n"

        "6️⃣ **Identify the problem**\n"
        "Use the Plant Disease AI scanner by uploading a clear photo "
        "of the affected leaf.\n\n"

        "📷 The scanner supports **Hibiscus, Chilli and Tomato**.\n\n"

        "⚠️ The image result is a screening aid. If symptoms are "
        "severe, rapidly spreading or unclear, seek appropriate "
        "local agricultural or plant-health advice."
    )


# ============================================================
# HOME RESPONSE: GENERAL TREATMENT INFO
# (no specific disease named - explains how treatment works
# in this app, plus general safe-use guidance)
# ============================================================

def response_general_treatment_info():

    return (
        "🧪 **How treatment works in this app**\n\n"

        "💊 **After a scan** - upload a leaf photo on the home page "
        "and the app gives the predicted disease plus tailored "
        "treatment/control steps and safety notes for that exact "
        "result.\n\n"

        "✍️ **Or ask me directly** - name a specific problem here, "
        "for example \"tomato early blight treatment\" or \"powdery "
        "mildew treatment\", and I'll give you the treatment info "
        "right away without needing a scan.\n\n"

        "🌿 **General treatment principles**\n\n"

        "⚠️ Always use products labelled for the identified crop "
        "and problem, and follow the label for rate and safety.\n\n"

        "🚫 Don't randomly mix multiple chemical products together.\n\n"

        "🧴 Where possible, start with the least aggressive option "
        "(pruning affected material, improving airflow, organic or "
        "biological controls) before moving to stronger chemical "
        "treatments.\n\n"

        "🔁 Re-check the plant after treatment - if there's no "
        "improvement, the diagnosis or product may need to be "
        "reassessed.\n\n"

        "🔎 Supported problems include Early Blight, Late Blight, "
        "Bacterial Spot, Leaf Mold, Septoria Leaf Spot, Target Spot, "
        "Mosaic Virus, Yellow Leaf Curl Virus, Cercospora, Mites & "
        "Thrips, Powdery Mildew, Rust, Leaf Spot, Gray Mold, "
        "Mealybug, Whitefly and Sooty Mold across Tomato, Chilli "
        "and Hibiscus."
    )


# ============================================================
# HOME RESPONSE: BEFORE/AFTER RECOVERY TRACKING
# ============================================================

def response_general_tracking():

    return (
        "📊 **Tracking your treatment progress with Before/After**\n\n"

        "This app has a dedicated **Before/After Treatment Recovery** "
        "tool built for exactly this — here's how it works:\n\n"

        "1️⃣ **Save a BEFORE image** - after scanning a plant, press "
        "**'Save as BEFORE Image'** on the result page. This stores "
        "that scan (photo + predicted disease + confidence) as your "
        "starting point.\n\n"

        "2️⃣ **Treat the plant** - follow the treatment/control steps "
        "given for that result.\n\n"

        "3️⃣ **Go to Before/After** and upload a fresh **AFTER** photo "
        "of the same plant or leaf, taken from a similar angle, "
        "distance and lighting as the BEFORE photo.\n\n"

        "4️⃣ **The same ResNet50 model analyzes both images "
        "independently** and the app compares the two results for "
        "you, showing:\n"
        "   • Whether the condition is the same or has changed\n"
        "   • The confidence change between BEFORE and AFTER\n"
        "   • A status (improving / worsening / no clear change / "
        "disease-change warning)\n"
        "   • A written recommendation for what to do next\n\n"

        "5️⃣ **Your comparison is saved** to your account under "
        "'My Comparison History' on the Before/After page, so you "
        "can review it anytime — and delete it later if you no "
        "longer need it.\n\n"

        "🔎 If the AFTER scan shows a different disease than BEFORE, "
        "the app will warn you — that can mean a new problem, "
        "treatment not working yet, or just a difference in photo "
        "quality, so it's worth rescanning before changing treatment.\n\n"

        "📷 Haven't scanned anything yet? Start with a leaf photo on "
        "the home page, then look for **'Save as BEFORE Image'** on "
        "the result page to begin tracking that plant's recovery."
    )


# ============================================================
# STYLE INTRO
# ============================================================

def style_intro(
    mentality,
    topic
):

    if mentality == "simple":

        return (
            f"🏡 In simple terms, here's **{topic}**:"
        )


    if mentality == "quick":

        return (
            f"⚡ Quick answer on **{topic}**:"
        )


    if mentality == "detailed":

        return (
            f"🔬 Here's a detailed, structured breakdown "
            f"of **{topic}**:"
        )


    return (
        f"😊 Here's what you need to know about **{topic}**."
    )


# ============================================================
# DISEASE RESPONSE FUNCTIONS
# ============================================================

def response_information(
    ctx,
    mentality
):

    description = (
        ctx["description"]
        or
        "The treatment database does not contain "
        "a detailed description."
    )


    return (
        f"{style_intro(mentality, ctx['disease'])}\n\n"

        f"📋 **What is it?**\n"
        f"{description}\n\n"

        f"🎯 **Prediction confidence:** "
        f"{ctx['confidence']}%\n\n"

        "⚠️ This is an image-based screening result. "
        "The actual plant should be checked against "
        "the visible symptoms."
    )


def response_reassurance(
    ctx,
    mentality
):

    if ctx["confidence"] < 60:

        confidence_text = (
            "The current prediction confidence is relatively "
            "low, so the visible symptoms should be checked carefully."
        )

    elif ctx["confidence"] < 85:

        confidence_text = (
            "The prediction has moderate confidence. "
            "Compare the plant symptoms before taking major action."
        )

    else:

        confidence_text = (
            "The prediction has high confidence for this image "
            "class, but the percentage does not replace plant inspection."
        )


    return (
        "😌 **Don't panic — let's handle it step by step.**\n\n"

        f"🌿 Current result: **{ctx['disease']}**\n"
        f"🎯 Confidence: **{ctx['confidence']}%**\n\n"

        f"{confidence_text}\n\n"

        "🔍 Check the affected leaves carefully.\n"
        "✂️ Remove severely affected material where appropriate.\n"
        "🌱 Keep healthy plants separated from affected material.\n"
        "🚫 Avoid randomly mixing several products."
    )


def response_first_action(
    ctx,
    mentality
):

    steps = (
        ctx["first_steps"]
        or
        "Inspect the affected leaves, maintain sanitation, "
        "remove severely affected material where appropriate, "
        "and monitor nearby plants."
    )


    return (
        f"🚨 **First steps for {ctx['disease']}**\n\n"
        f"{steps}\n\n"
        "👀 Monitor nearby leaves and plants for similar symptoms."
    )


def response_cause(
    ctx,
    mentality
):

    cause = (
        ctx["why"]
        or
        "A detailed cause is not available in "
        "the treatment database."
    )


    return (
        f"❓ **Why can {ctx['disease']} happen?**\n\n"
        f"{cause}\n\n"
        "🌱 Growing conditions can also influence how quickly "
        "the problem develops."
    )


def response_symptoms(
    ctx,
    mentality
):

    symptoms = (
        ctx["symptoms"]
        or
        "No detailed symptom list is available "
        "in the treatment database."
    )


    return (
        f"🔍 **Symptoms of {ctx['disease']}**\n\n"
        f"{symptoms}\n\n"
        "⚠️ Some symptoms can overlap with other plant problems, "
        "so inspect the plant carefully."
    )


def response_treatment(
    ctx,
    mentality
):

    treatment = (
        ctx["treatment"]
        or
        "No specific treatment information is available "
        "for this result."
    )


    response = (
        f"🧪 **Treatment / Control for {ctx['disease']}**\n\n"
        f"{treatment}"
    )


    if ctx["medicine"]:

        response += (
            "\n\n🧴 **Medicine / Product Information**\n\n"
            f"{ctx['medicine']}"
        )


    if ctx["safety"]:

        response += (
            "\n\n⚠️ **Safety**\n\n"
            f"{ctx['safety']}"
        )

    else:

        response += (
            "\n\n⚠️ Use only products labelled for the "
            "identified crop and problem. Follow the product "
            "label for application rate and safety instructions."
        )


    response += (
        "\n\n🚫 Do not randomly mix multiple products."
    )


    return response


def response_home_care(
    ctx,
    mentality
):

    home = (
        ctx["home_care"]
        or
        "Keep the plant area clean, maintain suitable growing "
        "conditions, monitor regularly, and remove severely "
        "affected material where appropriate."
    )


    return (
        f"🏡 **Home Care for {ctx['disease']}**\n\n"
        f"{home}\n\n"
        "⚠️ Home-care measures are supportive and are not "
        "guaranteed to cure an established problem."
    )


def response_prevention(
    ctx,
    mentality
):

    prevention = (
        ctx["prevention"]
        or
        "Maintain sanitation, suitable watering, good airflow "
        "and regular plant inspection."
    )


    return (
        f"🛡️ **Preventing {ctx['disease']}**\n\n"
        f"{prevention}\n\n"
        "👀 Regularly inspect new and nearby leaves."
    )


def response_confidence(
    ctx,
    mentality
):

    if ctx["confidence"] >= 85:

        level = "HIGH"

    elif ctx["confidence"] >= 60:

        level = "MODERATE"

    else:

        level = "LOW"


    return (
        "🎯 **Prediction Confidence**\n\n"
        f"Confidence: **{ctx['confidence']}%**\n"
        f"Level: **{level}**\n\n"

        "The confidence score indicates how strongly the "
        "prediction system favoured the selected class for "
        "the uploaded image.\n\n"

        "⚠️ It should not be interpreted as laboratory "
        "confirmation."
    )


def response_severity(
    ctx,
    mentality
):

    return (
        f"⚠️ **Severity Guidance for {ctx['disease']}**\n\n"

        "Monitor how quickly the symptoms are changing and "
        "whether nearby leaves or plants are becoming affected.\n\n"

        "🔎 Rapid spread, extensive damage or unclear symptoms "
        "are reasons to seek appropriate plant-health advice."
    )


def response_expert(
    ctx,
    mentality
):

    expert = (
        ctx["expert_help"]
        or
        "Consider local agricultural advice when symptoms are "
        "severe, rapidly spreading, unclear, or seriously affecting "
        "the plant."
    )


    return (
        f"👨‍🌾 **Expert Help for {ctx['disease']}**\n\n"
        f"{expert}\n\n"

        "📌 Expert confirmation is especially useful when the "
        "visible symptoms do not clearly match the predicted result."
    )


def response_mentality(
    mentality
):

    info = MENTALITIES.get(
        mentality,
        MENTALITIES["friendly"]
    )


    return (
        "🧠 **Current Chat Style**\n\n"
        f"{info['name']}\n\n"
        f"Style: **{info['style']}**"
    )


def response_greeting(
    ctx,
    mentality
):

    return (
        "👋 **Hello! 🌿**\n\n"

        "I am your Plant Health Assistant.\n\n"

        "🌺 Hibiscus\n"
        "🌶️ Chilli\n"
        "🍅 Tomato\n\n"

        "You can ask me about:\n"
        "🔎 Disease identification\n"
        "🩺 Symptoms\n"
        "🧪 Treatment\n"
        "🛡️ Prevention\n"
        "💧 Plant care\n"
        "🚨 What to do first\n\n"

        "📷 You can also upload a plant image "
        "for disease screening."
    )


def response_general(
    ctx,
    mentality
):

    return (
        "🌿 **Plant Health Assistant**\n\n"

        "I can help you with:\n\n"

        "🔎 Which diseases this app can identify\n"
        "🩺 Symptoms and disease information\n"
        "🧪 Treatment and control\n"
        "🛡️ Disease prevention\n"
        "🌱 General plant care\n"
        "🚨 What to do first when a plant is infected\n\n"

        "Supported plants:\n"
        "🌺 Hibiscus\n"
        "🌶️ Chilli\n"
        "🍅 Tomato\n\n"

        "📷 Upload a clear plant photo when you want "
        "disease screening."
    )


# ============================================================
# NO-DISEASE-CONTEXT GUIDANCE
#
# Used ONLY when the chat has no diagnosed/named disease to work
# with at all (e.g. opened from the home page with nothing scanned
# yet, and the message doesn't name a specific disease either).
# This does not change any response when a disease IS present -
# response_treatment / response_symptoms / etc. above are untouched
# and still run exactly as before for real diagnoses.
# ============================================================

DISEASE_REQUIRED_INTENTS = {

    "information",
    "reassurance",
    "cause",
    "symptoms",
    "treatment",
    "first_action",
    "home_care",
    "prevention",
    "severity",
    "expert"
}


NO_DISEASE_TOPIC_LABELS = {

    "information":
        "📋 **Disease information**",

    "reassurance":
        "😌 **Let's figure this out**",

    "cause":
        "❓ **Cause guidance**",

    "symptoms":
        "🔍 **Symptom guidance**",

    "treatment":
        "🧪 **Treatment guidance**",

    "first_action":
        "🚨 **First steps guidance**",

    "home_care":
        "🏡 **Home care guidance**",

    "prevention":
        "🛡️ **Prevention guidance**",

    "severity":
        "⚠️ **Severity guidance**",

    "expert":
        "👨‍🌾 **Expert help guidance**"
}


def response_no_disease_context(intent):

    label = NO_DISEASE_TOPIC_LABELS.get(
        intent,
        "🌿 **Plant Health Assistant**"
    )

    return (
        f"{label}\n\n"

        "I don't have a specific diagnosed disease to go on yet, "
        "so here's how to get a precise answer:\n\n"

        "📷 **Scan a leaf** - upload a clear photo on the home "
        "page and I'll pull up the exact treatment, symptoms and "
        "prevention info for the result.\n\n"

        "✍️ **Or just name it here** - for example \"tomato early "
        "blight treatment\" or \"powdery mildew on hibiscus\", and "
        "I'll answer directly.\n\n"

        "🔎 Supported problems include Early Blight, Late Blight, "
        "Bacterial Spot, Leaf Mold, Septoria Leaf Spot, Target Spot, "
        "Mosaic Virus, Yellow Leaf Curl Virus, Cercospora, Mites & "
        "Thrips, Powdery Mildew, Rust, Leaf Spot, Gray Mold, "
        "Mealybug, Whitefly and Sooty Mold across Tomato, Chilli "
        "and Hibiscus.\n\n"

        "🌱 For everyday (non-disease) watering, sunlight, soil and "
        "fertilizer guidance, open the **Plant Care** page from the "
        "home screen."
    )


def response_restricted_to_disease(disease, treatment, off_topic_disease=None):

    label = get_display_name(
        disease,
        treatment
    ) if disease else None

    intro = (
        f"🌿 This chat is focused on your **detected diagnosis"
        + (f" ({label})" if label else "")
        + "**.\n\n"
    )

    if off_topic_disease:

        intro += (
            f"I can only answer questions about {label or 'this result'} "
            "here — not about a different disease.\n\n"
        )

    return (
        intro +

        "For general plant-care questions, other diseases, or "
        "app-wide help, please use the main **Plant Health "
        "Assistant** chat from the home page.\n\n"

        "Here, ask me anything about your current result instead — "
        "treatment, symptoms, causes, prevention, or what to do "
        "first. 🌱"
    )


# ============================================================
# "QUICK" MENTALITY - CONDENSE A FULL ANSWER
# ============================================================

def condense_for_quick(text):

    """
    Keeps only the heading line and the first real content line
    (skipping blank lines), so "quick" mentality gives a short,
    to-the-point answer instead of the full multi-section reply.
    """

    text = str(
        text or ""
    )

    lines = [
        line
        for line in text.split("\n")
        if line.strip()
    ]

    if not lines:
        return text


    kept = lines[:2]

    condensed = "\n\n".join(
        kept
    )

    condensed += (
        "\n\n💬 Ask me for more detail anytime, "
        "or switch to Detailed mode."
    )

    return condensed


# ============================================================
# CHAT RESPONSE ENGINE
# ============================================================

def generate_chat_response(
    message,
    disease,
    confidence,
    treatment,
    mentality,
    restrict_to_disease=False
):

    # ========================================================
    # HOME QUESTIONS FIRST
    #
    # On the treatment/result page (restrict_to_disease=True),
    # general app-wide questions are redirected to the home page
    # chat instead of being answered here - this chat is scoped
    # to the detected disease only. The home page chat always
    # passes restrict_to_disease=False and keeps answering
    # everything exactly as before.
    # ========================================================

    home_intent = detect_home_intent(
        message,
        has_disease_context=bool(disease)
    )


    if home_intent and restrict_to_disease:

        return response_restricted_to_disease(
            disease,
            treatment
        )


    if home_intent == "about_app":

        return response_about_app()


    if home_intent == "supported_diseases":

        return response_supported_diseases()


    if home_intent == "general_prevention":

        return response_general_prevention()


    if home_intent == "general_care":

        return response_general_care()


    if home_intent == "general_first_action":

        return response_general_first_action()


    if home_intent == "general_treatment_info":

        return response_general_treatment_info()


    if home_intent == "general_tracking":

        return response_general_tracking()


    # ========================================================
    # DETECT PLANT / DISEASE
    # ========================================================

    message_disease = detect_disease_from_message(
        message
    )

    message_plant = detect_plant_from_message(
        message
    )


    # ========================================================
    # RESTRICTED CHAT: block questions about a DIFFERENT disease
    #
    # A disease named in the message that doesn't match the
    # page's own detected disease is off-topic for this chat -
    # redirect to the home page instead of silently answering
    # about the other disease.
    # ========================================================

    if (
        restrict_to_disease
        and
        message_disease
        and
        disease
        and
        compact_text(message_disease) != compact_text(disease)
    ):

        return response_restricted_to_disease(
            disease,
            treatment,
            off_topic_disease=message_disease
        )


    # ========================================================
    # ACTIVE DISEASE
    # ========================================================

    active_disease = (
        message_disease
        if message_disease
        else disease
    )


    active_treatment = find_treatment(
        active_disease
    )


    # ========================================================
    # PLANT-ONLY QUESTIONS
    # ========================================================
    # Use the detailed chat knowledge BEFORE the old generic care reply.
    if message_plant and not message_disease:
        text = normalize_text(message)
        plant_knowledge_answer = match_chat_knowledge(message, mentality)
        if plant_knowledge_answer:
            return plant_knowledge_answer
        plant_data = PLANT_CARE_DATA.get(message_plant, {})
        source_data = plant_data.get("source_data", {}) if isinstance(plant_data, dict) else {}
        care_words = ["care","water","watering","sunlight","light","fertilizer","fertiliser","npk","soil","pruning","temperature","heat","schedule","warning","safety"]
        if any(word in text for word in care_words) and source_data:
            lines = [f"{SUPPORTED_PLANTS[message_plant]['display']} **care guidance**", ""]
            keys=[]
            if any(x in text for x in ["water","watering"]): keys.append("watering")
            if any(x in text for x in ["sunlight","light"]): keys.append("sunlight")
            if any(x in text for x in ["temperature","heat"]): keys.append("temperature")
            if "soil" in text: keys.append("soil")
            if any(x in text for x in ["fertilizer","fertiliser","npk"]): keys.append("fertilizer")
            if "pruning" in text: keys.append("pruning")
            if not keys: keys=["watering","sunlight","temperature","soil","fertilizer","pruning"]
            for key in keys:
                section=source_data.get(key,{})
                if not isinstance(section,dict): continue
                title=section.get("title") or key.replace("_"," ").title()
                if section.get("summary"): lines += [f"**{title}**",str(section["summary"])]
                guidance=section.get("guidance",section.get("bullets",[]))
                if isinstance(guidance,list): lines += [f"• {x}" for x in guidance]
                elif guidance: lines.append(f"• {guidance}")
                for label,field in [("Frequency","frequency"),("Ideal Range","ideal_range"),("Recommended","recommended"),("Recommended Mix","recommended_mix"),("Schedule","schedule")]:
                    if section.get(field): lines.append(f"📌 **{label}:** {section[field]}")
                if section.get("warning"): lines.append(f"⚠️ {section['warning']}")
                lines.append("")
            return "\n".join(lines).strip()
        if any(x in text for x in ["treatment","treat","medicine","fungicide","pesticide","insecticide","product","products","spray","cure","kill","get rid of","remedy","solution"]):
            return f"{SUPPORTED_PLANTS[message_plant]['display']} **treatment guidance**\n\n🌿 Please upload a clear image of the affected leaf so the disease scanner can identify the possible problem.\n\n📷 After detection, the app will provide the available treatment and control information for the predicted disease."

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    ctx = build_context(
        active_disease,
        confidence,
        active_treatment
    )


    intent = detect_intent(
        message,
        has_disease_context=bool(active_disease)
    )


    detected_mentality = detect_message_mentality(
        message
    )


    if (
        mentality == "friendly"
        and
        detected_mentality
    ):

        effective_mentality = detected_mentality

    else:

        effective_mentality = mentality


    if effective_mentality not in MENTALITIES:

        effective_mentality = "friendly"


    print(
        "💬 CHAT",
        "| intent:",
        intent,
        "| mode:",
        effective_mentality,
        "| plant:",
        message_plant,
        "| disease:",
        active_disease
    )


    # ========================================================
    # NO DISEASE CONTEXT AT ALL
    #
    # active_disease is empty only when: the message didn't name
    # a disease AND the chat wasn't opened with one either (e.g.
    # home page chat with nothing scanned yet). In that case,
    # disease-specific intents (treatment/symptoms/cause/etc.)
    # would otherwise fall through to response_treatment() and
    # friends showing a confusing "for Unknown" answer. Give a
    # helpful redirect instead.
    #
    # This does NOT run when a disease is present (from a scan
    # result or named in the message) - response_treatment() and
    # the other disease responses below are completely unchanged
    # for that case.
    # ========================================================

    if (
        not active_disease
        and
        intent in DISEASE_REQUIRED_INTENTS
    ):

        return response_no_disease_context(
            intent
        )


    # ========================================================
    # RESPONSE ROUTING
    #
    # Collected into `result` instead of returning immediately,
    # so "quick" mentality can condense the final answer below
    # regardless of which branch produced it.
    # ========================================================

    if intent == "greeting":

        result = response_greeting(
            ctx,
            effective_mentality
        )


    elif intent == "information":

        result = response_information(
            ctx,
            effective_mentality
        )


    elif intent == "reassurance":

        result = response_reassurance(
            ctx,
            effective_mentality
        )


    elif intent == "cause":

        result = response_cause(
            ctx,
            effective_mentality
        )


    elif intent == "symptoms":

        result = response_symptoms(
            ctx,
            effective_mentality
        )


    elif intent == "first_action":

        result = response_first_action(
            ctx,
            effective_mentality
        )


    elif intent == "treatment":

        result = response_treatment(
            ctx,
            effective_mentality
        )


    elif intent == "home_care":

        result = response_home_care(
            ctx,
            effective_mentality
        )


    elif intent == "prevention":

        result = response_prevention(
            ctx,
            effective_mentality
        )


    elif intent == "confidence":

        result = response_confidence(
            ctx,
            effective_mentality
        )


    elif intent == "severity":

        result = response_severity(
            ctx,
            effective_mentality
        )


    elif intent == "expert":

        result = response_expert(
            ctx,
            effective_mentality
        )


    elif intent == "mentality":

        result = response_mentality(
            effective_mentality
        )


    else:

        # --------------------------------------------------------
        # CHAT KNOWLEDGE BASE FALLBACK
        #
        # Nothing above matched a specific hand-built intent.
        # Before giving up with the generic app intro, check the
        # broader chat_knowledge.json pattern set (weather,
        # watering mistakes, farmer/student framing, container
        # plants, seedlings, flowering, records, etc.) - this only
        # ever adds coverage, it never overrides a disease-specific
        # answer since those are already handled above.
        # --------------------------------------------------------

        knowledge_answer = match_chat_knowledge(
            message,
            effective_mentality
        )

        if knowledge_answer:

            result = knowledge_answer

        else:

            result = response_general(
                ctx,
                effective_mentality
            )


    if effective_mentality == "quick":

        result = condense_for_quick(
            result
        )


    return result


# ============================================================
# INJECT MOBILE STATE SCRIPT INTO HTML PAGES
# ============================================================

@app.after_request
def add_mobile_state_script(response):
    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type.lower() and response.status_code == 200 and not response.direct_passthrough:
        try:
            html = response.get_data(as_text=True)
            if "plant_ai_last_page_v1" not in html and "</body>" in html.lower():
                i = html.lower().rfind("</body>")
                response.set_data(html[:i] + MOBILE_STATE_SCRIPT + html[i:])
        except Exception as error:
            print("⚠️ Mobile state injection skipped:", error)
    return response


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    # /predict is the image-upload action endpoint. It must normally
    # receive POST data from the scan form. A GET can happen when a
    # mobile browser restores an old URL from history, so safely send
    # the user back to the Scan/Home screen instead of returning 405.
    if request.method == "GET":
        return redirect(url_for("home"))

    if "image" not in request.files:

        return render_template(
            "index.html",
            error="❌ Please select an image."
        )


    file = request.files["image"]


    if (
        not file
        or
        file.filename == ""
    ):

        return render_template(
            "index.html",
            error="❌ No image selected."
        )


    if not allowed_file(
        file.filename
    ):

        return render_template(
            "index.html",
            error="❌ Unsupported image format."
        )


    original_filename = secure_filename(
        file.filename
    )


    extension = os.path.splitext(
        original_filename
    )[1].lower()


    unique_filename = (
        uuid.uuid4().hex
        +
        extension
    )


    saved_path = os.path.join(
        UPLOAD_FOLDER,
        unique_filename
    )


    try:

        file.save(
            saved_path
        )


        (
            disease,
            confidence,
            confidence_level
        ) = predict_disease(
            saved_path
        )


    except Exception as error:

        print(
            "❌ Prediction error:",
            error
        )


        if os.path.exists(
            saved_path
        ):

            try:

                os.remove(
                    saved_path
                )

            except OSError:

                pass


        return render_template(
            "index.html",
            error=(
                "❌ Image prediction failed. "
                "Please try another clear image."
            )
        )


    treatment = find_treatment(
        disease
    )


    save_collection_record(
        original_filename,
        unique_filename,
        disease,
        confidence,
        confidence_level
    )


    image_url = (
        "/static/uploads/"
        +
        unique_filename
    )


    display_name = get_display_name(
        disease,
        treatment
    )


    # --------------------------------------------------------
    # FIX:
    # Determine plant from predicted disease
    # --------------------------------------------------------

    plant_name = detect_plant_from_disease(
        disease
    )


    if not plant_name:

        plant_name = "unknown"


    # --------------------------------------------------------
    # FIX:
    # result.html's "Plant Care & Recovery Plan" section reads
    # from a `plant_care` variable (sections / prevention), but
    # this route never used to pass one, so that whole section
    # silently rendered blank. Reuse the same PLANT_CARE_DATA
    # lookup that /plant-care uses so it actually shows up here.
    # --------------------------------------------------------

    plant_care_content = PLANT_CARE_DATA.get(
        plant_name,
        {}
    )


    print()
    print("=" * 60)
    print("🌱 PLANT DISEASE RESULT")
    print("=" * 60)

    print(
        "Image      :",
        original_filename
    )

    print(
        "Class      :",
        disease
    )

    print(
        "Display    :",
        display_name
    )

    print(
        "Confidence :",
        confidence,
        "%"
    )

    print(
        "Level      :",
        confidence_level
    )

    print(
        "Plant      :",
        plant_name
    )

    print(
        "Treatment  :",
        "FOUND"
        if treatment
        else
        "NOT FOUND"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # HEALTHY CLASS HANDLING
    # Healthy predictions are valid model results even when
    # there is no disease-treatment entry in the treatment DB.
    # --------------------------------------------------------
    normalized_health_class = str(
        disease or ""
    ).strip().lower().replace(
        "healthy_chili",
        "healthy_chilli"
    )

    is_healthy = normalized_health_class in {
        "healthy_tomato",
        "healthy_chilli",
        "healthy_hibiscus"
    }

    return render_template(
        "result.html",

        disease=disease,

        display_name=display_name,

        confidence=confidence,

        confidence_level=confidence_level,

        image_url=image_url,

        original_filename=original_filename,

        treatment=treatment,

        plant_name=plant_name,

        saved_filename=unique_filename,

        plant_care=plant_care_content,

        is_healthy=is_healthy,

        current_user=get_current_user()
    )


# ============================================================
# PLANT CARE PAGE
# ============================================================

@app.route(
    "/plant-care",
    methods=["GET"]
)
def plant_care_page():

    plant = request.args.get(
        "plant",
        ""
    ).strip().lower()


    return_to = request.args.get(
        "return_to",
        "/"
    )


    # --------------------------------------------------------
    # No plant selected yet - plant_care.html already has a
    # built-in "choose your plant" picker for this case
    # ({% if not plant_name %} branch), so we just render it
    # with plant_name empty instead of redirecting home.
    # --------------------------------------------------------

    plant_display = {

        "hibiscus":
            "🌺 Hibiscus",

        "chilli":
            "🌶️ Chilli",

        "tomato":
            "🍅 Tomato"

    }.get(
        plant,
        plant.title() if plant else ""
    )


    plant_care_content = PLANT_CARE_DATA.get(
        plant,
        {}
    )


    # --------------------------------------------------------
    # If your existing plant_care.html exists,
    # use it without changing your UI.
    # --------------------------------------------------------

    template_path = os.path.join(
        BASE_DIR,
        "templates",
        "plant_care.html"
    )


    if os.path.exists(
        template_path
    ):

        return render_template(
            "plant_care.html",

            plant_name=plant_display,

            plant=plant,

            plant_care=plant_care_content,

            return_to=return_to,

            # Needed by plant_care.html's footer nav, which checks
            # "'upload_history' in view_functions" - without this,
            # that line raises a Jinja UndefinedError and the whole
            # page fails to render.
            view_functions=app.view_functions
        )


    # --------------------------------------------------------
    # Fallback page
    # --------------------------------------------------------

    return (
        """
        <!DOCTYPE html>
        <html lang="en">

        <head>

            <meta charset="UTF-8">

            <meta
                name="viewport"
                content="width=device-width, initial-scale=1.0"
            >

            <title>Plant Care</title>

            <style>

                * {
                    box-sizing: border-box;
                }

                body {

                    margin: 0;

                    padding: 30px;

                    font-family:
                        Arial,
                        sans-serif;

                    background:
                        #f1f8f4;

                    color:
                        #183b2a;
                }

                .card {

                    max-width:
                        700px;

                    margin:
                        30px auto;

                    background:
                        white;

                    padding:
                        30px;

                    border-radius:
                        20px;

                    box-shadow:
                        0 10px 30px
                        rgba(0,0,0,0.08);
                }

                h1 {

                    color:
                        #176b43;
                }

                .item {

                    margin:
                        15px 0;

                    padding:
                        15px;

                    background:
                        #f5faf7;

                    border-radius:
                        12px;
                }

                .back {

                    display:
                        inline-block;

                    margin-top:
                        20px;

                    padding:
                        12px 18px;

                    background:
                        #176b43;

                    color:
                        white;

                    text-decoration:
                        none;

                    border-radius:
                        10px;
                }

            </style>

        </head>

        <body>

            <div class="card">

                <h1>
                    """
        + plant_display
        + """
                    Plant Care 🌱
                </h1>

                <div class="item">
                    💧 <strong>Watering</strong><br>
                    Water according to the plant's needs
                    and avoid prolonged waterlogging.
                </div>

                <div class="item">
                    ☀️ <strong>Sunlight</strong><br>
                    Provide suitable sunlight for healthy growth.
                </div>

                <div class="item">
                    🪴 <strong>Soil</strong><br>
                    Use suitable, well-draining soil.
                </div>

                <div class="item">
                    🌱 <strong>Fertilizer</strong><br>
                    Use suitable fertilizer according to
                    the plant's growth stage.
                </div>

                <div class="item">
                    🍃 <strong>Pruning</strong><br>
                    Remove dead or severely damaged plant
                    material with clean tools where appropriate.
                </div>

                <div class="item">
                    🐛 <strong>Pest Checking</strong><br>
                    Regularly inspect leaves and young shoots
                    for pests and disease symptoms.
                </div>

                <a
                    class="back"
                    href="/"
                >
                    ← Back to Plant Disease AI
                </a>

            </div>

        </body>

        </html>
        """
    )


# ============================================================
# LOGIN (name + phone number)
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login_page():

    return_to = request.values.get(
        "return_to",
        "/"
    )


    if request.method == "GET":

        # Already logged in - no need to log in again.
        if get_current_user():

            return redirect(
                return_to
            )


        return render_template(
            "login.html",

            return_to=return_to,

            mode=request.args.get("mode", "login"),

            error=None
        )


    # ------------------------------------------------------
    # POST - attempt login / account creation
    # ------------------------------------------------------

    mode = str(
        request.form.get(
            "mode",
            "login"
        )
    ).strip().lower()

    if mode not in ("login", "signup"):
        mode = "login"

    name = str(
        request.form.get(
            "name",
            ""
        )
    ).strip()

    phone = str(
        request.form.get(
            "phone",
            ""
        )
    ).strip()


    if not is_valid_phone(phone):

        return render_template(
            "login.html",

            return_to=return_to,

            mode=mode,

            error="Please enter a valid phone number."
        )


    if len(name) > 60:

        return render_template(
            "login.html",

            return_to=return_to,

            mode=mode,

            error="Name is too long."
        )


    existing_user = find_user_by_phone(
        normalize_phone(phone)
    )

    # ------------------------------------------------------
    # "Log In" tab: an account must already exist for this
    # phone number. Name is not required here - the phone
    # number is the account's identity.
    # ------------------------------------------------------
    if mode == "login" and not existing_user:

        return render_template(
            "login.html",

            return_to=return_to,

            mode=mode,

            error="We couldn't find an account with that phone number. Try Create Account instead."
        )


    # ------------------------------------------------------
    # "Create Account" tab: a name is required to set up a
    # brand new account. If the phone is already registered,
    # we just log them into the existing account.
    # ------------------------------------------------------
    if mode == "signup" and not existing_user and not name:

        return render_template(
            "login.html",

            return_to=return_to,

            mode=mode,

            error="Please enter your name to create an account."
        )


    user = get_or_create_user(
        name or (existing_user.get("name") if existing_user else ""),
        phone
    )


    session["user_id"] = user["id"]

    session["user_name"] = user["name"]

    _terminal_record_log(
        "👤 USER LOGIN / ACCOUNT",
        user,
        {
            "Action": "Login" if existing_user else "Create Account",
            "Return to": return_to,
            "Supabase": "ENABLED" if SUPABASE_ENABLED else "DISABLED"
        }
    )

    return redirect(
        return_to
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route(
    "/logout",
    methods=["POST"]
)
def logout():

    return_to = request.form.get(
        "return_to",
        "/"
    )


    session.clear()


    return redirect(
        return_to
    )


# ============================================================
# PROFILE
# ============================================================

@app.route(
    "/profile",
    methods=["GET"]
)
def profile_page():

    return_to = request.args.get(
        "return_to",
        "/"
    )


    user = get_current_user()


    if not user:

        return redirect(
            url_for(
                "login_page",
                return_to=request.full_path
                if request.query_string
                else request.path
            )
        )


    scan_count = len(
        load_collection_records(
            user_id=user.get("id")
        )
    )

    comparison_count = len(
        load_before_after_records(
            user_id=user.get("id")
        )
    )

    return render_template(
        "profile.html",

        username=user.get("name"),

        phone=user.get("phone"),

        created_at=user.get("created_at"),

        scan_count=scan_count,

        comparison_count=comparison_count,

        return_to=return_to
    )


@app.route(
    "/api/profile/update",
    methods=["POST"]
)
def update_profile():

    user = get_current_user()

    if not user:

        return jsonify({
            "success": False,
            "error": "You need to be logged in to update your profile."
        }), 401


    data = request.get_json(silent=True) or request.form

    name = (data.get("name") or "").strip()
    phone_input = data.get("phone") or ""


    if not name:

        return jsonify({
            "success": False,
            "error": "Name cannot be empty."
        }), 400


    if len(name) > 60:

        return jsonify({
            "success": False,
            "error": "Name is too long (60 characters max)."
        }), 400


    if not is_valid_phone(phone_input):

        return jsonify({
            "success": False,
            "error": "Enter a valid phone number (7-15 digits)."
        }), 400


    digits = normalize_phone(
        phone_input
    )

    existing = find_user_by_phone(
        digits
    )

    if existing and existing.get("id") != user.get("id"):

        return jsonify({
            "success": False,
            "error": "That phone number is already linked to another account."
        }), 400


    users = load_users()

    updated = False

    for stored_user in users:

        if stored_user.get("id") == user.get("id"):

            stored_user["name"] = name
            stored_user["phone"] = digits
            updated = True
            break


    if not updated:

        return jsonify({
            "success": False,
            "error": "Could not find your account to update."
        }), 404


    save_users(
        users
    )

    session["user_name"] = name

    return jsonify({
        "success": True,
        "name": name,
        "phone": digits
    }), 200


# ============================================================
# CHAT PAGE
# ============================================================

@app.route(
    "/chat",
    methods=["GET"]
)
def chat_page():

    # No disease param (e.g. opened from the home page "AI Chat"
    # tile) means "no diagnosis yet" - default to an EMPTY string,
    # not "Unknown". An empty string is falsy in the template, so
    # the general "ask me anything" welcome screen shows correctly
    # instead of a fake "Current diagnosis: Unknown" context card.
    disease = request.args.get(
        "disease",
        ""
    ).strip()


    confidence_raw = request.args.get(
        "confidence",
        "0"
    )


    try:

        confidence = float(
            confidence_raw
        )

    except (
        TypeError,
        ValueError
    ):

        confidence = 0.0


    treatment = find_treatment(
        disease
    )


    display_name = (
        get_display_name(
            disease,
            treatment
        )
        if disease
        else ""
    )


    plant_key = (
        detect_plant_from_disease(disease)
        if disease
        else request.args.get(
            "plant",
            ""
        ).strip().lower()
    )

    plant_name = (
        SUPPORTED_PLANTS[plant_key]["display"]
        if plant_key in SUPPORTED_PLANTS
        else ""
    )


    return_to = request.args.get(
        "return_to",
        "/"
    )


    return render_template(
        "chat.html",

        disease=disease,

        display_name=display_name,

        confidence=round(
            confidence,
            2
        ),

        treatment=treatment,

        plant_name=plant_name,

        return_to=return_to,

        mentalities=MENTALITIES
    )


# ============================================================
# CHAT API
# ============================================================

@app.route(
    "/api/chat",
    methods=["POST"]
)
def chat_api():

    data = request.get_json(
        silent=True
    )


    if not isinstance(
        data,
        dict
    ):

        return jsonify({

            "success":
                False,

            "error":
                "Invalid request."

        }), 400


    message = str(
        data.get(
            "message",
            ""
        )
    ).strip()


    disease = str(
        data.get(
            "disease",
            ""
        )
    ).strip()


    mentality = str(
        data.get(
            "mentality",
            "friendly"
        )
    ).strip().lower()


    page = str(
        data.get(
            "page",
            ""
        )
    ).strip().lower()

    restrict_to_disease = (page == "treatment")


    try:

        confidence = float(
            data.get(
                "confidence",
                0
            )
        )

    except (
        TypeError,
        ValueError
    ):

        confidence = 0.0


    confidence = max(
        0.0,
        min(
            confidence,
            100.0
        )
    )


    if not message:

        return jsonify({

            "success":
                False,

            "error":
                "Please enter a message."

        }), 400


    if len(message) > 1000:

        return jsonify({

            "success":
                False,

            "error":
                (
                    "Message is too long. "
                    "Please keep it under "
                    "1000 characters."
                )

        }), 400


    if mentality not in MENTALITIES:

        mentality = "friendly"


    treatment = find_treatment(
        disease
    )


    response = generate_chat_response(

        message=message,

        disease=disease,

        confidence=confidence,

        treatment=treatment,

        mentality=mentality,

        restrict_to_disease=restrict_to_disease
    )


    save_chat_message(

        message=message,

        response=response,

        mentality=mentality,

        disease=disease
    )


    return jsonify({

        "success":
            True,

        "response":
            response,

        "mentality":
            mentality,

        "disease":
            disease,

        "display_name":
            get_display_name(
                disease,
                treatment
            ),

        "confidence":
            confidence
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "success":
            True,

        "model":
            "loaded",

        "classes":
            len(index_to_class),

        "treatments":
            len(treatments),

        "chat":
            "available",

        "status":
            "running"
    })


# ============================================================
# UPLOAD HISTORY
# ============================================================

@app.route(
    "/upload-history",
    methods=["GET"]
)
def upload_history():

    user = get_current_user()

    if not user:

        return redirect(
            url_for(
                "login_page",
                return_to=request.full_path
                if request.query_string
                else request.path
            )
        )

    records = load_collection_records(
        user.get("id")
    )

    records = list(
        reversed(records)
    )


    return render_template(
        "upload_history.html",

        records=records
    )


# ============================================================
# DELETE A SCAN (UPLOAD HISTORY RECORD)
# ============================================================

@app.route(
    "/api/upload-history/<record_id>",
    methods=["DELETE"]
)
def delete_upload_history_record(record_id):

    user = get_current_user()

    if not user:

        return jsonify({
            "success": False,
            "error": "Please login first."
        }), 401

    try:

        records = load_collection_records()

        target = None
        remaining = []

        for record in records:

            if (
                str(record.get("id", ""))
                == str(record_id)
                and
                str(record.get("user_id", ""))
                == str(user.get("id"))
            ):

                target = record

            else:

                remaining.append(
                    record
                )

        if target is None:

            return jsonify({
                "success": False,
                "error": "Scan record not found."
            }), 404

        saved_filename = secure_filename(
            str(
                target.get(
                    "saved_filename",
                    ""
                )
            )
        )

        if saved_filename:

            image_path = os.path.join(
                UPLOAD_FOLDER,
                saved_filename
            )

            if os.path.exists(image_path):

                try:
                    os.remove(image_path)
                except OSError:
                    pass

            _delete_storage_image(
                "uploads/" + saved_filename
            )

        if not _persistent_save_records(
            COLLECTION_FILE,
            remaining
        ):

            return jsonify({
                "success": False,
                "error": "Could not save the updated scan history."
            }), 500

        return jsonify({
            "success": True,
            "message": "Scan deleted."
        }), 200

    except Exception as error:

        print(
            "❌ Upload history delete error:",
            error
        )

        return jsonify({
            "success": False,
            "error": "Could not delete the scan."
        }), 500


# ============================================================
# CHAT HISTORY
# ============================================================

@app.route(
    "/chat-history",
    methods=["GET"]
)
def chat_history():

    user = get_current_user()

    if not user:

        return redirect(
            url_for(
                "login_page",
                return_to=request.full_path
                if request.query_string
                else request.path
            )
        )

    records = load_chat_records(
        user.get("id")
    )

    records = list(
        reversed(records)
    )


    return render_template(
        "chat_history.html",

        records=records
    )

# ============================================================
# BEFORE / AFTER TREATMENT RECOVERY TRACKING
# ============================================================
#
# IMPORTANT DESIGN:
# 1. A BEFORE scan can be saved from the normal result page.
# 2. Saved BEFORE images belong to the logged-in user.
# 3. During comparison, BEFORE and AFTER are independently passed
#    through the SAME ResNet50 model.
# 4. Disease prediction is the PRIMARY recovery signal.
# 5. Pixel similarity is SECONDARY only and must never decide
#    whether treatment worked.
# 6. If BEFORE and AFTER disease predictions differ, the API sends
#    a prominent warning because this may represent a new problem,
#    treatment failure, image variation, or model uncertainty.
# ============================================================

BEFORE_AFTER_FILE = os.path.join(
    DATA_FOLDER,
    "before_after_records.json"
)

BEFORE_FOLDER = os.path.join(
    UPLOAD_FOLDER,
    "before"
)

AFTER_FOLDER = os.path.join(
    UPLOAD_FOLDER,
    "after"
)

os.makedirs(
    BEFORE_FOLDER,
    exist_ok=True
)

os.makedirs(
    AFTER_FOLDER,
    exist_ok=True
)


def _load_json_list(file_path):

    if not os.path.exists(file_path):
        return []

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return data if isinstance(data, list) else []

    except Exception as error:

        print(
            "⚠️ JSON list read error:",
            file_path,
            error
        )

        return []


def _save_json_list(file_path, records):

    try:

        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True
        )

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                records,
                file,
                indent=4,
                ensure_ascii=False
            )

        return True

    except Exception as error:

        print(
            "⚠️ JSON list save error:",
            file_path,
            error
        )

        return False


def load_before_after_records(user_id=None):

    records = _load_json_list(
        BEFORE_AFTER_FILE
    )

    if user_id is None:
        return records

    return [
        record
        for record in records
        if str(record.get("user_id", "")) == str(user_id)
    ]


def save_before_after_records(records):

    return _save_json_list(
        BEFORE_AFTER_FILE,
        records
    )


def load_saved_before_records(user_id=None):

    records = _load_json_list(
        BEFORE_SAVED_FILE
    )

    if user_id is None:
        return records

    return [
        record
        for record in records
        if str(record.get("user_id", "")) == str(user_id)
    ]


def save_saved_before_records(records):

    return _save_json_list(
        BEFORE_SAVED_FILE,
        records
    )


def calculate_image_difference(
    before_path,
    after_path
):

    """
    Secondary visual-change metric only.
    This does NOT decide treatment recovery.
    """

    try:

        before_image = Image.open(
            before_path
        ).convert("RGB")

        after_image = Image.open(
            after_path
        ).convert("RGB")

        before_image = before_image.resize(
            (224, 224)
        )

        after_image = after_image.resize(
            (224, 224)
        )

        before_array = np.asarray(
            before_image,
            dtype=np.float32
        )

        after_array = np.asarray(
            after_image,
            dtype=np.float32
        )

        difference = float(
            np.mean(
                np.abs(
                    before_array - after_array
                )
            )
        )

        similarity = max(
            0.0,
            min(
                100.0,
                100.0 - (
                    difference / 255.0 * 100.0
                )
            )
        )

        return round(
            similarity,
            2
        )

    except Exception as error:

        print(
            "⚠️ Image comparison error:",
            error
        )

        return None


def _comparison_display_name(disease):

    treatment = find_treatment(
        disease
    )

    if isinstance(
        treatment,
        dict
    ):

        return get_display_name(
            disease,
            treatment
        )

    return str(
        disease or "Unknown"
    )


def _comparison_plant(disease):

    treatment = find_treatment(
        disease
    )

    if isinstance(
        treatment,
        dict
    ):

        plant = (
            treatment.get("plant")
            or treatment.get("plant_name")
        )

        if plant:
            return plant

    return detect_plant_from_disease(
        disease
    )


def _is_healthy_disease(disease):

    text = str(
        disease or ""
    ).lower()

    return "healthy" in text


def _severity_tier(confidence):
    """
    Turns a raw AFTER-scan confidence number into a plain-language
    severity tier so recommendations can give different real-life
    advice for a severe / moderate / mild reading, instead of one
    generic paragraph for every case.
    """

    confidence = float(confidence or 0)

    if confidence >= 70:
        return (
            "severe",
            "the AFTER scan is still strongly matching this condition, "
            "so the symptoms are likely clear and established"
        )

    elif confidence >= 40:
        return (
            "moderate",
            "the AFTER scan is only moderately confident, so symptoms "
            "are present but not extreme"
        )

    else:
        return (
            "mild",
            "the AFTER scan's confidence is low, so this could be an "
            "early, mild case — or simply a less clear photo"
        )


def compare_recovery_predictions(
    before_disease,
    before_confidence,
    after_disease,
    after_confidence,
    visual_similarity
):

    before_disease = str(
        before_disease or "Unknown"
    )

    after_disease = str(
        after_disease or "Unknown"
    )

    before_confidence = float(
        before_confidence or 0
    )

    after_confidence = float(
        after_confidence or 0
    )

    before_display = _comparison_display_name(
        before_disease
    )

    after_display = _comparison_display_name(
        after_disease
    )

    before_plant = _comparison_plant(
        before_disease
    )

    after_plant = _comparison_plant(
        after_disease
    )

    same_disease = (
        before_disease.strip().lower()
        == after_disease.strip().lower()
    )

    same_plant = (
        str(before_plant or "").strip().lower()
        == str(after_plant or "").strip().lower()
    )

    before_healthy = _is_healthy_disease(
        before_disease
    )

    after_healthy = _is_healthy_disease(
        after_disease
    )

    confidence_change = round(
        after_confidence - before_confidence,
        2
    )

    # ------------------------------------------------------------
    # FIRST PRIORITY: healthy -> disease is a warning.
    # ------------------------------------------------------------

    if before_healthy and not after_healthy:

        status = "warning"
        status_title = "New Problem Detected"
        status_icon = "🚨"

        message = (
            f"The BEFORE scan appeared healthy, but the AFTER scan "
            f"was classified as {after_display}."
        )

        severity_tier, severity_note = _severity_tier(after_confidence)

        if severity_tier == "severe":
            severity_advice = (
                "Treat this as active — remove the worst-affected leaves with a "
                "clean blade, isolate the plant from its neighbours, and start "
                f"{after_display}'s treatment steps today rather than waiting to see if it clears on its own."
            )
        elif severity_tier == "moderate":
            severity_advice = (
                "Start the standard treatment steps for this condition now, but "
                "there's no need to panic — keep the dose light-to-normal and "
                "recheck in 3-4 days before increasing anything."
            )
        else:
            severity_advice = (
                "Hold off on treating yet — retake a sharp, well-lit photo of the "
                "exact same spot first. Low confidence often means this is very "
                "early, or the photo just wasn't clear enough to be sure."
            )

        recommendation = (
            f"Since {severity_note}, {severity_advice}"
        )

        action_plan = [
            {
                "icon": "🔍",
                "label": "Do this now",
                "tone": "warn",
                "items": [
                    "Inspect the same leaf/area closely in bright natural light — check both sides for spots, webbing, sticky residue or visible pests.",
                    "Move this plant a little away from healthy neighbours so nothing spreads while you confirm what's going on.",
                    "Snip off only the clearly affected leaves with a clean or alcohol-wiped blade, and bin them rather than composting."
                ]
            },
            {
                "icon": "📊",
                "label": f"What to do — this reads as {severity_tier}",
                "tone": "danger" if severity_tier == "severe" else ("warn" if severity_tier == "moderate" else "info"),
                "items": [severity_advice]
            },
            {
                "icon": "🌿",
                "label": "Keep doing",
                "tone": "good",
                "items": [
                    "Keep the watering, light and feeding routine that had this plant healthy — one new spot doesn't mean the whole care plan needs to change."
                ]
            },
            {
                "icon": "🚫",
                "label": "Avoid",
                "tone": "danger",
                "items": [
                    "Don't reach for a random spray before confirming the problem — treating the wrong issue adds stress without helping.",
                    "Don't assume a past treatment caused this; new pests, a different leaf, or normal wear are common unrelated causes."
                ]
            }
        ]

    # ------------------------------------------------------------
    # SECOND PRIORITY: disease changed.
    # ------------------------------------------------------------

    elif not same_disease:

        status = "warning"
        status_title = "Disease Prediction Changed"
        status_icon = "⚠️"

        message = (
            f"The BEFORE scan was classified as {before_display}, "
            f"while the AFTER scan was classified as {after_display}."
        )

        recommendation = (
            "The predicted condition is different between the two scans. Confirm it's "
            "real with a matching retake, then treat the AFTER result as the current, "
            "more accurate diagnosis."
        )

        action_plan = [
            {
                "icon": "🔁",
                "label": "Verify first",
                "tone": "warn",
                "items": [
                    "Retake both photos under matching conditions — same leaf, same distance, same lighting — and run the comparison again before changing treatment.",
                    "A changed result can be a genuine new/overlapping problem, an ineffective original treatment, or just a photo difference — the retake tells you which."
                ]
            },
            {
                "icon": "🩹",
                "label": "If the change is confirmed",
                "tone": "danger",
                "items": [
                    f"Switch to {after_display}'s specific treatment steps — don't keep following the plan meant for {before_display}.",
                    "Isolate the plant, remove the worst-affected leaves, and improve airflow around it while the new treatment takes effect."
                ]
            },
            {
                "icon": "🚫",
                "label": "Avoid",
                "tone": "danger",
                "items": [
                    "Don't layer a second product on top of the first without checking they're safe to combine.",
                    "Don't panic-treat for both conditions at once before confirming — over-treating can stress the plant more than either disease alone."
                ]
            }
        ]

    # ------------------------------------------------------------
    # THIRD PRIORITY: disease -> healthy.
    # ------------------------------------------------------------

    elif not before_healthy and after_healthy:

        status = "improving"
        status_title = "Healthy Condition Detected"
        status_icon = "🟢"

        message = (
            f"The BEFORE scan was {before_display}, and the AFTER "
            "scan is classified as healthy. This is a positive sign."
        )

        recommendation = (
            "Great news — the plant now reads as healthy. Finish out the treatment "
            "course and keep monitoring for a couple of weeks before calling it fully resolved."
        )

        action_plan = [
            {
                "icon": "🌿",
                "label": "Keep doing",
                "tone": "good",
                "items": [
                    "Keep the exact watering, sunlight and feeding routine that supported this recovery — don't change what's working.",
                    "If the treatment has a recommended full course length, finish it rather than stopping the moment symptoms disappear; stopping early is a common cause of relapse."
                ]
            },
            {
                "icon": "👀",
                "label": "Confirm it's holding",
                "tone": "info",
                "items": [
                    "Check the same leaf/area every few days for the next one to two weeks — one healthy scan is encouraging but doesn't guarantee it's fully resolved.",
                    "Run one more comparison scan in a week or two to confirm the healthy result holds."
                ]
            },
            {
                "icon": "🛡️",
                "label": "Prevent relapse",
                "tone": "good",
                "items": [
                    "Remove any dead or previously affected leaves you notice lying around the plant.",
                    "Keep this plant a little apart from others that are still showing symptoms, and keep airflow around it good."
                ]
            }
        ]

    # ------------------------------------------------------------
    # FOURTH PRIORITY: same disease, confidence changed.
    # ------------------------------------------------------------

    elif same_disease:

        if confidence_change <= -10:

            # A confidence DROP for the same disease is treated as a good
            # sign here — the symptoms are becoming less pronounced/less
            # clear-cut, which is what recovery typically looks like.
            status = "improving"
            status_title = "Possible Improvement"
            status_icon = "🟢"

            message = (
                f"The same condition ({after_display}) was detected "
                "again, but model confidence decreased noticeably — a "
                "sign the symptoms may be fading."
            )

            severity_tier, severity_note = _severity_tier(after_confidence)

            if severity_tier == "severe":
                severity_advice = (
                    f"{after_display} is still showing fairly strongly in the AFTER scan, so "
                    "it's healing but not finished — keep following the full treatment course "
                    "rather than stopping early just because confidence dropped."
                )
            elif severity_tier == "moderate":
                severity_advice = (
                    "Symptoms are fading but still present — continue the current treatment "
                    "as planned and recheck in a few days to confirm the improvement continues."
                )
            else:
                severity_advice = (
                    "Confidence is now quite low, which lines up with the leaf largely "
                    "clearing up — keep monitoring for another week or two to confirm it's "
                    "holding before calling this fully resolved."
                )

            recommendation = (
                f"Confidence dropped for the same condition — a good sign the symptoms are "
                f"fading. Since {severity_note}, {severity_advice}"
            )

            action_plan = [
                {
                    "icon": "🌿",
                    "label": "Keep doing",
                    "tone": "good",
                    "items": [
                        "Don't stop treatment early just because confidence dropped — finish the full recommended course so it doesn't relapse.",
                        "A confidence drop for the same disease usually means the symptoms are becoming less pronounced, which is what recovery looks like."
                    ]
                },
                {
                    "icon": "📊",
                    "label": f"What to do — this reads as {severity_tier}",
                    "tone": "warn" if severity_tier == "severe" else ("info" if severity_tier == "moderate" else "good"),
                    "items": [severity_advice]
                },
                {
                    "icon": "👀",
                    "label": "Confirm it's holding",
                    "tone": "info",
                    "items": [
                        "Run one more comparison in a week to make sure this trend continues rather than reversing.",
                        "Watch for any new spreading — that would be the real sign this isn't over yet, regardless of what the confidence number says."
                    ]
                },
                {
                    "icon": "🚫",
                    "label": "Avoid",
                    "tone": "danger",
                    "items": [
                        "Don't assume it's fully cured just because confidence dropped — wait for a healthy result (or very low, stable confidence over repeat scans) before stopping care entirely."
                    ]
                }
            ]

        elif confidence_change >= 10:

            # A confidence INCREASE for the same disease is treated as a
            # warning sign here — the symptoms are becoming clearer/more
            # pronounced, which points toward active worsening.
            status = "worsening"
            status_title = "Possible Worsening"
            status_icon = "🔴"

            message = (
                f"The same condition ({after_display}) was detected "
                "again, and model confidence increased noticeably — a "
                "sign the symptoms may be getting more pronounced."
            )

            severity_tier, severity_note = _severity_tier(after_confidence)

            if severity_tier == "severe":
                severity_advice = (
                    f"{after_display} is now showing very strongly in the AFTER scan — treat "
                    "this as active: remove the worst-affected leaves with a clean blade, "
                    "improve airflow around the plant, keep foliage dry when watering, and "
                    "apply the next treatment dose on schedule rather than skipping it."
                )
            elif severity_tier == "moderate":
                severity_advice = (
                    "Symptoms are more pronounced but not extreme yet — stay the course with "
                    "the current treatment, recheck the plant every 2-3 days, and watch "
                    "specifically for new leaves becoming affected."
                )
            else:
                severity_advice = (
                    "This rise is small enough to possibly just be a clearer photo rather "
                    "than real worsening — retake a sharp, well-lit photo of the exact same "
                    "spot and compare again before changing anything."
                )

            recommendation = (
                f"Confidence went up for the same condition — a sign symptoms may be getting "
                f"more pronounced. Since {severity_note}, {severity_advice}"
            )

            action_plan = [
                {
                    "icon": "🩺",
                    "label": "Keep the current treatment going",
                    "tone": "good",
                    "items": [
                        "Don't stop or swap the treatment yet — a confidence rise alone isn't proof it's failing, and switching too early wastes progress already made.",
                        "Keep the dose and schedule exactly as recommended in the treatment guide for this condition."
                    ]
                },
                {
                    "icon": "📊",
                    "label": f"What to do — this reads as {severity_tier}",
                    "tone": "danger" if severity_tier == "severe" else ("warn" if severity_tier == "moderate" else "info"),
                    "items": [severity_advice]
                },
                {
                    "icon": "🚫",
                    "label": "Avoid",
                    "tone": "danger",
                    "items": [
                        "Don't double the dosage or add a second product on top — overtreating can damage the plant faster than the disease itself.",
                        "Don't judge progress from one shaky or badly-lit photo; a fresh, matching photo is the only fair comparison."
                    ]
                },
                {
                    "icon": "⏭️",
                    "label": "When to escalate",
                    "tone": "info",
                    "items": [
                        "If a clear, well-lit retake still shows the rise, or you see visible spreading to new leaves within the next few days, treat it as real worsening — increase how closely you're following the treatment guide's strongest steps for this condition (removing more affected material, tightening the schedule) rather than switching products at random."
                    ]
                }
            ]

        else:

            status = "monitor"
            status_title = "Condition Still Detected"
            status_icon = "🟠"

            message = (
                f"The same condition ({after_display}) was detected "
                "in both scans. The AI does not show a clear diagnostic "
                "change."
            )

            recommendation = (
                "No clear change yet — that's normal early on. Stay on the current "
                "treatment plan and give it more time before judging results."
            )

            action_plan = [
                {
                    "icon": "🌿",
                    "label": "Keep doing",
                    "tone": "good",
                    "items": [
                        "Continue the existing treatment plan exactly as recommended rather than switching approaches too soon.",
                        "Many treatments take one to two weeks of consistent care before results show up clearly — this is often just too early to tell."
                    ]
                },
                {
                    "icon": "👀",
                    "label": "Care while you wait",
                    "tone": "info",
                    "items": [
                        "Keep monitoring the same leaf or affected area every few days, under similar lighting, so future comparisons stay accurate.",
                        "Remove any leaves that clearly aren't recovering, and keep the area around the plant tidy of fallen debris."
                    ]
                },
                {
                    "icon": "⏭️",
                    "label": "If there's still no change",
                    "tone": "warn",
                    "items": [
                        "After a full treatment cycle with no improvement, re-check the treatment steps for this condition to make sure dosing and timing are being followed correctly.",
                        "Scan a second affected leaf to see whether the problem is spreading elsewhere on the plant."
                    ]
                }
            ]

    else:

        status = "monitor"
        status_title = "Monitor Closely"
        status_icon = "🟠"

        message = (
            "The comparison did not produce a clear recovery signal."
        )

        recommendation = (
            "Take another clear photo under similar conditions and "
            "compare again."
        )

        action_plan = [
            {
                "icon": "📷",
                "label": "Do this now",
                "tone": "info",
                "items": [
                    "Take another clear, well-lit photo of the same spot as the BEFORE scan and run the comparison again.",
                    "Keep the current care routine unchanged in the meantime."
                ]
            }
        ]

    if visual_similarity is not None:

        visual_note = (
            f"Visual similarity between the two photos: "
            f"{visual_similarity}%. This is only a visual-change "
            "metric and is NOT a recovery or disease-severity score."
        )

    else:

        visual_note = (
            "Visual similarity could not be calculated."
        )

    return {
        "status": status,
        "status_title": status_title,
        "status_icon": status_icon,
        "message": message,
        "recommendation": recommendation,
        "action_plan": action_plan,
        "before_disease": before_disease,
        "before_disease_display": before_display,
        "before_confidence": round(before_confidence, 2),
        "before_plant": before_plant,
        "after_disease": after_disease,
        "after_disease_display": after_display,
        "after_confidence": round(after_confidence, 2),
        "after_plant": after_plant,
        "same_disease": same_disease,
        "same_plant": same_plant,
        "before_healthy": before_healthy,
        "after_healthy": after_healthy,
        "confidence_change": confidence_change,
        "visual_note": visual_note,
    }


def _save_comparison_image(
    file,
    folder,
    record_id,
    suffix
):

    safe_name = secure_filename(
        file.filename or ""
    )

    extension = (
        safe_name.rsplit(".", 1)[-1].lower()
        if "." in safe_name
        else ""
    )

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Invalid {suffix.upper()} image format."
        )

    filename = (
        f"{record_id}_{suffix}.{extension}"
    )

    path = os.path.join(
        folder,
        filename
    )

    file.save(
        path
    )

    try:

        with Image.open(path) as image:
            image.verify()

    except Exception:

        if os.path.exists(path):
            os.remove(path)

        raise ValueError(
            f"Invalid {suffix.upper()} image."
        )

    return (
        filename,
        path,
        safe_name
    )


# ============================================================
# SAVE A NORMAL SCAN AS A BEFORE IMAGE
# ============================================================

@app.route(
    "/api/before-save",
    methods=["POST"]
)
def save_scan_as_before():

    user = get_current_user()

    if not user:

        return jsonify({
            "success": False,
            "error": "Please login before saving a BEFORE image."
        }), 401

    try:

        data = request.get_json(
            silent=True
        ) or {}

        source_filename = secure_filename(
            str(
                data.get(
                    "saved_filename",
                    ""
                )
            )
        )

        if not source_filename:

            return jsonify({
                "success": False,
                "error": "The scan image could not be identified."
            }), 400

        source_path = os.path.join(
            UPLOAD_FOLDER,
            source_filename
        )

        if not os.path.isfile(source_path):

            return jsonify({
                "success": False,
                "error": "The original scan image is no longer available."
            }), 404

        # Prevent path traversal even if a malicious filename is sent.
        if os.path.basename(source_filename) != source_filename:

            return jsonify({
                "success": False,
                "error": "Invalid image reference."
            }), 400

        # Avoid accidentally allowing a file outside UPLOAD_FOLDER.
        source_real = os.path.realpath(
            source_path
        )

        upload_real = os.path.realpath(
            UPLOAD_FOLDER
        )

        if not source_real.startswith(
            upload_real + os.sep
        ):

            return jsonify({
                "success": False,
                "error": "Invalid image location."
            }), 400

        # Verify the source image before copying it.
        with Image.open(source_path) as image:
            image.verify()

        existing_records = _load_json_list(
            BEFORE_SAVED_FILE
        )

        # A scan can be saved once as BEFORE. Check this before creating
        # a new file so duplicate clicks do not leave orphaned images.
        for existing in existing_records:

            if (
                str(existing.get("user_id", ""))
                == str(user.get("id"))
                and
                existing.get("source_scan_filename")
                == source_filename
            ):

                return jsonify({
                    "success": True,
                    "already_saved": True,
                    "record": existing,
                    "message": "This scan is already saved as a BEFORE image."
                }), 200

        before_id = uuid.uuid4().hex

        # The saved copy is always JPEG, so the extension matches the
        # actual file format.
        before_filename = (
            f"{before_id}_before.jpg"
        )

        before_path = os.path.join(
            BEFORE_FOLDER,
            before_filename
        )

        # Copy through PIL so the saved BEFORE file is a valid JPEG.
        with Image.open(source_path) as image:

            image.convert("RGB").save(
                before_path,
                format="JPEG",
                quality=95
            )

        disease = str(
            data.get("disease", "Unknown")
        ).strip() or "Unknown"

        confidence = float(
            data.get("confidence", 0) or 0
        )

        plant_name = str(
            data.get("plant_name", "")
        ).strip()

        before_image_url = url_for(
            "static",
            filename=f"uploads/before/{before_filename}"
        )

        storage_before_url = _upload_image_to_supabase(
            before_path,
            "before/" + before_filename
        )

        if storage_before_url:
            before_image_url = storage_before_url

        record = {
            "id": before_id,
            "user_id": user.get("id"),
            "type": "saved_before",
            "before_image": before_image_url,
            "before_filename": data.get(
                "original_filename",
                source_filename
            ),
            "stored_filename": before_filename,
            "source_scan_filename": source_filename,
            "disease": disease,
            "disease_display": _comparison_display_name(
                disease
            ),
            "confidence": round(
                confidence,
                2
            ),
            "plant_name": plant_name or _comparison_plant(
                disease
            ),
            "created_at": datetime.now().isoformat(
                timespec="seconds"
            )
        }

        records = existing_records

        records.append(
            record
        )

        if not save_saved_before_records(
            records
        ):

            if os.path.exists(before_path):
                os.remove(before_path)

            return jsonify({
                "success": False,
                "error": "Could not save the BEFORE record."
            }), 500

        print(
            "💾 BEFORE baseline saved for user:",
            user.get("id")
        )

        return jsonify({
            "success": True,
            "already_saved": False,
            "record": record,
            "message": "BEFORE image saved successfully."
        }), 200

    except Exception as error:

        print(
            "❌ Save BEFORE error:",
            error
        )

        return jsonify({
            "success": False,
            "error": "Could not save this scan as a BEFORE image."
        }), 500


# ============================================================
# GET SAVED BEFORE IMAGES FOR CURRENT USER
# ============================================================

@app.route(
    "/api/before-images",
    methods=["GET"]
)
def before_images_api():

    user = get_current_user()

    if not user:

        return jsonify({
            "success": False,
            "records": [],
            "error": "Please login first."
        }), 401

    records = load_saved_before_records(
        user.get("id")
    )

    return jsonify({
        "success": True,
        "records": list(
            reversed(records)
        )
    }), 200


# ============================================================
# BEFORE / AFTER PAGE
# ============================================================

@app.route(
    "/before-after",
    methods=["GET"]
)
def before_after_page():

    user = get_current_user()

    if not user:

        return redirect(
            url_for(
                "login_page",
                return_to=request.full_path
                if request.query_string
                else request.path
            )
        )

    try:

        records = load_before_after_records(
            user.get("id")
        )

        saved_before = load_saved_before_records(
            user.get("id")
        )

        return render_template(
            "before_after.html",
            records=list(
                reversed(records)
            ),
            saved_before=list(
                reversed(saved_before)
            ),
            current_user=user
        )

    except Exception as error:

        print(
            "❌ Before/After page error:",
            error
        )

        return (
            "Before/After page could not be loaded.",
            500
        )


# ============================================================
# BEFORE / AFTER RESNET50 API
# ============================================================

@app.route(
    "/api/before-after",
    methods=["POST"]
)
def before_after_api():

    user = get_current_user()

    if not user:

        return jsonify({
            "success": False,
            "error": "Please login before comparing treatment progress."
        }), 401

    before_path = None
    after_path = None
    before_owned_by_comparison = False

    try:

        print(
            "\n" + "=" * 70
        )

        print(
            "🌿 BEFORE / AFTER RESNET50 RECOVERY COMPARISON"
        )

        print(
            "=" * 70
        )

        # --------------------------------------------------------
        # BEFORE can come from a saved baseline OR a fresh upload.
        # --------------------------------------------------------

        before_record_id = str(
            request.form.get(
                "before_record_id",
                ""
            )
        ).strip()

        before_file = (
            request.files.get("before_image")
            or request.files.get("before")
        )

        after_file = (
            request.files.get("after_image")
            or request.files.get("after")
        )

        if not after_file:

            return jsonify({
                "success": False,
                "error": "Please upload the AFTER image."
            }), 400

        if not validate_image_file(
            after_file
        ):

            return jsonify({
                "success": False,
                "error": (
                    "Invalid AFTER image format. "
                    "Please use JPG, JPEG, PNG or WEBP."
                )
            }), 400

        # --------------------------------------------------------
        # Resolve saved BEFORE image securely.
        # --------------------------------------------------------

        if before_record_id:

            saved_records = load_saved_before_records(
                user.get("id")
            )

            saved_before = next(
                (
                    record
                    for record in saved_records
                    if str(record.get("id", ""))
                    == before_record_id
                ),
                None
            )

            if saved_before is None:

                return jsonify({
                    "success": False,
                    "error": "Saved BEFORE image was not found for this account."
                }), 404

            stored_filename = secure_filename(
                str(
                    saved_before.get(
                        "stored_filename",
                        ""
                    )
                )
            )

            if not stored_filename:

                return jsonify({
                    "success": False,
                    "error": "Saved BEFORE image reference is invalid."
                }), 400

            before_path = os.path.join(
                BEFORE_FOLDER,
                stored_filename
            )

            before_original = saved_before.get(
                "before_filename",
                stored_filename
            )

            if not os.path.isfile(before_path):

                if not _download_storage_image(
                    saved_before.get("before_image", ""),
                    before_path
                ):
                    return jsonify({
                        "success": False,
                        "error": "The saved BEFORE image file is missing."
                    }), 404

            # Verify that the saved file is a valid image.
            with Image.open(before_path) as image:
                image.verify()

        else:

            if not before_file:

                return jsonify({
                    "success": False,
                    "error": (
                        "Please select a saved BEFORE image or upload "
                        "a BEFORE image."
                    )
                }), 400

            if not validate_image_file(
                before_file
            ):

                return jsonify({
                    "success": False,
                    "error": (
                        "Invalid BEFORE image format. "
                        "Please use JPG, JPEG, PNG or WEBP."
                    )
                }), 400

            comparison_id = uuid.uuid4().hex

            (
                before_filename,
                before_path,
                before_original
            ) = _save_comparison_image(
                before_file,
                BEFORE_FOLDER,
                comparison_id,
                "before"
            )

            before_owned_by_comparison = True

        # --------------------------------------------------------
        # Save AFTER image.
        # --------------------------------------------------------

        comparison_id = (
            uuid.uuid4().hex
        )

        (
            after_filename,
            after_path,
            after_original
        ) = _save_comparison_image(
            after_file,
            AFTER_FOLDER,
            comparison_id,
            "after"
        )

        # --------------------------------------------------------
        # PRIMARY ANALYSIS:
        # SAME ResNet50 model independently predicts both images.
        # --------------------------------------------------------

        (
            before_disease,
            before_confidence,
            before_confidence_level
        ) = predict_disease(
            before_path
        )

        (
            after_disease,
            after_confidence,
            after_confidence_level
        ) = predict_disease(
            after_path
        )

        # --------------------------------------------------------
        # SECONDARY visual comparison.
        # --------------------------------------------------------

        visual_similarity = calculate_image_difference(
            before_path,
            after_path
        )

        comparison = compare_recovery_predictions(
            before_disease,
            before_confidence,
            after_disease,
            after_confidence,
            visual_similarity
        )

        # --------------------------------------------------------
        # Build persistent record.
        # --------------------------------------------------------

        if before_record_id:

            before_image_url = saved_before.get(
                "before_image"
            )

        else:

            before_image_url = url_for(
                "static",
                filename=f"uploads/before/{os.path.basename(before_path)}"
            )

            storage_before_url = _upload_image_to_supabase(
                before_path,
                "before/" + os.path.basename(before_path)
            )

            if storage_before_url:
                before_image_url = storage_before_url

        after_image_url = url_for(
            "static",
            filename=f"uploads/after/{after_filename}"
        )

        storage_after_url = _upload_image_to_supabase(
            after_path,
            "after/" + after_filename
        )

        if storage_after_url:
            after_image_url = storage_after_url

        record = {
            "id": comparison_id,
            "user_id": user.get("id"),
            "before_source": (
                "saved_before"
                if before_record_id
                else "uploaded_for_comparison"
            ),
            "before_record_id": (
                before_record_id or None
            ),
            "before_image": before_image_url,
            "after_image": after_image_url,
            "before_filename": before_original,
            "after_filename": after_original,
            "before_disease": before_disease,
            "before_disease_display": comparison[
                "before_disease_display"
            ],
            "before_confidence": before_confidence,
            "before_confidence_level": before_confidence_level,
            "before_plant": comparison[
                "before_plant"
            ],
            "after_disease": after_disease,
            "after_disease_display": comparison[
                "after_disease_display"
            ],
            "after_confidence": after_confidence,
            "after_confidence_level": after_confidence_level,
            "after_plant": comparison[
                "after_plant"
            ],
            "confidence_change": comparison[
                "confidence_change"
            ],
            "same_disease": comparison[
                "same_disease"
            ],
            "same_plant": comparison[
                "same_plant"
            ],
            "before_healthy": comparison[
                "before_healthy"
            ],
            "after_healthy": comparison[
                "after_healthy"
            ],
            "similarity": visual_similarity,
            "visual_similarity": visual_similarity,
            "comparison_status": comparison[
                "status"
            ],
            "status_title": comparison[
                "status_title"
            ],
            "status_icon": comparison[
                "status_icon"
            ],
            "comparison_message": comparison[
                "message"
            ],
            "recommendation": comparison[
                "recommendation"
            ],
            "action_plan": comparison.get(
                "action_plan", []
            ),
            "visual_note": comparison[
                "visual_note"
            ],
            "created_at": datetime.now().isoformat(
                timespec="seconds"
            )
        }

        records = _load_json_list(
            BEFORE_AFTER_FILE
        )

        records.append(
            record
        )

        if not save_before_after_records(
            records
        ):

            if after_path and os.path.exists(after_path):
                os.remove(after_path)

            if (
                before_owned_by_comparison
                and before_path
                and os.path.exists(before_path)
            ):
                os.remove(before_path)

            return jsonify({
                "success": False,
                "error": (
                    "Images were processed, but the comparison "
                    "record could not be saved."
                )
            }), 500

        # --------------------------------------------------------
        # Console diagnostics.
        # --------------------------------------------------------

        print(
            "BEFORE:",
            before_disease,
            before_confidence,
            "%"
        )

        print(
            "AFTER :",
            after_disease,
            after_confidence,
            "%"
        )

        print(
            "STATUS:",
            comparison["status"]
        )

        if not comparison["same_disease"]:

            print(
                "⚠️ WARNING: BEFORE and AFTER disease predictions differ."
            )

        print(
            "=" * 70 + "\n"
        )

        return jsonify({
            "success": True,
            "record": record,
            "comparison": comparison,
            "message": (
                "Before/After ResNet50 comparison completed successfully."
            )
        }), 200

    except ValueError as error:

        if (
            before_owned_by_comparison
            and before_path
            and os.path.exists(before_path)
        ):
            os.remove(before_path)

        if after_path and os.path.exists(after_path):
            os.remove(after_path)

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except Exception as error:

        print(
            "❌ Before/After comparison failed:",
            error
        )

        import traceback

        traceback.print_exc()

        if (
            before_owned_by_comparison
            and before_path
            and os.path.exists(before_path)
        ):
            try:
                os.remove(before_path)
            except OSError:
                pass

        if after_path and os.path.exists(after_path):
            try:
                os.remove(after_path)
            except OSError:
                pass

        return jsonify({
            "success": False,
            "error": (
                "Could not process the Before/After comparison. "
                "Please try again."
            )
        }), 500


# ============================================================
# GET CURRENT USER'S BEFORE / AFTER RECORDS
# ============================================================

@app.route(
    "/api/before-after/records",
    methods=["GET"]
)
def before_after_records_api():

    user = get_current_user()

    if not user:

        return jsonify({
            "success": False,
            "records": [],
            "error": "Please login first."
        }), 401

    try:

        records = load_before_after_records(
            user.get("id")
        )

        return jsonify({
            "success": True,
            "records": list(
                reversed(records)
            )
        }), 200

    except Exception as error:

        print(
            "❌ Before/After records API error:",
            error
        )

        return jsonify({
            "success": False,
            "records": [],
            "error": "Could not load Before/After records."
        }), 500


# ============================================================
# DELETE SAVED BEFORE BASELINE
# ============================================================

@app.route(
    "/api/before-images/<record_id>",
    methods=["DELETE"]
)
def delete_saved_before(record_id):

    user = get_current_user()

    if not user:

        return jsonify({
            "success": False,
            "error": "Please login first."
        }), 401

    try:

        records = _load_json_list(
            BEFORE_SAVED_FILE
        )

        target = None
        remaining = []

        for record in records:

            if (
                str(record.get("id", ""))
                == str(record_id)
                and
                str(record.get("user_id", ""))
                == str(user.get("id"))
            ):

                target = record

            else:

                remaining.append(
                    record
                )

        if target is None:

            return jsonify({
                "success": False,
                "error": "Saved BEFORE image not found."
            }), 404

        filename = secure_filename(
            str(
                target.get(
                    "stored_filename",
                    ""
                )
            )
        )

        if filename:

            path = os.path.join(
                BEFORE_FOLDER,
                filename
            )

            if os.path.exists(path):
                os.remove(path)

        _delete_storage_image(
            "before/" + filename
        )

        if not save_saved_before_records(
            remaining
        ):

            return jsonify({
                "success": False,
                "error": "Could not save the updated BEFORE records."
            }), 500

        return jsonify({
            "success": True,
            "message": "Saved BEFORE image deleted."
        }), 200

    except Exception as error:

        print(
            "❌ Saved BEFORE delete error:",
            error
        )

        return jsonify({
            "success": False,
            "error": "Could not delete the saved BEFORE image."
        }), 500


# ============================================================
# DELETE BEFORE / AFTER COMPARISON RECORD
# ============================================================

@app.route(
    "/api/before-after/<record_id>",
    methods=["DELETE"]
)
def delete_before_after_record(record_id):

    user = get_current_user()

    if not user:

        return jsonify({
            "success": False,
            "error": "Please login first."
        }), 401

    try:

        records = _load_json_list(
            BEFORE_AFTER_FILE
        )

        target = None
        remaining = []

        for record in records:

            if (
                str(record.get("id", ""))
                == str(record_id)
                and
                str(record.get("user_id", ""))
                == str(user.get("id"))
            ):

                target = record

            else:

                remaining.append(
                    record
                )

        if target is None:

            return jsonify({
                "success": False,
                "error": "Before/After record not found."
            }), 404

        # Do NOT delete a saved baseline when deleting a comparison.
        before_source = target.get(
            "before_source",
            "uploaded_for_comparison"
        )

        if before_source != "saved_before":

            before_url = target.get(
                "before_image",
                ""
            )

            before_filename = os.path.basename(
                before_url
            )

            if before_filename:

                before_path = os.path.join(
                    BEFORE_FOLDER,
                    secure_filename(before_filename)
                )

                if os.path.exists(before_path):
                    os.remove(before_path)

            _delete_storage_image(
                "before/" + before_filename
            )

        after_url = target.get(
            "after_image",
            ""
        )

        after_filename = os.path.basename(
            after_url
        )

        if after_filename:

            after_path = os.path.join(
                AFTER_FOLDER,
                secure_filename(after_filename)
            )

            if os.path.exists(after_path):
                os.remove(after_path)

        _delete_storage_image(
            "after/" + after_filename
        )

        if not save_before_after_records(
            remaining
        ):

            return jsonify({
                "success": False,
                "error": "Could not save updated comparison records."
            }), 500

        return jsonify({
            "success": True,
            "message": "Before/After record deleted."
        }), 200

    except Exception as error:

        print(
            "❌ Before/After delete error:",
            error
        )

        return jsonify({
            "success": False,
            "error": "Could not delete the Before/After record."
        }), 500


# ============================================================
# PLANT PROFILES
# ("My Plants" - one record per real, physical plant. This is
# the foundation the Health dashboard, History, Treatment
# tracking and Care Calendar are all built on top of.)
# ============================================================

def load_plant_profiles(user_id=None):

    profiles = _load_json_list(
        PLANT_PROFILES_FILE
    )

    if user_id is None:
        return profiles

    return [
        profile
        for profile in profiles
        if str(profile.get("user_id", "")) == str(user_id)
    ]


def save_plant_profiles(profiles):

    return _save_json_list(
        PLANT_PROFILES_FILE,
        profiles
    )


def get_plant_profile(plant_id, user_id=None):

    profiles = _load_json_list(
        PLANT_PROFILES_FILE
    )

    for profile in profiles:

        if str(profile.get("id", "")) != str(plant_id):
            continue

        if (
            user_id is not None
            and
            str(profile.get("user_id", "")) != str(user_id)
        ):
            continue

        return profile

    return None


def _next_plant_number(user_id):

    # "Plant #001", "Plant #002", ... counted per account, in
    # the order plants were added. Sequential and stable - it
    # never changes even if an earlier plant is renamed.
    existing = load_plant_profiles(
        user_id=user_id
    )

    return "Plant #{:03d}".format(
        len(existing) + 1
    )


# ============================================================
# TREATMENT TRACKING
# Tied directly to a scan (saved_filename) - no plant profile
# is required. User-confirmed status only:
# 🟠 started -> 🟡 monitoring -> 🟢 completed
# The AI never sets or advances these statuses on its own.
# ============================================================

TREATMENT_STATUSES = (
    "started",
    "monitoring",
    "completed"
)


def load_treatment_records(user_id=None, plant_id=None):

    records = _load_json_list(
        TREATMENT_RECORDS_FILE
    )

    if user_id is not None:

        records = [
            record
            for record in records
            if str(record.get("user_id", "")) == str(user_id)
        ]

    if plant_id is not None:

        records = [
            record
            for record in records
            if str(record.get("plant_id", "")) == str(plant_id)
        ]

    return records


def save_treatment_records(records):

    return _save_json_list(
        TREATMENT_RECORDS_FILE,
        records
    )


def start_treatment(
    saved_filename,
    disease,
    plant_name,
    user_id
):

    """
    Creates a new "started" treatment record for a scan, unless
    that scan already has an active (not yet completed)
    treatment in progress - in which case the existing record is
    returned as-is instead of creating a duplicate.
    """

    records = _load_json_list(
        TREATMENT_RECORDS_FILE
    )

    for record in records:

        if (
            str(record.get("saved_filename", "")) == str(saved_filename)
            and
            str(record.get("user_id", "")) == str(user_id)
            and
            record.get("status") != "completed"
        ):

            return record

    new_record = {

        "id":
            uuid.uuid4().hex,

        "user_id":
            user_id,

        "saved_filename":
            saved_filename,

        "disease":
            disease,

        "plant_name":
            plant_name,

        "status":
            "started",

        "started_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "updated_at":
            datetime.now().isoformat(
                timespec="seconds"
            )
    }

    records.append(
        new_record
    )

    save_treatment_records(
        records
    )

    return new_record


def update_treatment_status(
    record_id,
    new_status,
    user_id
):

    if new_status not in TREATMENT_STATUSES:
        return None

    records = _load_json_list(
        TREATMENT_RECORDS_FILE
    )

    target = None

    for record in records:

        if (
            str(record.get("id", "")) == str(record_id)
            and
            str(record.get("user_id", "")) == str(user_id)
        ):

            record["status"] = new_status

            record["updated_at"] = datetime.now().isoformat(
                timespec="seconds"
            )

            target = record

            break

    if target is None:
        return None

    save_treatment_records(
        records
    )

    return target


def remove_treatment(
    record_id,
    user_id
):

    """
    Deletes a treatment record entirely (used by the
    "🗑️ Remove Treatment" button). Returns the removed record,
    or None if nothing matched.
    """

    records = _load_json_list(
        TREATMENT_RECORDS_FILE
    )

    removed = None
    remaining = []

    for record in records:

        if (
            str(record.get("id", "")) == str(record_id)
            and
            str(record.get("user_id", "")) == str(user_id)
        ):

            removed = record
            continue

        remaining.append(record)

    if removed is None:
        return None

    save_treatment_records(
        remaining
    )

    return removed


# ============================================================
# API - TREATMENT TRACKING
# ============================================================

@app.route(
    "/api/treatment/start",
    methods=["POST"]
)
def start_treatment_api():

    user = get_current_user()

    if not user:

        return jsonify({
            "success": False,
            "error": "Please login first."
        }), 401

    payload = request.get_json(
        silent=True
    ) or request.form

    saved_filename = payload.get(
        "saved_filename",
        ""
    )

    disease = payload.get(
        "disease",
        ""
    )

    plant_name = payload.get(
        "plant_name",
        ""
    )

    if not saved_filename:

        return jsonify({
            "success": False,
            "error": "saved_filename is required."
        }), 400

    record = start_treatment(
        saved_filename,
        disease,
        plant_name,
        user.get("id")
    )

    return jsonify({
        "success": True,
        "treatment": record
    }), 201


@app.route(
    "/api/treatment/<record_id>/status",
    methods=["POST"]
)
def update_treatment_status_api(record_id):

    user = get_current_user()

    if not user:

        return jsonify({
            "success": False,
            "error": "Please login first."
        }), 401

    payload = request.get_json(
        silent=True
    ) or request.form

    new_status = str(
        payload.get(
            "status",
            ""
        )
    ).strip().lower()

    updated = update_treatment_status(
        record_id,
        new_status,
        user.get("id")
    )

    if updated is None:

        return jsonify({
            "success": False,
            "error": "Treatment record not found, or invalid status."
        }), 404

    return jsonify({
        "success": True,
        "treatment": updated
    }), 200


@app.route(
    "/api/treatment/<record_id>",
    methods=["DELETE"]
)
def remove_treatment_api(record_id):

    user = get_current_user()

    if not user:

        return jsonify({
            "success": False,
            "error": "Please login first."
        }), 401

    removed = remove_treatment(
        record_id,
        user.get("id")
    )

    if removed is None:

        return jsonify({
            "success": False,
            "error": "Treatment record not found."
        }), 404

    return jsonify({
        "success": True,
        "treatment": removed
    }), 200


@app.route(
    "/api/treatment",
    methods=["GET"]
)
def list_treatment_api():

    user = get_current_user()

    if not user:

        return jsonify({
            "success": False,
            "error": "Please login first."
        }), 401

    return jsonify({
        "success": True,
        "treatments": load_treatment_records(
            user_id=user.get("id")
        )
    }), 200


# ============================================================
# SCAN -> PLANT LOOKUP
# (used by /health and /history to show which saved plant each
# scan belongs to, without changing anything in upload_records)
# ============================================================

def build_scan_to_plant_map(user_id):

    plants = load_plant_profiles(
        user_id=user_id
    )

    scan_map = {}

    for plant in plants:

        for scan in plant.get("scan_history", []) or []:

            saved_filename = scan.get(
                "saved_filename"
            )

            if not saved_filename:
                continue

            scan_map[saved_filename] = plant

    return scan_map


# ============================================================
# MY PLANT HEALTH (DASHBOARD)
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health_page():

    user = get_current_user()

    if not user:

        return redirect(
            url_for(
                "login_page",
                return_to=request.full_path
                if request.query_string
                else request.path
            )
        )

    user_id = user.get("id")

    # ========================================================
    # EXISTING MY RECORDS
    # READ ONLY - DO NOT MODIFY
    # ========================================================

    scans = load_collection_records(
        user_id=user_id
    )

    # ========================================================
    # COUNT HEALTHY / DISEASE SCANS
    # FROM EXISTING SCAN RECORDS
    # ========================================================

    healthy_count = 0
    diseases_found_count = 0

    for scan in scans:

        disease = scan.get(
            "predicted_disease",
            ""
        )

        if _is_healthy_disease(disease):

            healthy_count += 1

        else:

            diseases_found_count += 1

    # ========================================================
    # ACTIVE TREATMENTS
    # STARTED + MONITORING
    #
    # COMPLETED IS NOT COUNTED
    # ========================================================

    treatment_records = load_treatment_records(
        user_id=user_id
    )

    under_treatment_count = sum(
        1
        for record in treatment_records
        if str(
            record.get("status", "")
        ).lower()
        in (
            "started",
            "monitoring"
        )
    )

    # ========================================================
    # RECENT SCANS
    #
    # THIS READS YOUR EXISTING RECORDS ONLY
    # ========================================================

    recent_scans = list(
        reversed(scans)
    )[:5]

    # ========================================================
    # CONNECT EXISTING SCANS TO PLANT PROFILES
    #
    # THIS DOES NOT MODIFY THE SAVED RECORDS
    # ========================================================

    scan_to_plant = build_scan_to_plant_map(
        user_id
    )

    for scan in recent_scans:

        matched_plant = scan_to_plant.get(
            scan.get("saved_filename")
        )

        scan["plant_number"] = (
            matched_plant.get("plant_number")
            if matched_plant
            else None
        )

        scan["plant_display_name"] = (
            matched_plant.get("name")
            if matched_plant
            else None
        )

    # ========================================================
    # FILTERING
    #
    # Clicking a Health card opens the related records
    # ========================================================

    selected_filter = request.args.get(
        "filter",
        ""
    ).strip().lower()

    if selected_filter not in (
        "all",
        "healthy",
        "disease",
        "treatment"
    ):

        selected_filter = ""

    if selected_filter == "healthy":

        recent_scans = [
            scan
            for scan in reversed(scans)
            if _is_healthy_disease(
                scan.get(
                    "predicted_disease",
                    ""
                )
            )
        ][:10]

    elif selected_filter == "disease":

        recent_scans = [
            scan
            for scan in reversed(scans)
            if not _is_healthy_disease(
                scan.get(
                    "predicted_disease",
                    ""
                )
            )
        ][:10]

    elif selected_filter == "all":

        recent_scans = list(
            reversed(scans)
        )[:10]

    # ========================================================
    # ADD PLANT INFORMATION TO FILTERED RECORDS
    #
    # AGAIN: ONLY TEMPORARY DISPLAY DATA
    # ========================================================

    for scan in recent_scans:

        matched_plant = scan_to_plant.get(
            scan.get("saved_filename")
        )

        scan["plant_number"] = (
            matched_plant.get("plant_number")
            if matched_plant
            else None
        )

        scan["plant_display_name"] = (
            matched_plant.get("name")
            if matched_plant
            else None
        )

    # ========================================================
    # FULL TREATMENT DISPLAY
    # Every treatment record (started, monitoring AND
    # completed) enriched with the same home-care / first-aid /
    # prevention guidance shown on the scan result page, so the
    # dashboard is a complete treatment center rather than just
    # a status list.
    # ========================================================

    treatment_details = []

    for record in sorted(
        treatment_records,
        key=lambda r: r.get("started_at", ""),
        reverse=True
    ):

        status = str(
            record.get(
                "status",
                ""
            )
        ).lower()

        if status not in TREATMENT_STATUSES:
            continue

        disease_raw = record.get(
            "disease",
            ""
        )

        treatment_info = find_treatment(
            disease_raw
        )

        treatment_details.append({

            "id":
                record.get("id"),

            "plant_id":
                record.get(
                    "saved_filename"
                ),

            "plant_name":
                record.get(
                    "plant_name"
                )
                or "Plant",

            "disease":
                clean_label(
                    disease_raw
                ),

            "status":
                status,

            "started_at":
                record.get(
                    "started_at",
                    ""
                ),

            "updated_at":
                record.get(
                    "updated_at",
                    ""
                ),

            "emoji":
                treatment_info.get(
                    "emoji",
                    "🌿"
                ),

            # Full treatment database entry for the Treatment Center UI.
            # The template can read all treatment fields without changing
            # the saved treatment record itself.
            "treatment_info":
                treatment_info,

            "home_care":
                treatment_info.get(
                    "home_care"
                ),

            "what_to_do_first":
                treatment_info.get(
                    "what_to_do_first"
                ),

            "prevention":
                treatment_info.get(
                    "prevention"
                ),

            "professional_treatment_guidance":
                treatment_info.get(
                    "professional_treatment_guidance"
                ),

            "label_explanation":
                treatment_info.get(
                    "label_explanation"
                ),

            "common_treatment_options":
                treatment_info.get(
                    "common_treatment_options"
                ),

            "example_products":
                treatment_info.get(
                    "example_products"
                ),

            "example_products_disclaimer":
                treatment_info.get(
                    "example_products_disclaimer"
                ),

            "plain_language_help":
                treatment_info.get(
                    "plain_language_help"
                )
        })

    # ========================================================
    # RENDER HEALTH DASHBOARD
    # ========================================================

    return render_template(
        "health.html",

        # FINAL FOUR DASHBOARD VALUES
        total_scans=len(scans),

        healthy_count=healthy_count,

        diseases_found_count=
            diseases_found_count,

        under_treatment_count=
            under_treatment_count,

        # DISPLAY DATA
        recent_scans=recent_scans,

        treatment_details=
            treatment_details,

        selected_filter=
            selected_filter
    )


# ============================================================
# TREATMENT DETAILS PAGE
# ============================================================
@app.route(
    "/treatment/<record_id>",
    methods=["GET"]
)
def treatment_detail_page(record_id):
    """Show one complete treatment record on its own readable page."""
    user = get_current_user()

    if not user:
        return redirect(
            url_for(
                "login_page",
                return_to=request.full_path
                if request.query_string
                else request.path
            )
        )

    user_id = user.get("id")
    records = load_treatment_records(user_id=user_id)

    record = None
    for item in records:
        if str(item.get("id", "")) == str(record_id):
            record = item
            break

    if record is None:
        return "Treatment record not found.", 404

    disease_raw = record.get("disease", "")
    treatment_info = find_treatment(disease_raw) or {}

    treatment_detail = {
        "id": record.get("id"),
        "plant_id": record.get("saved_filename", ""),
        "plant_name": record.get("plant_name") or "Plant",
        "disease": clean_label(disease_raw),
        "status": str(record.get("status", "started")).lower(),
        "started_at": record.get("started_at", ""),
        "updated_at": record.get("updated_at", ""),
        "emoji": treatment_info.get("emoji", "🌿"),
        "treatment_info": treatment_info,
        "image_path": (
            "/static/uploads/" + str(record.get("saved_filename"))
            if record.get("saved_filename")
            else ""
        )
    }

    return render_template(
        "treatment_details.html",
        item=treatment_detail
    )


# ============================================================
# SCAN HISTORY (PER PLANT)
# ============================================================

@app.route(
    "/history",
    methods=["GET"]
)
def history_page():

    user = get_current_user()

    if not user:

        return redirect(
            url_for(
                "login_page",
                return_to=request.full_path
                if request.query_string
                else request.path
            )
        )

    user_id = user.get("id")

    scans = list(
        reversed(
            load_collection_records(
                user_id=user_id
            )
        )
    )

    scan_to_plant = build_scan_to_plant_map(
        user_id
    )

    for scan in scans:

        matched_plant = scan_to_plant.get(
            scan.get("saved_filename")
        )

        scan["plant_number"] = (
            matched_plant.get("plant_number")
            if matched_plant
            else None
        )

        scan["plant_display_name"] = (
            matched_plant.get("name")
            if matched_plant
            else None
        )

        scan["plant_type"] = (
            matched_plant.get("plant_type")
            if matched_plant
            else None
        )

    return render_template(
        "history.html",

        scans=scans,

        SUPPORTED_PLANTS=SUPPORTED_PLANTS
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("🌱 PLANT HEALTH ASSISTANT")
    print("=" * 60)

    print(
        "🌿 Model:",
        os.path.basename(
            MODEL_PATH
        )
    )

    print(
        "📚 Classes:",
        len(index_to_class)
    )

    print(
        "📋 Treatments:",
        len(treatments)
    )

    print(
        "💬 Chat:",
        "AVAILABLE"
    )

    print(
        "🧠 Chat modes:",
        ", ".join(
            MENTALITIES.keys()
        )
    )

    print(
        "🌐 Local access:",
        "http://127.0.0.1:5000"
    )

    print("=" * 60)
    print()


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )