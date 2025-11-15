import io
import os
import sys
import time
import uuid
from datetime import datetime

import bcrypt
import cv2
import jwt
import numpy as np
from bson.objectid import ObjectId
from flask import Flask, jsonify, request, send_from_directory, url_for
from flask_cors import CORS
from gridfs import GridFS
from PIL import Image
from pymongo import MongoClient
import logging

# Check if running on Render (production) to avoid loading heavy ML libs at startup
# to prevent OOM on small instances
_is_render = 'RENDER' in os.environ
_is_production = os.getenv('ENV', '').lower() == 'production' or _is_render

# Conditionally import heavy ML libraries
# On production/Render, these will be lazy-loaded on first use
if not _is_production:
    import torch
    import torch.nn as nn
    from torchvision import transforms
else:
    torch = None
    nn = None
    transforms = None

_diffusers_available = False
_diffusers_err = None


def _lazy_load_torch():
    """Lazily import torch and related modules on first use."""
    global torch, nn, transforms
    if torch is None:
        try:
            import torch as _torch
            import torch.nn as _nn
            from torchvision import transforms as _transforms
            torch = _torch
            nn = _nn
            transforms = _transforms
        except Exception as e:
            raise RuntimeError(f"Failed to load torch: {e}")
    return torch, nn, transforms


def _lazy_load_diffusers():
    """Lazily import diffusers on first use."""
    global _diffusers_available, _diffusers_err
    if _diffusers_available:
        return
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        _diffusers_available = True
        _diffusers_err = None
    except Exception as _e:
        _diffusers_available = False
        _diffusers_err = str(_e)

# -----------------------\n# Flask Setup\n# -----------------------
app = Flask(__name__, static_folder='static/frontend', static_url_path='')

# Import and register routes (use package-qualified imports so imports work
# when gunicorn loads `backend.app` as a module)
from backend.timeline_routes import timeline_bp
from backend.history_routes import history_bp

app.register_blueprint(timeline_bp, url_prefix='/timeline')
app.register_blueprint(history_bp, url_prefix='/history')

# Allow credentials (cookies) for cross-origin requests from frontend
# Read allowed origins from FRONTEND_ORIGINS env var (comma-separated) or default common Vite ports
frontend_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5000,http://127.0.0.1:5000"
).split(",")
CORS(
    app,
    supports_credentials=True,
    origins=[o.strip() for o in frontend_origins if o.strip()],
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"]
)


# --- Create folders for static files and predictions ---
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), "static")
PREDICTION_FOLDER = os.path.join(STATIC_FOLDER, "predictions")
os.makedirs(PREDICTION_FOLDER, exist_ok=True)
GENERATED_FOLDER = os.path.join(STATIC_FOLDER, "generated_stages")
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# -----------------------\n# MongoDB Setup\n# -----------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["agri_llava"]
history_collection = db["history"]
timelines_collection = db["timelines"]
fs = GridFS(db)
users_collection = db["users"]
sessions_collection = db["sessions"]

JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_EXPIRE_SECONDS = int(os.getenv("JWT_EXPIRE_SECONDS", "604800"))  # 7 days
SMOOTH_SAMPLES = int(os.getenv("SMOOTH_SAMPLES", "8"))
SMOOTH_STD = float(os.getenv("SMOOTH_STD", "0.15"))

# configure simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agri_llava")


def _parse_smooth_params(form):
    """Parse smooth parameters from request form (if provided) with safe fallbacks."""
    n_samples = SMOOTH_SAMPLES
    stdev = SMOOTH_STD
    colormap = "inferno"
    try:
        if form is not None:
            if form.get("smooth_samples"):
                try:
                    n_samples = int(form.get("smooth_samples"))
                except Exception:
                    n_samples = SMOOTH_SAMPLES
            if form.get("smooth_std"):
                try:
                    stdev = float(form.get("smooth_std"))
                except Exception:
                    stdev = SMOOTH_STD
            if form.get("colormap"):
                colormap = str(form.get("colormap"))
    except Exception:
        pass
    return n_samples, stdev, colormap


_COLORMAP_MAP = {
    "inferno": cv2.COLORMAP_INFERNO,
    "jet": cv2.COLORMAP_JET,
    "plasma": cv2.COLORMAP_PLASMA
    if hasattr(cv2, "COLORMAP_PLASMA")
    else cv2.COLORMAP_JET,
}


def hash_password(plain: str) -> bytes:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt())


def verify_password(plain: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed)
    except Exception:
        return False


def create_session(user_id: str) -> str:
    # create a simple session in Mongo and return session id
    sid = str(uuid.uuid4())
    now = int(time.time())
    expires = now + JWT_EXPIRE_SECONDS
    sessions_collection.insert_one(
        {"_id": sid, "user_id": str(user_id), "created_at": now, "expires_at": expires}
    )
    return sid


def get_current_user_id() -> str | None:
    sid = request.cookies.get("session_id")
    if not sid:
        return None
    try:
        s = sessions_collection.find_one({"_id": sid})
        if not s:
            return None
        # optional: check expiry
        if s.get("expires_at", 0) < int(time.time()):
            sessions_collection.delete_one({"_id": sid})
            return None
        return s.get("user_id")
    except Exception:
        return None


def auth_required():
    uid = get_current_user_id()
    if not uid:
        return None, (jsonify({"error": "Unauthorized"}), 401)
    return uid, None


@app.route("/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not name or not email or not password:
        return jsonify({"error": "name, email, password are required"}), 400
    if users_collection.find_one({"email": email}):
        return jsonify({"error": "Email already registered"}), 409
    hashed = hash_password(password)
    doc = {"name": name, "email": email, "password": hashed}
    res = users_collection.insert_one(doc)
    # create session and set cookie
    sid = create_session(str(res.inserted_id))
    resp = jsonify({"user": {"id": str(res.inserted_id), "name": name, "email": email}})
    resp.set_cookie("session_id", sid, httponly=True, samesite="Lax")
    return resp


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400
    user = users_collection.find_one({"email": email})
    if not user or not verify_password(password, user.get("password", b"")):
        return jsonify({"error": "Invalid credentials"}), 401
    # create session and set cookie
    sid = create_session(str(user["_id"]))
    resp = jsonify(
        {"user": {"id": str(user["_id"]), "name": user.get("name", ""), "email": email}}
    )
    resp.set_cookie("session_id", sid, httponly=True, samesite="Lax")
    return resp


@app.route("/auth/me", methods=["GET"])
def auth_me():
    uid = get_current_user_id()
    if not uid:
        return jsonify({"user": None}), 200
    user = users_collection.find_one({"_id": ObjectId(uid)})
    if not user:
        return jsonify({"user": None}), 200
    return jsonify(
        {
            "user": {
                "id": str(user["_id"]),
                "name": user.get("name", ""),
                "email": user.get("email", ""),
            }
        }
    )


@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    sid = request.cookies.get("session_id")
    if sid:
        sessions_collection.delete_one({"_id": sid})
    resp = jsonify({"ok": True})
    resp.set_cookie("session_id", "", expires=0)
    return resp


# -----------------------\n# CNN Model Definition (lazy-loaded)\n# -----------------------
_LeafCNN = None


def get_LeafCNN():
    """Lazily define and return the LeafCNN class."""
    global _LeafCNN
    if _LeafCNN is not None:
        return _LeafCNN
    
    _torch, _nn, _ = _lazy_load_torch()
    
    class LeafCNN(_nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv = _nn.Sequential(
                _nn.Conv2d(3, 32, 3, padding=1),
                _nn.BatchNorm2d(32),
                _nn.ReLU(),
                _nn.MaxPool2d(2),
                _nn.Conv2d(32, 64, 3, padding=1),
                _nn.BatchNorm2d(64),
                _nn.ReLU(),
                _nn.MaxPool2d(2),
                _nn.Conv2d(64, 128, 3, padding=1),
                _nn.BatchNorm2d(128),
                _nn.ReLU(),
                _nn.MaxPool2d(2),
            )
            self.fc = _nn.Sequential(
                _nn.Linear(128 * 16 * 16, 256),
                _nn.ReLU(),
                _nn.Dropout(0.4),
                _nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    _LeafCNN = LeafCNN
    return _LeafCNN


# ---------------------------------\n# Grad-CAM Function (Helper)\n# ---------------------------------
def grad_cam(model, image_tensor, class_idx=None):
    model.eval()
    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        # grad_out[0] is the gradient w.r.t. the output of the conv layer
        gradients = grad_out[0]

    last_conv = None
    for module in model.conv.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise RuntimeError("Could not find a Conv2d layer in model.conv")

    handle_f = last_conv.register_forward_hook(forward_hook)
    # register_full_backward_hook works for modern PyTorch and captures gradients reliably
    handle_b = last_conv.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    if class_idx is None:
        class_idx = output.argmax().item()

    model.zero_grad()
    # Backprop the target class score
    output[0, class_idx].backward()

    handle_f.remove()
    handle_b.remove()

    if gradients is None or features is None:
        raise RuntimeError("Gradients or features were not captured by hooks")

    # Global-average-pool the gradients over spatial dims
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Weight the channels by the pooled gradients
    # features: (N, C, H, W) -> take detached features
    weighted = features.detach() * pooled_gradients.view(1, -1, 1, 1)

    # Sum across channels and convert to numpy
    heatmap = weighted[0].sum(dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)

    return heatmap, class_idx


def smooth_grad_cam(
    model, image_tensor, class_idx=None, n_samples: int = 8, stdev_spread: float = 0.15):
    """
    Compute an averaged Grad-CAM heatmap over multiple noisy perturbations (SmoothGrad-like).
    Returns averaged heatmap and the class index used.
    """
    model.eval()
    accumulated = None
    used_idx = class_idx
    # Ensure tensors on same device
    image_tensor = image_tensor.detach().to(device)
    for i in range(max(1, int(n_samples))):
        # add Gaussian noise scaled by stdev_spread * (max - min) approx; since input is normalized, use stdev_spread
        noise = torch.randn_like(image_tensor) * stdev_spread
        noisy = (image_tensor + noise).clamp(-3.0, 3.0)
        try:
            heatmap_i, idx_i = grad_cam(model, noisy, class_idx=used_idx)
        except Exception:
            # fallback to calling grad_cam on original if noisy fails
            heatmap_i, idx_i = grad_cam(model, image_tensor, class_idx=used_idx)
        if used_idx is None:
            used_idx = idx_i
        if accumulated is None:
            accumulated = np.array(heatmap_i, dtype=float)
        else:
            accumulated += np.array(heatmap_i, dtype=float)
    avg = accumulated / float(max(1, int(n_samples)))
    # ensure non-negative and normalized
    avg = np.maximum(avg, 0)
    if np.max(avg) > 0:
        avg = avg / np.max(avg)
    return avg, used_idx


# ---------------------------------\n# Severity computation (Helper)\n# ---------------------------------
def compute_severity_percent(heatmap_uint8: np.ndarray) -> float:
    # Try Otsu first
    try:
        if heatmap_uint8.dtype != np.uint8:
            heatmap_uint8 = np.uint8(np.clip(heatmap_uint8, 0, 255))
        _, mask = cv2.threshold(
            heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        diseased_pixels = int((mask > 0).sum())
        total_pixels = int(mask.size)
        if total_pixels == 0:
            return 0.0
        if diseased_pixels > 0:
            return (diseased_pixels / total_pixels) * 100.0
    except Exception:
        # fall through to percentile fallback
        diseased_pixels = 0

    # Fallback attempts: try high percentile(s) then a fixed threshold
    try:
        for p in (95, 90):
            thresh = np.percentile(heatmap_uint8.ravel(), p)
            if thresh <= 0:
                continue
            mask2 = heatmap_uint8 >= thresh
            diseased_pixels = int((mask2 > 0).sum())
            total_pixels = int(heatmap_uint8.size)
            if total_pixels == 0:
                return 0.0
            if diseased_pixels > 0:
                return (diseased_pixels / total_pixels) * 100.0
        # Final fallback: simple mid-level threshold
        mask3 = heatmap_uint8 >= 128
        diseased_pixels = int((mask3 > 0).sum())
        total_pixels = int(heatmap_uint8.size)
        if total_pixels == 0:
            return 0.0
        return (diseased_pixels / total_pixels) * 100.0
    except Exception:
        return 0.0


def compute_diseased_mask(heatmap_uint8: np.ndarray) -> tuple:
    """
    Return (mask_uint8, diseased_pixels) for a heatmap. Tries Otsu, percentiles, and fixed threshold.
    """
    try:
        if heatmap_uint8.dtype != np.uint8:
            heatmap_uint8 = np.uint8(np.clip(heatmap_uint8, 0, 255))
        # try Otsu
        _, mask = cv2.threshold(
            heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        mask = mask.astype(np.uint8)
        diseased_pixels = int((mask > 0).sum())
        if diseased_pixels > 0:
            # ensure small noise removed
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            return (mask, int((mask > 0).sum()))
    except Exception:
        pass

    # try percentiles then fixed (wider set of percentiles to avoid empty masks)
    # include more permissive percentiles (75,70) for aggressive detection when needed
    try:
        for p in (95, 90, 85, 80, 75, 70):
            thresh = np.percentile(heatmap_uint8.ravel(), p)
            if thresh <= 0:
                continue
            mask2 = (heatmap_uint8 >= thresh).astype(np.uint8) * 255
            # small morphological clean
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=1)
            diseased_pixels = int((mask2 > 0).sum())
            if diseased_pixels > 0:
                return (mask2.astype(np.uint8), diseased_pixels)
        # Final fallback: try a lower fixed threshold before giving up
        for fixed in (128, 110, 100):
            mask3 = (heatmap_uint8 >= fixed).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, kernel, iterations=1)
            if int((mask3 > 0).sum()) > 0:
                return (mask3.astype(np.uint8), int((mask3 > 0).sum()))
        return (np.zeros_like(heatmap_uint8, dtype=np.uint8), 0)
    except Exception:
        return (np.zeros_like(heatmap_uint8, dtype=np.uint8), 0)


def segment_leaf_hsv(bgr_np: np.ndarray, out_dir: str) -> dict:
    """
    Segment the leaf using HSV-based green mask and create a dimmed-background overlay.
    Returns dict with relative 'mask' and 'overlay' paths (under static/predictions).
    """
    # Convert to HSV and create a broad green mask (tunable)
    hsv = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2HSV)
    lower1 = np.array([25, 40, 20])
    upper1 = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1)

    # Morphological cleanup
    # Use a slightly smaller kernel to avoid removing small leaf regions (more permissive)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Save mask image
    mask_filename = f"mask_{uuid.uuid4()}.png"
    mask_path_abs = os.path.join(out_dir, mask_filename)
    cv2.imwrite(mask_path_abs, mask)

    # Create overlay: dim background (where mask==0) and keep leaf area bright
    colored = bgr_np.copy()
    # reduce background brightness to 25%
    colored[mask == 0] = (colored[mask == 0] * 0.25).astype(colored.dtype)

    overlay_filename = f"seg_{uuid.uuid4()}.jpg"
    overlay_path_abs = os.path.join(out_dir, overlay_filename)
    cv2.imwrite(overlay_path_abs, colored)

    # Compute simple health metrics from the mask and HSV
    total_pixels = mask.size
    leaf_pixels = int((mask > 0).sum())
    leaf_area_percent = (leaf_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    # Mean saturation and mean green channel within leaf area
    sat = hsv[:, :, 1]
    mean_saturation = float(sat[mask > 0].mean()) if leaf_pixels > 0 else 0.0
    mean_green = float(bgr_np[:, :, 1][mask > 0].mean()) if leaf_pixels > 0 else 0.0

    return {
        "mask": os.path.join("static", "predictions", mask_filename).replace("\\", "/"),
        "overlay": os.path.join("static", "predictions", overlay_filename).replace("\\", "/"),
        "metrics": {
            "leaf_pixels": leaf_pixels,
            "leaf_area_percent": leaf_area_percent,
            "mean_saturation": mean_saturation,
            "mean_green": mean_green,
        },
    }


def align_images(fixed_bgr: np.ndarray, moving_bgr: np.ndarray, fixed_mask: np.ndarray = None, moving_mask: np.ndarray = None):
    """
    Align moving_bgr to fixed_bgr. Try ECC (intensity-based) first using masks as ROI, fallback to ORB+RANSAC homography.
    Returns (aligned_moving_bgr, transform_matrix) where transform_matrix is a 3x3 homography or 2x3 warp.
    """
    try:
        # convert to gray
        fixed_gray = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2GRAY)
        moving_gray = cv2.cvtColor(moving_bgr, cv2.COLOR_BGR2GRAY)

        # define warp_mode
        warp_mode = cv2.MOTION_AFFINE
        if fixed_gray.shape != moving_gray.shape:
            moving_gray = cv2.resize(moving_gray, (fixed_gray.shape[1], fixed_gray.shape[0]))
            moving_bgr = cv2.resize(moving_bgr, (fixed_gray.shape[1], fixed_gray.shape[0]))

        # initial warp
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # If masks provided, use them to weight the ECC
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-7)
            if fixed_mask is not None and moving_mask is not None:
                # Ensure masks same size
                if fixed_mask.shape != fixed_gray.shape:
                    fixed_mask = cv2.resize(fixed_mask, (fixed_gray.shape[1], fixed_gray.shape[0]))
                if moving_mask.shape != moving_gray.shape:
                    moving_mask = cv2.resize(moving_mask, (fixed_gray.shape[1], fixed_gray.shape[0]))
                mask = (fixed_mask > 0).astype(np.uint8)
                try:
                    (cc, warp_matrix) = cv2.findTransformECC(
                        fixed_gray, moving_gray, warp_matrix, warp_mode, criteria, inputMask=mask, gaussFiltSize=5
                    )
                except Exception:
                    # fallback to no mask
                    (cc, warp_matrix) = cv2.findTransformECC(fixed_gray, moving_gray, warp_matrix, warp_mode, criteria, gaussFiltSize=5)
            else:
                (cc, warp_matrix) = cv2.findTransformECC(fixed_gray, moving_gray, warp_matrix, warp_mode, criteria, gaussFiltSize=5)

            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                aligned = cv2.warpPerspective(moving_bgr, warp_matrix, (fixed_gray.shape[1], fixed_gray.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            else:
                aligned = cv2.warpAffine(moving_bgr, warp_matrix, (fixed_gray.shape[1], fixed_gray.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            return aligned, warp_matrix
        except Exception:
            raise
    except Exception:
        # fallback: ORB + RANSAC
        try:
            orb = cv2.ORB_create(5000)
            kp1, des1 = orb.detectAndCompute(fixed_gray, None)
            kp2, des2 = orb.detectAndCompute(moving_gray, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            if des1 is None or des2 is None:
                return moving_bgr, None
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) < 8:
                return moving_bgr, None
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if H is None:
                return moving_bgr, None
            aligned = cv2.warpPerspective(moving_bgr, H, (fixed_gray.shape[1], fixed_gray.shape[0]))
            return aligned, H
        except Exception:
            return moving_bgr, None


def predict_trend_from_series(values: list[float]) -> dict:
    """
    Lightweight linear trend predictor using numpy.polyfit degree=1.
    Returns predicted next value, slope, intercept, and simple stderr/confidence based on residuals.
    """
    import math

    result = {"predicted": None, "slope": None, "intercept": None, "stderr": None, "r2": None}
    try:
        if not values or len(values) < 2:
            return result
        x = np.arange(len(values)).astype(float)
        y = np.array(values).astype(float)
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        pred_x = float(len(values))
        predicted = slope * pred_x + intercept
        # residuals
        fitted = slope * x + intercept
        resid = y - fitted
        stderr = float(np.std(resid))
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum()) if ss_res + 1e-12 else 0.0
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
        result.update({"predicted": predicted, "slope": slope, "intercept": intercept, "stderr": stderr, "r2": r2})
        return result
    except Exception:
        return result


def detect_dark_lesions(bgr_np: np.ndarray, leaf_mask: np.ndarray = None) -> tuple:
    """
    Detect dark/brown necrotic lesions (useful for late blight) by thresholding the grayscale values
    relative to the leaf region. Returns (mask_uint8, diseased_pixels).
    """
    try:
        gray = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2GRAY)
        if leaf_mask is not None and leaf_mask.shape == gray.shape:
            vals = gray[leaf_mask > 0]
            if vals.size == 0:
                return (np.zeros_like(gray, dtype=np.uint8), 0)
            # threshold at the 40th percentile of leaf gray values to capture darker lesions
            thresh = int(np.percentile(vals, 40))
            mask = (gray <= thresh).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, (leaf_mask > 0).astype(np.uint8) * 255)
        else:
            thresh = int(np.percentile(gray.ravel(), 40))
            mask = (gray <= thresh).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        diseased_pixels = int((mask > 0).sum())
        return (mask.astype(np.uint8), diseased_pixels)
    except Exception:
        return (np.zeros_like(bgr_np[:, :, 0], dtype=np.uint8), 0)


def analyze_leaf_simple_from_np(img_np: np.ndarray) -> dict:
    """
    Lightweight segmentation-based analyzer adapted from the provided snippet.
    Input: BGR numpy array.
    Returns disease_type, severity_fraction (0..1), severity_percent (0..100), and metrics dict.
    """
    try:
        hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)

        # Segment leaf
        lower_leaf = np.array([25, 40, 20])
        upper_leaf = np.array([85, 255, 255])
        mask_leaf = cv2.inRange(hsv, lower_leaf, upper_leaf)
        leaf_area = int(np.count_nonzero(mask_leaf))
        if leaf_area == 0:
            return {"severity_fraction": 0.0, "severity_percent": 0.0, "error": "No leaf detected"}

        # Detect diseased area (brown/yellow ranges)
        lower_disease = np.array([10, 50, 20])
        upper_disease = np.array([35, 255, 255])
        mask_disease = cv2.inRange(hsv, lower_disease, upper_disease)
        # restrict disease mask to within leaf
        mask_disease = cv2.bitwise_and(mask_disease, mask_leaf)
        disease_area = int(np.count_nonzero(mask_disease))

        # Severity
        severity_fraction = float(disease_area) / float(leaf_area) if leaf_area > 0 else 0.0
        severity_percent = severity_fraction * 100.0

        # Color health index (ratio of green leaf pixels to total image pixels)
        green_pixels = int(np.count_nonzero(mask_leaf))
        total_pixels = img_np.shape[0] * img_np.shape[1]
        color_health_index = float(green_pixels) / float(total_pixels) if total_pixels > 0 else 0.0

        disease_type = "Leaf Spot" if severity_fraction > 0.25 else "Healthy"

        return {
            "disease_type": disease_type,
            "severity_fraction": severity_fraction,
            "severity_percent": round(severity_percent, 2),
            "metrics": {
                "affected_area_percent": round(100 * severity_fraction, 2),
                "color_health_index": round(color_health_index, 3),
                "leaf_pixels": leaf_area,
                "diseased_pixels": disease_area,
            },
        }
    except Exception as e:
        return {"error": str(e)}


def compare_with_previous_entry(prev_entry: dict, current_metrics: dict) -> dict:
    """
    Compare current_metrics (from analyze_leaf_simple_from_np) with a previous timeline entry.
    Uses thresholds similar to the snippet: change > 0.05 (fraction) -> worsened, < -0.05 -> improved.
    Returns previous/current severities (percent) and status.
    """
    try:
        prev_sev_percent = None
        # try health_metrics first
        hm = prev_entry.get("health_metrics") or {}
        if hm and hm.get("diseased_percent_of_leaf") is not None:
            prev_sev_percent = float(hm.get("diseased_percent_of_leaf"))
        elif prev_entry.get("severity_percent") is not None:
            prev_sev_percent = float(prev_entry.get("severity_percent"))
        else:
            prev_sev_percent = 0.0

        curr_sev_percent = float(current_metrics.get("severity_percent", 0.0))
        prev_frac = prev_sev_percent / 100.0
        curr_frac = curr_sev_percent / 100.0
        change_frac = round(curr_frac - prev_frac, 3)
        change_percent = round(curr_sev_percent - prev_sev_percent, 2)

        if change_frac < -0.05:
            status = "Improved"
        elif change_frac > 0.05:
            status = "Worsened"
        else:
            status = "Stable"

        return {
            "previous_severity_percent": round(prev_sev_percent, 2),
            "current_severity_percent": round(curr_sev_percent, 2),
            "change_in_percent_points": change_percent,
            "change_fraction": change_frac,
            "status": status,
        }
    except Exception as e:
        return {"error": str(e)}


@app.route("/timeline/analyze_simple", methods=["POST"])
def timeline_analyze_simple():
    """Endpoint: Upload an image and get back simple analysis and optional comparison to a previous entry id."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    prev_id = request.form.get("prev_entry_id", None)
    file = request.files["image"]
    try:
        image = Image.open(file.stream).convert("RGB")
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    analysis = analyze_leaf_simple_from_np(img_np)
    result = {"analysis": analysis}
    if prev_id:
        try:
            prev = timelines_collection.find_one({"_id": ObjectId(prev_id)})
            if prev:
                cmp = compare_with_previous_entry(prev, analysis)
                result["comparison"] = cmp
        except Exception:
            result["comparison_error"] = "Could not fetch previous entry"

    return jsonify(result)


def trend_from_series(values: list[float], eps: float = 1.0) -> str:
    # Compare last two values with tolerance eps (% points)
    if len(values) < 2:
        return "insufficient"
    prev, curr = values[-2], values[-1]
    if curr - prev > eps:
        return "increasing"
    if prev - curr > eps:
        return "decreasing"
    return "stable"


def _normalize_entry_percentages(entry: dict) -> dict:
    """
    Ensure severity_percent and health_metrics percent fields are in 0..100 range
    (some older code or analysis helpers may have returned fractions 0..1).
    This mutates the entry in-place and returns it.
    """
    try:
        if not isinstance(entry, dict):
            return entry
        sp = entry.get("severity_percent", None)
        if sp is not None:
            try:
                spf = float(sp)
                if spf <= 1.0:
                    spf = spf * 100.0
                entry["severity_percent"] = float(spf)
            except Exception:
                pass
        hm = entry.get("health_metrics") or {}
        if isinstance(hm, dict):
            dp = hm.get("diseased_percent_of_leaf", None)
            if dp is not None:
                try:
                    dpf = float(dp)
                    if dpf <= 1.0:
                        dpf = dpf * 100.0
                    hm["diseased_percent_of_leaf"] = float(dpf)
                except Exception:
                    pass
            # leaf_area_percent should already be 0..100, but guard anyway
            lap = hm.get("leaf_area_percent", None)
            if lap is not None:
                try:
                    lapf = float(lap)
                    if lapf <= 1.0:
                        lapf = lapf * 100.0
                    hm["leaf_area_percent"] = float(lapf)
                except Exception:
                    pass
            entry["health_metrics"] = hm
    except Exception:
        pass
    return entry


# -----------------------\n# Model Loading (Lazy)\n# -----------------------
_model_device = None
_model = None
_transform = None

class_names = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
num_classes = len(class_names)
model_path = os.path.join(os.path.dirname(__file__), "cnn_leaf_model.pth")


def get_device():
    """Get torch device (CPU or GPU)."""
    global _model_device
    if _model_device is None:
        _torch, _, _ = _lazy_load_torch()
        _model_device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    return _model_device


def get_model():
    """Lazily load the CNN model."""
    global _model
    if _model is not None:
        return _model
    
    _torch, _, _ = _lazy_load_torch()
    LeafCNN = get_LeafCNN()
    device = get_device()
    
    _model = LeafCNN(num_classes=num_classes).to(device)
    _model.load_state_dict(_torch.load(model_path, map_location=device))
    _model.eval()
    return _model


def get_transform():
    """Get image transform."""
    global _transform
    if _transform is not None:
        return _transform
    
    _, _, _transforms = _lazy_load_torch()
    _transform = _transforms.Compose(
        [
            _transforms.Resize((128, 128)),
            _transforms.ToTensor(),
            _transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return _transform


# -------------------------------------------------\n# --- This is the single /predict Endpoint ---\n# --- It matches your React component's request ---\n# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # On Render production, torch is not loaded to save memory
    # Return a graceful error message
    if _is_production and torch is None:
        return jsonify({
            "error": "Disease detection is not available on this deployment (requires GPU). "
                     "Please use a local version or contact support for ML features."
        }), 503
    
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    try:
        image = Image.open(file.stream).convert("RGB")
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_np_float = img_np.astype(float) / 255.0
        img_tensor = transform(image).unsqueeze(0).to(device)

    except Exception as e:
        return jsonify({"error": f"Invalid image file: {e}"}), 400

    # 1. --- Run Prediction ---
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence_tensor, predicted_index_tensor = torch.max(probs, 1)

        label = class_names[predicted_index_tensor.item()]
        confidence = confidence_tensor.item()
        confidence_percent = f"{confidence * 100:.2f}%"
        predicted_index = predicted_index_tensor.item()

    # --- Initialize response fields ---
    heatmap_path = None  # Will be null if healthy
    probabilities_dict = {
        class_names[i]: float(probs[0][i]) for i in range(num_classes)
    }

    # 2. --- Only run Grad-CAM if NOT healthy ---
    if "healthy" not in label.lower():
        try:
            # 3. --- Generate Grad-CAM (smoothed via multiple noisy passes) ---
            n_samples, stdev, colormap = _parse_smooth_params(request.form)
            heatmap, _ = smooth_grad_cam(
                model,
                img_tensor,
                class_idx=predicted_index,
                n_samples=n_samples,
                stdev_spread=stdev,
            )

            # 4. --- Post-process heatmap for better visualization ---
            # Resize to original image size
            heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

            # Clip extremes using percentiles to remove outliers, then smooth
            p_low, p_high = np.percentile(heatmap, [5, 99])
            if p_high - p_low > 1e-6:
                heatmap = np.clip((heatmap - p_low) / (p_high - p_low), 0.0, 1.0)
            else:
                heatmap = np.clip(heatmap, 0.0, 1.0)

            # Slight Gaussian blur to reduce noise
            try:
                heatmap = (
                    cv2.GaussianBlur((heatmap * 255).astype(np.uint8), (7, 7), 0)
                    / 255.0
                )
            except Exception:
                heatmap = np.uint8(255 * heatmap)

            # Convert to 8-bit and apply a perceptual colormap (default: inferno)
            heatmap_uint8 = np.uint8(255 * heatmap)
            cmap = _COLORMAP_MAP.get(colormap, cv2.COLORMAP_INFERNO)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cmap)

            # Blend overlay with original using alpha composite for clearer emphasis
            alpha = 0.6
            overlay = (alpha * heatmap_color.astype(float) / 255.0) + (
                (1 - alpha) * img_np_float
            )
            overlay = np.clip(overlay, 0, 1)
            overlay = np.uint8(overlay * 255)

            filename = f"{uuid.uuid4()}.jpg"
            save_path = os.path.join(PREDICTION_FOLDER, filename)
            cv2.imwrite(save_path, overlay)

            # 5. --- Create relative path for React ---
            # This makes the path "static/predictions/filename.jpg"
            heatmap_path = os.path.join("static", "predictions", filename).replace(
                "\\", "/"
            )

            # 6. --- Create and Save Segmentation Mask & Colored Overlay ---
            try:
                seg_results = segment_leaf_hsv(img_np.copy(), PREDICTION_FOLDER)
                segmentation_path = seg_results.get("overlay")
                # optionally keep mask path in record if useful
                seg_mask_path = seg_results.get("mask")
            except Exception as seg_e:
                print(f"Error generating segmentation: {seg_e}")
                segmentation_path = None

        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            pass  # Continue without a heatmap

    # 6. --- Log to Database ---
    record = {
        "filename": file.filename,
        "prediction": label,
        "confidence": confidence,
        "confidence_percent": confidence_percent,
        "probabilities": probabilities_dict,
        "heatmap": heatmap_path,  # Save the path
        "segmentation": locals().get("segmentation_path", None),
        "legend": [{"color": "#FF0000", "label": "Diseased region"}],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    # Link to authenticated user if token present
    user_id = get_current_user_id()
    if user_id:
        record["user_id"] = user_id

    try:
        history_collection.insert_one(record)
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")

    # 7. --- Return JSON that matches your React code ---
    return jsonify(
        {
            "prediction": label,
            "confidence": confidence,
            "confidence_percent": confidence_percent,
            "probabilities": record["probabilities"],
            "message": f"🌿 The uploaded leaf is predicted as: {label} (Confidence: {confidence_percent})",
            "heatmap": heatmap_path,  # This is the field your React code needs
            "segmentation": record.get("segmentation"),
            "legend": record.get("legend"),
        }
    )


# -------------------------------------------------\n# --- This route serves the generated images ---\n# -------------------------------------------------
@app.route("/static/predictions/<filename>")
def send_prediction_image(filename):
    return send_from_directory(PREDICTION_FOLDER, filename)


# -------------------------------------------------\n# --- Serve generated stage images ---\n# -------------------------------------------------
@app.route("/static/generated_stages/<filename>")
def send_generated_image(filename):
    return send_from_directory(GENERATED_FOLDER, filename)


# -------------------------------------------------\n# --- Generate next-stage diseased images endpoint ---\n# -------------------------------------------------
_img2img_pipe = None


@app.route("/generate_stages", methods=["POST"])
def generate_stages():
    # On Render production, diffusers is not available
    # Return a graceful error message
    if _is_production and torch is None:
        return jsonify({
            "error": "Image generation is not available on this deployment (requires GPU and diffusers). "
                     "Please use a local version or contact support for ML features."
        }), 503
    
    global _diffusers_available, _diffusers_err
    if not _diffusers_available:
        # Retry lazy import here to catch newly installed packages after server start
        try:
            from diffusers import (
                StableDiffusionImg2ImgPipeline as _RetryPipe,  # noqa: F401
            )

            _diffusers_available = True
            _diffusers_err = None
        except Exception as e:
            _diffusers_err = str(e)
            return jsonify(
                {
                    "error": "Diffusers is not installed/available in this Python environment.",
                    "detail": _diffusers_err,
                }
            ), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    disease = request.form.get("disease", "leaf disease")

    try:
        init_image = Image.open(request.files["image"].stream).convert("RGB")
        init_image = init_image.resize((512, 512))
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {e}"}), 400

    global _img2img_pipe
    if _img2img_pipe is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        _img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=dtype
        ).to(device_str)

    prompts = [
        f"a realistic leaf showing the next stage of {disease}, with darker necrotic patches, blackening, and partial dryness",
        f"a realistic leaf showing the advanced stage of {disease}, with large blackened areas, holes, dryness, and curling edges",
    ]

    current_image = init_image
    saved_paths = []
    try:
        for i, prompt in enumerate(prompts, start=1):
            print(f"[DEBUG] Generating stage {i} with prompt: {prompt}")
            result = _img2img_pipe(
                prompt=prompt, image=current_image, strength=0.2, guidance_scale=8.5
            )
            next_image = result.images[0]
            filename = f"{uuid.uuid4()}_stage_{i}.jpg"
            save_path = os.path.join(GENERATED_FOLDER, filename)
            next_image.save(save_path)
            print(f"[DEBUG] Saved stage {i} image to {save_path}")
            saved_paths.append(
                os.path.join("static", "generated_stages", filename).replace("\\", "/")
            )
            current_image = next_image
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500

    return jsonify(
        {"stages": saved_paths, "count": len(saved_paths), "disease": disease}
    )


# -------------------------------------------------\n# --- Environment diagnostics (to debug venv issues) ---\n# -------------------------------------------------
@app.route("/env_info", methods=["GET"])
def env_info():
    info = {
        "python_executable": sys.executable,
        "cuda_available": torch.cuda.is_available(),
        "diffusers_available": _diffusers_available,
        "diffusers_error": _diffusers_err,
    }
    try:
        import diffusers  # noqa: F401
        from diffusers import __version__ as diffusers_version  # noqa: F401

        info.update(
            {
                "diffusers_import_runtime": True,
                "diffusers_version": diffusers_version,
            }
        )
    except Exception as e:
        info.update(
            {
                "diffusers_import_runtime": False,
                "diffusers_runtime_error": str(e),
            }
        )
    return jsonify(info)


# -------------------------------------------------\n# --- Timeline upload & trend endpoints ---\n# -------------------------------------------------
@app.route("/timeline/upload", methods=["POST"])
def timeline_upload():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    farmer_id = request.form.get("farmer_id", "default")
    # support tracking multiple plants per farmer
    plant_id = request.form.get("plant_id", "plant1")
    # crop and farmer-provided disease label
    crop = request.form.get("crop", "unknown")
    disease_label = request.form.get("disease", "Unknown")
    # Decide canonical disease value: prefer farmer-provided non-empty label, otherwise use model prediction

    file = request.files["image"]
    try:
        image = Image.open(file.stream).convert("RGB")
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_np_float = img_np.astype(float) / 255.0
        img_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {e}"}), 400

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence_tensor, predicted_index_tensor = torch.max(probs, 1)
        label = class_names[predicted_index_tensor.item()]
        confidence = confidence_tensor.item()
        predicted_index = predicted_index_tensor.item()

    heatmap_path = None
    segmentation_path = None
    severity_percent = 0.0

    if "healthy" not in label.lower():
        try:
            n_samples, stdev, colormap = _parse_smooth_params(request.form)
            heatmap, _ = smooth_grad_cam(
                model,
                img_tensor,
                class_idx=predicted_index,
                n_samples=n_samples,
                stdev_spread=stdev,
            )

            # resize to original
            heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

            # Percentile clipping and normalize
            p_low, p_high = np.percentile(heatmap, [5, 99])
            if p_high - p_low > 1e-6:
                heatmap = np.clip((heatmap - p_low) / (p_high - p_low), 0.0, 1.0)
            else:
                heatmap = np.clip(heatmap, 0.0, 1.0)

            try:
                heatmap = (
                    cv2.GaussianBlur((heatmap * 255).astype(np.uint8), (7, 7), 0) / 255.0
                )
            except Exception:
                heatmap = np.uint8(255 * heatmap)

            heatmap_uint8 = np.uint8(255 * heatmap)
            severity_percent = compute_severity_percent(heatmap_uint8)
            # Debug logging for heatmap statistics to diagnose zero-severity cases
            try:
                logger.info(
                    "Heatmap stats - shape=%s min=%s max=%s mean=%s",
                    heatmap_uint8.shape,
                    int(np.min(heatmap_uint8)),
                    int(np.max(heatmap_uint8)),
                    float(np.mean(heatmap_uint8)),
                )
            except Exception:
                pass

            cmap = _COLORMAP_MAP.get(colormap, cv2.COLORMAP_INFERNO)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cmap)
            alpha_overlay = 0.6
            overlay = (alpha_overlay * heatmap_color.astype(float) / 255.0) + (
                (1 - alpha_overlay) * img_np_float
            )
            overlay = np.clip(overlay, 0, 1)
            overlay = np.uint8(overlay * 255)

            # Save the blended overlay once. Ensure it's a proper uint8 image
            # (overlay is already in 0..1 float range, converted above).
            if overlay.dtype != np.uint8:
                overlay_to_save = np.uint8(np.clip(overlay, 0, 1) * 255)
            else:
                overlay_to_save = overlay
            filename = f"{uuid.uuid4()}.jpg"
            save_path = os.path.join(PREDICTION_FOLDER, filename)
            cv2.imwrite(save_path, overlay_to_save)
            heatmap_path = os.path.join("static", "predictions", filename).replace(
                "\\", "/"
            )

            # Segmentation: use HSV-based leaf segmentation helper and prefer color/dark masks when they have signal
            try:
                seg_results = segment_leaf_hsv(img_np.copy(), PREDICTION_FOLDER)
                segmentation_path = seg_results.get("mask")
                segmentation_overlay = seg_results.get("overlay")
                seg_mask_path = seg_results.get("mask")
                seg_metrics = seg_results.get("metrics", {})

                # read leaf mask if saved
                leaf_mask = None
                try:
                    if seg_mask_path:
                        abs_mask = os.path.join(os.path.dirname(__file__), seg_mask_path.replace("static/", "static/"))
                        leaf_mask = cv2.imread(abs_mask, cv2.IMREAD_GRAYSCALE)
                        if leaf_mask is None:
                            leaf_mask = None
                except Exception:
                    leaf_mask = None

                # Prepare HSV for color-based disease detection (brown/yellow ranges)
                hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
                lower_disease = np.array([10, 50, 20])
                upper_disease = np.array([35, 255, 255])
                mask_disease_color = cv2.inRange(hsv, lower_disease, upper_disease)
                if leaf_mask is not None and leaf_mask.shape == mask_disease_color.shape:
                    mask_disease_color = cv2.bitwise_and(mask_disease_color, (leaf_mask > 0).astype(np.uint8) * 255)
                # small clean
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_disease_color = cv2.morphologyEx(mask_disease_color, cv2.MORPH_OPEN, k, iterations=1)
                color_diseased_pixels = int((mask_disease_color > 0).sum())

                # Dark/grayscale fallback (necrotic spots)
                dark_mask, dark_pixels = detect_dark_lesions(img_np, leaf_mask)

                # Grad-CAM derived diseased mask (as previous fallback)
                diseased_mask_heatmap = None
                heatmap_diseased_pixels = 0
                try:
                    if 'heatmap_uint8' in locals():
                        diseased_mask_heatmap, heatmap_diseased_pixels = compute_diseased_mask(heatmap_uint8)
                except Exception:
                    diseased_mask_heatmap = None

                # Choose canonical diseased mask in priority: color HSV -> dark lesions -> grad-cam heatmap
                diseased_mask = None
                diseased_pixels = 0
                chosen_source = None

                if color_diseased_pixels > 0:
                    diseased_mask = mask_disease_color
                    diseased_pixels = color_diseased_pixels
                    chosen_source = "color_hsv"
                    logger.info("Using color HSV disease mask (pixels=%s)", diseased_pixels)
                elif dark_pixels > 0:
                    diseased_mask = dark_mask
                    diseased_pixels = dark_pixels
                    chosen_source = "dark_grayscale"
                    logger.info("Using dark/grayscale disease mask (pixels=%s)", diseased_pixels)
                elif heatmap_diseased_pixels > 0:
                    diseased_mask = diseased_mask_heatmap
                    diseased_pixels = heatmap_diseased_pixels
                    chosen_source = "gradcam_heatmap"
                    logger.info("Using grad-cam disease mask (pixels=%s)", diseased_pixels)
                else:
                    # nothing found; keep zero and emit debug later
                    diseased_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
                    diseased_pixels = 0
                    chosen_source = "none"

                # If we have a leaf mask, ensure shapes align and compute intersection if needed
                leaf_pixels = int(seg_metrics.get("leaf_pixels", 0))
                if leaf_mask is not None and diseased_mask is not None and leaf_mask.shape != diseased_mask.shape:
                    diseased_mask = cv2.resize(diseased_mask, (leaf_mask.shape[1], leaf_mask.shape[0]))

                if leaf_mask is not None and diseased_mask is not None:
                    inter = (diseased_mask > 0) & (leaf_mask > 0)
                    diseased_pixels = int(np.count_nonzero(inter)) if leaf_pixels > 0 else int((diseased_mask > 0).sum())

                # gentle dilation fallback if intersection is zero but mask had content
                try:
                    if diseased_pixels == 0 and (('diseased_mask_heatmap' in locals() and int((diseased_mask_heatmap > 0).sum()) > 0) or int((mask_disease_color > 0).sum()) > 0 or int((dark_mask > 0).sum()) > 0):
                        kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        dil = cv2.dilate(diseased_mask, kernel_d, iterations=1)
                        if leaf_mask is not None:
                            if dil.shape != leaf_mask.shape:
                                dil = cv2.resize(dil, (leaf_mask.shape[1], leaf_mask.shape[0]))
                            nd = int(np.count_nonzero((dil > 0) & (leaf_mask > 0)))
                        else:
                            nd = int((dil > 0).sum())
                        if nd > 0:
                            diseased_pixels = nd
                            diseased_mask = dil
                            logger.info("Dilation fallback used, diseased_pixels=%s", diseased_pixels)
                except Exception:
                    pass

                # Prefer diseased_percent_of_leaf when leaf_pixels available
                diseased_percent_of_leaf = (diseased_pixels / leaf_pixels) * 100.0 if leaf_pixels > 0 else 0.0
                health_metrics = {
                    "leaf_pixels": leaf_pixels,
                    "leaf_area_percent": seg_metrics.get("leaf_area_percent", 0.0),
                    "mean_saturation": seg_metrics.get("mean_saturation", 0.0),
                    "mean_green": seg_metrics.get("mean_green", 0.0),
                    "diseased_pixels": diseased_pixels,
                    "diseased_percent_of_leaf": diseased_percent_of_leaf,
                    "percent_of_image": (diseased_pixels / (img_np.shape[0] * img_np.shape[1]) * 100.0) if img_np.size > 0 else 0.0,
                    "disease_mask_source": chosen_source,
                }

                if leaf_pixels > 0:
                    severity_percent = diseased_percent_of_leaf
                else:
                    # fallback to heatmap severity if no leaf found
                    if 'heatmap_uint8' in locals():
                        severity_percent = compute_severity_percent(heatmap_uint8)
                    else:
                        severity_percent = 0.0

                # store segmentation overlay if present
                try:
                    locals()["segmentation_overlay"] = segmentation_overlay
                except Exception:
                    pass

                # write debug artifacts when no diseased pixels were found (for tracing)
                try:
                    if severity_percent == 0.0 or int(health_metrics.get("diseased_pixels", 0)) == 0:
                        dbg_id = str(uuid.uuid4())
                        if 'heatmap_uint8' in locals():
                            try:
                                dbg_heat_path = os.path.join(PREDICTION_FOLDER, f"debug_heat_{dbg_id}.png")
                                cv2.imwrite(dbg_heat_path, heatmap_uint8)
                                locals().setdefault('debug_paths', {})['heatmap'] = os.path.join('static','predictions', os.path.basename(dbg_heat_path)).replace('\\','/')
                            except Exception:
                                pass
                        try:
                            if diseased_mask is not None:
                                dbg_mask_path = os.path.join(PREDICTION_FOLDER, f"debug_mask_{dbg_id}.png")
                                cv2.imwrite(dbg_mask_path, diseased_mask)
                                locals().setdefault('debug_paths', {})['diseased_mask'] = os.path.join('static','predictions', os.path.basename(dbg_mask_path)).replace('\\','/')
                        except Exception:
                            pass
                        try:
                            if leaf_mask is not None:
                                dbg_leaf_path = os.path.join(PREDICTION_FOLDER, f"debug_leaf_{dbg_id}.png")
                                cv2.imwrite(dbg_leaf_path, leaf_mask)
                                locals().setdefault('debug_paths', {})['leaf_mask'] = os.path.join('static','predictions', os.path.basename(dbg_leaf_path)).replace('\\','/')
                        except Exception:
                            pass
                        try:
                            dbg_ctx_path = os.path.join(PREDICTION_FOLDER, f"debug_ctx_{dbg_id}.jpg")
                            ctx_vis = img_np.copy()
                            if 'heatmap_uint8' in locals():
                                cmap = _COLORMAP_MAP.get(colormap, cv2.COLORMAP_INFERNO)
                                hcolor = cv2.applyColorMap(heatmap_uint8, cmap)
                                alpha = 0.5
                                ctx_vis = cv2.addWeighted(hcolor, alpha, ctx_vis, 1 - alpha, 0)
                            cv2.imwrite(dbg_ctx_path, ctx_vis)
                            locals().setdefault('debug_paths', {})['context_overlay'] = os.path.join('static','predictions', os.path.basename(dbg_ctx_path)).replace('\\','/')
                        except Exception:
                            pass
                        logger.info("Wrote debug artifacts for zero-severity case: %s", locals().get('debug_paths', {}))
                except Exception:
                    pass
            except Exception as _e:
                print(f"Timeline segmentation error: {_e}")
        except Exception as e:
            print(f"Timeline Grad-CAM error: {e}")

    # Persist images in MongoDB GridFS under this farmer_id
    original_image_id = None
    heatmap_image_id = None
    segmentation_image_id = None
    try:
        # Original image bytes
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)
        original_image_id = fs.put(
            buf.getvalue(),
            filename=file.filename or f"{uuid.uuid4()}.jpg",
            contentType="image/jpeg",
            metadata={"farmer_id": farmer_id, "kind": "original"},
        )
        # Heatmap and segmentation if available (read from saved paths)
        if heatmap_path:
            abs_heatmap = os.path.join(
                os.path.dirname(__file__), heatmap_path.replace("static/", "static/")
            )
            try:
                with open(abs_heatmap, "rb") as fh:
                    heatmap_image_id = fs.put(
                        fh.read(),
                        filename=os.path.basename(abs_heatmap),
                        contentType="image/jpeg",
                        metadata={"farmer_id": farmer_id, "kind": "heatmap"},
                    )
            except Exception as _e:
                print(f"GridFS heatmap store error: {_e}")
        if segmentation_path:
            abs_seg = os.path.join(
                os.path.dirname(__file__),
                segmentation_path.replace("static/", "static/"),
            )
            try:
                with open(abs_seg, "rb") as fh:
                    segmentation_image_id = fs.put(
                        fh.read(),
                        filename=os.path.basename(abs_seg),
                        contentType="image/jpeg",
                        metadata={"farmer_id": farmer_id, "kind": "segmentation"},
                    )
            except Exception as _e:
                print(f"GridFS segmentation store error: {_e}")
    except Exception as e:
        print(f"GridFS store error: {e}")

    # Determine week index as sequential count per farmer+plant+crop
    try:
        existing_count = timelines_collection.count_documents({"farmer_id": farmer_id, "plant_id": plant_id, "crop": crop})
    except Exception:
        existing_count = 0
    week_index = int(existing_count) + 1

    # Decide canonical disease to store: prefer farmer-provided label when present, otherwise use model prediction
    if disease_label and str(disease_label).strip().lower() != "unknown":
        disease_to_store = disease_label
        disease_source = "farmer"
    else:
        disease_to_store = label
        disease_source = "model"

    # Special-case sensitivity for Late Blight: these lesions can be dark and small early on.
    # Optionally amplify the reported severity slightly for visibility, but keep raw counts in health_metrics.
    try:
        if "late" in label.lower() or "blight" in label.lower() or "late_blight" in label.lower():
            try:
                orig_sp = float(severity_percent)
            except Exception:
                orig_sp = 0.0
            # store original in health_metrics for transparency
            try:
                locals().setdefault('health_metrics', {})['severity_percent_original'] = orig_sp
            except Exception:
                pass
            # If severity is small (<10%), amplify modestly so it's more visible in the timeline
            if orig_sp > 0 and orig_sp < 10.0:
                factor = 3.0
                new_sp = min(100.0, orig_sp * factor)
                severity_percent = new_sp
                try:
                    locals().setdefault('health_metrics', {})['severity_adjustment'] = {
                        'factor': factor,
                        'reason': 'late_blight_visibility_amplification'
                    }
                except Exception:
                    pass
                logger.info("Adjusted severity for late blight from %s to %s (factor=%s)", orig_sp, severity_percent, factor)
    except Exception:
        pass

    # Ensure stored severity values are expressed as percentages (0..100).
    try:
        # If a fraction was produced (0..1), scale to 0..100
        if severity_percent is not None:
            try:
                sp = float(severity_percent)
                if sp <= 1.0:
                    logger.info("Normalizing severity_percent from fraction %s to percent", sp)
                    severity_percent = sp * 100.0
            except Exception:
                pass
        # Ensure health_metrics percent fields are also 0..100
        if 'health_metrics' in locals() and isinstance(health_metrics, dict):
            try:
                dp = float(health_metrics.get('diseased_percent_of_leaf', 0.0))
                if dp <= 1.0:
                    health_metrics['diseased_percent_of_leaf'] = dp * 100.0
            except Exception:
                pass

    except Exception:
        pass

    entry = {
        "farmer_id": farmer_id,
        "plant_id": plant_id,
        "crop": crop,
        "disease": disease_to_store,
        "disease_source": disease_source,
        "filename": file.filename,
        "prediction": label,
        "confidence": confidence,
        "severity_percent": severity_percent,
        "heatmap": heatmap_path,
        # segmentation is the original mask; keep the visual overlay separately
        "segmentation": segmentation_path,
        "segmentation_overlay": locals().get("segmentation_overlay"),
        "health_metrics": locals().get("health_metrics", {}),
        "week_index": week_index,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_image_id": str(original_image_id) if original_image_id else None,
        "heatmap_image_id": str(heatmap_image_id) if heatmap_image_id else None,
        "segmentation_image_id": str(segmentation_image_id)
        if segmentation_image_id
        else None,
    }
    # Attach user_id if authenticated
    user_id = get_current_user_id()
    if user_id:
        entry["user_id"] = user_id
    # Determine previous entry for same farmer+plant so we can compare
    comparison = None
    try:
        prev = timelines_collection.find_one({"farmer_id": farmer_id, "plant_id": plant_id, "crop": crop}, sort=[("week_index", -1)])
        if prev and int(prev.get("week_index", 0)) > 0:
            prev_sev = float(prev.get("severity_percent", 0.0))
            sev_delta = severity_percent - prev_sev
            if sev_delta > 1.0:
                sev_trend = "worsening"
            elif sev_delta < -1.0:
                sev_trend = "improving"
            else:
                sev_trend = "stable"

            prev_metrics = prev.get("health_metrics", {}) or {}
            prev_diseased = float(prev_metrics.get("diseased_percent_of_leaf", 0.0))
            curr_diseased = float(locals().get("health_metrics", {}).get("diseased_percent_of_leaf", 0.0))
            diseased_delta = curr_diseased - prev_diseased
            if diseased_delta > 0.5:
                diseased_trend = "worsening"
            elif diseased_delta < -0.5:
                diseased_trend = "improving"
            else:
                diseased_trend = "stable"

            prev_green = float(prev_metrics.get("leaf_area_percent", 0.0))
            curr_green = float(locals().get("health_metrics", {}).get("leaf_area_percent", 0.0))
            green_delta = curr_green - prev_green
            if green_delta > 1.0:
                green_trend = "improving"
            elif green_delta < -1.0:
                green_trend = "worsening"
            else:
                green_trend = "stable"

            comparison = {
                "prev_week_index": int(prev.get("week_index", 0)),
                "severity_delta": sev_delta,
                "severity_trend": sev_trend,
                "diseased_percent_delta": diseased_delta,
                "diseased_percent_trend": diseased_trend,
                "leaf_area_percent_delta": green_delta,
                "leaf_area_percent_trend": green_trend,
            }
    except Exception:
        comparison = None

    # Insert entry and then compute trend across all saved severities
    try:
        # normalize before saving to ensure consistent percentage scales
        try:
            _normalize_entry_percentages(entry)
        except Exception:
            pass
        res = timelines_collection.insert_one(entry)
        entry["_id"] = str(res.inserted_id)
    except Exception as e:
        print(f"Error saving timeline entry: {e}")

    try:
        # trend for this farmer+plant+crop
        records = list(
            timelines_collection.find(
                {"farmer_id": farmer_id, "plant_id": plant_id, "crop": crop},
                {"_id": 0, "severity_percent": 1, "timestamp": 1},
            )
        )
        records.sort(key=lambda r: r.get("timestamp", ""))
        severities = [float(r.get("severity_percent", 0.0)) for r in records]
        trend = trend_from_series(severities)
    except Exception:
        trend = "unknown"

    # Attach comparison into returned entry
    if comparison is not None:
        entry["comparison"] = comparison

    # Persist comparison field into DB if possible
    try:
        if comparison is not None and entry.get("_id"):
            timelines_collection.update_one({"_id": ObjectId(entry["_id"])}, {"$set": {"comparison": comparison}})
    except Exception:
        pass

    return jsonify({"saved": True, "trend": trend, "entry": entry})


@app.route("/timeline", methods=["GET"])
def get_timeline():
    farmer_id = request.args.get("farmer_id", "default")
    plant_id = request.args.get("plant_id", None)
    crop = request.args.get("crop", None)
    try:
        uid = get_current_user_id()
        query = {"farmer_id": farmer_id}
        if uid:
            query["user_id"] = uid
        if plant_id:
            query["plant_id"] = plant_id
        if crop:
            query["crop"] = crop
        records = list(timelines_collection.find(query))
        # convert ObjectId to string for JSON transport
        for r in records:
            if r.get("_id") is not None:
                try:
                    r["_id"] = str(r["_id"])
                except Exception:
                    pass
            # normalize percentage fields for safety
            try:
                _normalize_entry_percentages(r)
            except Exception:
                pass
        records.sort(key=lambda r: r.get("timestamp", ""))
        severities = [float(r.get("severity_percent", 0.0)) for r in records]
        trend = trend_from_series(severities)
        return jsonify(
            {
                "farmer_id": farmer_id,
                "trend": trend,
                "count": len(records),
                "entries": records,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Could not fetch timeline: {e}"}), 500


@app.route("/timeline/predict", methods=["GET"])
def timeline_predict():
    farmer_id = request.args.get("farmer_id", "default")
    plant_id = request.args.get("plant_id", None)
    crop = request.args.get("crop", None)
    try:
        query = {"farmer_id": farmer_id}
        if plant_id:
            query["plant_id"] = plant_id
        if crop:
            query["crop"] = crop
        records = list(timelines_collection.find(query, {"severity_percent": 1, "week_index": 1}).sort([("week_index", 1)]))
        # normalize records if any stored as fractions
        for r in records:
            try:
                _normalize_entry_percentages(r)
            except Exception:
                pass
        severities = [float(r.get("severity_percent", 0.0)) for r in records]
        weeks = [int(r.get("week_index", i + 1)) for i, r in enumerate(records)]
        pred = predict_trend_from_series(severities)
        return jsonify({"input_weeks": weeks, "input_severities": severities, "prediction": pred})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/timeline/summary", methods=["GET"])
def timeline_summary():
    """Generate a PDF summary for the last N weeks for farmer+plant+crop."""
    farmer_id = request.args.get("farmer_id", "default")
    plant_id = request.args.get("plant_id", None)
    crop = request.args.get("crop", None)
    weeks = int(request.args.get("weeks", "4"))
    try:
        query = {"farmer_id": farmer_id}
        if plant_id:
            query["plant_id"] = plant_id
        if crop:
            query["crop"] = crop
        records = list(timelines_collection.find(query).sort([("week_index", 1)]))
        for r in records:
            try:
                _normalize_entry_percentages(r)
            except Exception:
                pass
        # take last `weeks` entries
        recent = records[-weeks:] if len(records) >= weeks else records
        severities = [float(r.get("severity_percent", 0.0)) for r in recent]
        labels = [f"Week {int(r.get('week_index', i+1))}" for i, r in enumerate(recent)]

        # Try to import matplotlib lazily
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            return jsonify({"error": f"matplotlib is required for PDF generation: {e}"}), 500

        fig, ax = plt.subplots(figsize=(8, 4))
        if severities:
            ax.plot(range(len(severities)), severities, marker="o", linestyle="-", color="tab:green")
            ax.set_ylim(0, max(100, max(severities) * 1.2))
            ax.set_xticks(range(len(severities)))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_ylabel("Severity (%)")
            ax.set_title(f"Severity trend for {farmer_id} {plant_id or ''} {crop or ''}")
        else:
            ax.text(0.5, 0.5, "No entries available", ha="center", va="center")

        # prediction
        pred = predict_trend_from_series(severities)
        pred_text = f"Predicted next-week severity: {pred.get('predicted'):.2f} (slope={pred.get('slope'):.3f})" if pred.get("predicted") is not None else "No prediction"
        # textual summary
        if len(severities) >= 2:
            delta = severities[-1] - severities[0]
            summary = "increasing" if delta > 0 else "decreasing" if delta < 0 else "stable"
            summary_text = f"Over {len(severities)} weeks the severity changed by {delta:.2f}%. Trend: {summary}."
        else:
            summary_text = "Insufficient data for trend summary."

        # add summary text below plot
        fig.text(0.01, 0.01, summary_text + "\n" + pred_text, fontsize=9)

        pdf_fname = f"summary_{uuid.uuid4()}.pdf"
        pdf_path = os.path.join(PREDICTION_FOLDER, pdf_fname)
        plt.tight_layout()
        fig.savefig(pdf_path)
        plt.close(fig)

        return jsonify({"pdf": os.path.join("static", "predictions", pdf_fname).replace('\\', '/')})
    except Exception as e:
        return jsonify({"error": f"Could not create summary: {e}"}), 500


# -----------------------\n# Fetch History API\n# -----------------------
@app.route("/history", methods=["GET"])
def get_history():
    # If authenticated, filter by user_id; else return empty list for privacy
    uid = get_current_user_id()
    query = {"user_id": uid} if uid else {"user_id": "__none__"}
    try:
        records = list(history_collection.find(query, {"_id": 0}))
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": f"Could not fetch history: {e}"}), 500

# -------------------------
# Serve frontend & static (React Router friendly)
# -------------------------
from flask import send_from_directory

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path='index.html'):
    """
    Serve files from backend/static/frontend.
    If a requested path exists as a static file, serve it.
    Otherwise return index.html so React Router can handle the route.
    """
    static_dir = os.path.join(os.path.dirname(__file__), "static", "frontend")
    # If a specific static file is requested and exists, serve it.
    requested = os.path.join(static_dir, path)
    if path != "" and os.path.exists(requested) and os.path.isfile(requested):
        # path is relative to static/frontend
        return send_from_directory(static_dir, path)
    # fallback to index.html
    index_path = os.path.join(static_dir, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(static_dir, 'index.html')
    # helpful error if index missing
    return jsonify({"error": "index.html not found. Did you build the frontend?"}), 404


# -----------------------\n# Catch-all route for SPA\n# -----------------------
# Serve index.html for all non-API routes so React Router can handle client-side routing
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    """Serve the frontend SPA, handling all non-API routes."""
    # Don't serve index.html for API routes, timeline, history, or static assets
    if path.startswith(('api/', 'timeline', 'history', 'assets/', 'auth/', 'predict', 'generate_stages')):
        # Let Flask's normal routing handle these
        return None
    
    # For all other routes (including /login, /home, etc.), serve index.html
    # so React Router can handle client-side navigation
    index_path = os.path.join(app.static_folder, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({"error": "index.html not found. Did you build the frontend?"}), 404


# -----------------------\n# Run Server\n# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

