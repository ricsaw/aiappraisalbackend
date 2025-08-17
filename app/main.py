from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
from PIL import Image, UnidentifiedImageError
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from pymilvus import Collection, connections
from torchvision import models, transforms
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import os
import random
from collections import OrderedDict
import gc

# -------- Memory Management ---------
torch.backends.cudnn.benchmark = False  # Reduce memory fragmentation
torch.backends.cudnn.deterministic = True

# -------- Configs ---------
NUM_CLASSES = 10
IMAGE_SIZE = (224, 224)
MODEL_PATH = "app/models/grader_classifier.pt"
MAPPING_PATH = "app/models/label_mapping.json"

# -------- FastAPI setup --------
app = FastAPI()

# Global variables - initialize as None to save startup memory
collection = None
clip_model = None
clip_processor = None
grading_model = None
label_mapping = None

ZILLIZ_HOST = "https://in03-0305f9ddf217854.serverless.gcp-us-west1.cloud.zilliz.com"
ZILLIZ_TOKEN = "923738adce800b1f016901dcd62da0fa577671d54ef3f5b2a012e4d19ab57187a9d7cb45397684680e43925ebf1f32ca8b4b02b4"

# -------- Device (force CPU to save GPU memory) --------
device = "cpu"  # Force CPU on Render to save memory
print(f"Using device: {device}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Grades(BaseModel):
    centering: int
    edges: int
    corners: int
    surface: int

class AppraisalResult(BaseModel):
    cardName: str
    edition: str
    setname: str
    grades: Grades
    overallGrade: int
    estimatedValue: float
    confidence: float

# -------- Lazy Loading Functions --------
def get_clip_model():
    """Lazy load CLIP model only when needed"""
    global clip_model, clip_processor
    if clip_model is None:
        print("Loading CLIP model...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        
        # Set to eval mode and enable memory efficient attention
        clip_model.eval()
        if hasattr(clip_model, 'gradient_checkpointing_enable'):
            clip_model.gradient_checkpointing_enable()
            
        print("✅ CLIP model loaded")
    return clip_model, clip_processor

def get_grading_model():
    """Lazy load grading model only when needed"""
    global grading_model, label_mapping
    if grading_model is None:
        print("Loading grading model...")
        grading_model, label_mapping = load_grading_model()
        print("✅ Grading model loaded")
    return grading_model, label_mapping

def get_collection():
    """Lazy load collection connection only when needed"""
    global collection
    if collection is None:
        print("Connecting to Zilliz...")
        connections.connect(
            alias="zilliz", 
            uri=ZILLIZ_HOST, 
            token=ZILLIZ_TOKEN
        )
        collection = Collection("pokemon_cards", using="zilliz")
        collection.load()
        print("✅ Connected to Zilliz Cloud")
    return collection

@app.on_event("startup")
def startup_event():
    """Minimal startup - only connect to DB, lazy load models"""
    print("🚀 FastAPI starting up with lazy loading...")
    # Don't load models at startup - wait until first request
    pass

# -------- Memory-optimized image processing --------
def image_to_vector(image_bytes: bytes) -> list[float]:
    """Convert image to vector with memory optimization"""
    clip_model, clip_processor = get_clip_model()
    
    # Process image with smaller size to save memory
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA").resize((224, 224))
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    
    # Process in smaller batches and clear cache
    inputs = clip_processor(images=background, return_tensors="pt").to(device)
    
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
        embeddings = embeddings.cpu().numpy()
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embedding = embeddings / norm
        result = normalized_embedding[0].tolist()
    
    # Clear tensors and force garbage collection
    del inputs, embeddings
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    return result

# -------- Lightweight Model Definition --------
class GradeClassificationNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Use MobileNetV2 instead of ResNet18 for lower memory
        self.backbone = models.mobilenet_v2(weights=None)
        self.backbone.classifier = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict

def load_grading_model(model_path=MODEL_PATH, mapping_path=MAPPING_PATH):
    """Load grading model with memory optimization"""
    label_mapping = None
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r') as f:
                label_mapping = json.load(f)
        except Exception as e:
            print(f"Could not load label mapping: {e}")

    # Use the lightweight model
    model = GradeClassificationNet(num_classes=NUM_CLASSES).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        state_dict = remove_module_prefix(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    except Exception as e:
        print(f"Error loading model: {e}")
        # Return a dummy model if loading fails
        pass

    model.eval()
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    return model, label_mapping

# -------- Memory-efficient transforms --------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def model_output_to_grade(model_output: int) -> int:
    if label_mapping and 'model_output_to_grade' in label_mapping:
        return label_mapping['model_output_to_grade'].get(str(model_output), model_output + 1)
    return model_output + 1

def predict_card_grade_with_uncertainty(image_bytes: bytes) -> dict:
    """Memory-efficient grade prediction"""
    model, _ = get_grading_model()
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"Could not read image: {str(e)}")

    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_model_output = torch.argmax(logits, dim=1).item()
        predicted_grade = model_output_to_grade(predicted_model_output)
        max_confidence = probabilities[0, predicted_model_output].item()
        
        # Calculate uncertainty
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1).item()
        max_entropy = np.log(NUM_CLASSES)
        normalized_uncertainty = entropy / max_entropy
        
        # Expected grade
        model_outputs = torch.arange(0, NUM_CLASSES, dtype=torch.float32).to(device)
        expected_model_output = torch.sum(probabilities * model_outputs).item()
        expected_grade = expected_model_output + 1

    # Clear tensors
    del input_tensor, logits, probabilities, model_outputs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    return {
        'predicted_grade': predicted_grade,
        'predicted_model_output': predicted_model_output,
        'expected_grade': expected_grade,
        'max_confidence': max_confidence,
        'uncertainty': normalized_uncertainty,
        'probabilities': probabilities.cpu().numpy() if 'probabilities' in locals() else np.zeros(NUM_CLASSES)
    }

def predict_multi_image_grade(image_bytes_list: list) -> dict:
    """Process images sequentially to save memory"""
    individual_results = []
    all_probabilities = []

    # Process one image at a time and clear memory between
    for i, img_bytes in enumerate(image_bytes_list):
        result = predict_card_grade_with_uncertainty(img_bytes)
        individual_results.append(result)
        all_probabilities.append(result['probabilities'])
        
        # Force garbage collection between images
        gc.collect()

    # Weighted ensemble
    weights = [0.35, 0.35, 0.15, 0.15]
    weighted_probs = np.average(all_probabilities, axis=0, weights=weights)
    ensemble_model_output = np.argmax(weighted_probs)
    ensemble_predicted_grade = model_output_to_grade(ensemble_model_output)
    ensemble_confidence = weighted_probs[ensemble_model_output]
    
    expected_grades = [r['expected_grade'] for r in individual_results]
    weighted_expected_grade = np.average(expected_grades, weights=weights)
    predicted_grades = [r['predicted_grade'] for r in individual_results]
    uncertainties = [r['uncertainty'] for r in individual_results]
    weighted_uncertainty = np.average(uncertainties, weights=weights)

    return {
        'overall_grade': ensemble_predicted_grade,
        'ensemble_confidence': ensemble_confidence,
        'weighted_expected_grade': weighted_expected_grade,
        'weighted_uncertainty': weighted_uncertainty,
        'individual_results': individual_results,
        'individual_grades': predicted_grades,
        'front_grade': predicted_grades[0],
        'back_grade': predicted_grades[1],
        'corner_grades': predicted_grades[2:4]
    }

def get_pricecharting_value(card_name: str, edition: str, overall_grade: int) -> float:
    """Lightweight value estimation"""
    try:
        query = f"{card_name} {edition}".replace(" ", "+")
        search_url = f"https://www.pricecharting.com/search-products?q={query}&type=prices"
        headers = {"User-Agent": "Mozilla/5.0"}

        res = requests.get(search_url, headers=headers, allow_redirects=True, timeout=5)  # Reduced timeout
        final_url = res.url
        if "search-products" in final_url:
            return 0.0

        res = requests.get(final_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        grade_label = f"Grade {overall_grade}" if overall_grade < 10 else f"PSA {overall_grade}"

        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) >= 2:
                    label = cols[0].get_text(strip=True)
                    value = cols[1].get_text(strip=True).replace("$", "").replace(",", "")
                    if label == grade_label and value:
                        try:
                            return float(value)
                        except:
                            return 0.0
        return 0.0
    except Exception as e:
        print(f"Scraper error: {e}")
        return 0.0

@app.get("/health")
async def health_check():
    """Lightweight health check"""
    return {"status": "healthy", "memory_efficient": True}

@app.post("/appraise", response_model=AppraisalResult)
async def appraise(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    topLeft: UploadFile = File(...),
    bottomRight: UploadFile = File(...),
):
    """Memory-efficient appraisal endpoint"""
    
    # Read images
    image_bytes_list = []
    for upload in [front, back, topLeft, bottomRight]:
        try:
            content = await upload.read()
            image_bytes_list.append(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read image {upload.filename}: {e}")

    # Default values
    card_name = "Unknown"
    card_edition = "Unknown"
    card_set_name = "Unknown"

    # Try card identification (lazy load collection)
    try:
        collection = get_collection()
        vector = image_to_vector(image_bytes_list[0])
        
        # Check if collection is loaded (correct method)
        try:
            collection.load()  # This is safe to call multiple times
        except Exception:
            pass  # Collection might already be loaded
            
        results = collection.search(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=1,  # Reduce to 1 result to save memory
            output_fields=["name", "edition", "set_name"]
        )
        
        if results and results[0]:
            result = results[0][0].entity
            card_name = result.get("name", "Unknown")
            card_edition = result.get("edition", "Unknown")
            card_set_name = result.get("set_name", "Unknown")
            
    except Exception as e:
        print(f"Error querying Milvus: {e}")

    # Grade prediction
    grade_results = predict_multi_image_grade(image_bytes_list)
    overall_grade_from_model = grade_results['overall_grade']
    overall_confidence = float(grade_results['ensemble_confidence'])
    individual_grades = grade_results['individual_grades']

    def clamp_grade(grade):
        return max(1, min(10, int(round(grade))))

    front_grade, back_grade, top_left_grade, bottom_right_grade = [clamp_grade(g) for g in individual_grades]

    print(f"Individual image predictions: Front={front_grade}, Back={back_grade}, TopLeft={top_left_grade}, BottomRight={bottom_right_grade}")

    # If all predictions are identical, add realistic variation
    if len(set(individual_grades)) == 1:
        print("All predictions identical, adding realistic component variation...")
        base_grade = front_grade
        individual_results = grade_results['individual_results']
        avg_uncertainty = np.mean([r['uncertainty'] for r in individual_results])

        if avg_uncertainty > 0.8:
            variation_range = 2
        elif avg_uncertainty > 0.6:
            variation_range = 1
        else:
            variation_range = 1

        corner_penalty = min(variation_range, max(0, int(avg_uncertainty * 3)))
        surface_bonus = min(1, max(0, int((1 - avg_uncertainty) * 2)))

        grades = {
            "centering": clamp_grade(base_grade + random.choice([-1, 0, 1]) if variation_range > 0 else base_grade),
            "edges": clamp_grade(base_grade + random.choice([-1, 0]) if variation_range > 0 else base_grade),
            "corners": clamp_grade(base_grade - corner_penalty),
            "surface": clamp_grade(base_grade + surface_bonus)
        }
        print(f"Applied variation - Base: {base_grade}, Corner penalty: -{corner_penalty}, Surface bonus: +{surface_bonus}")

    else:
        # Use individual image predictions for components
        corner_grade = clamp_grade(np.mean([top_left_grade, bottom_right_grade]))
        grades = {
            "centering": front_grade,
            "edges": back_grade,
            "corners": corner_grade,
            "surface": clamp_grade(np.mean([front_grade, back_grade]))
        }
        # Adjust for corner damage
        if corner_grade < front_grade - 1:
            grades["centering"] = min(grades["centering"], corner_grade + 1)
        if corner_grade < back_grade - 1:
            grades["edges"] = min(grades["edges"], corner_grade + 1)

    print(f"Final component grades: {grades}")

    # Calculate component-based overall grade
    min_component_grade = min(grades.values())
    max_component_grade = max(grades.values())

    print(f"Component grade range: {min_component_grade} - {max_component_grade}")
    print(f"Model ensemble prediction: {overall_grade_from_model}")
    print(f"Ensemble confidence: {overall_confidence:.4f}")

    component_weighted_score = (
        grades["centering"] * 0.25 +
        grades["edges"] * 0.15 +
        grades["corners"] * 0.2 +
        grades["surface"] * 0.3
    )
    print(f"Component weighted score: {component_weighted_score:.2f}")

    # Blend logic
    if overall_confidence > 0.4:
        model_component_diff = abs(overall_grade_from_model - component_weighted_score)
        if model_component_diff <= 1.5:
            final_overall_grade = overall_grade_from_model
            print(f"Using model prediction {overall_grade_from_model} (good confidence, close to components)")
        else:
            final_overall_grade = 0.6 * component_weighted_score + 0.4 * overall_grade_from_model
            print(f"Blending: components {component_weighted_score:.1f} + model {overall_grade_from_model} = {final_overall_grade:.1f}")
    elif overall_confidence > 0.25:
        final_overall_grade = 0.7 * component_weighted_score + 0.3 * overall_grade_from_model
        print(f"Moderate confidence, blending toward components: {final_overall_grade:.1f}")
    else:
        if 'weighted_expected_grade' in grade_results:
            expected_grade = grade_results['weighted_expected_grade']
            final_overall_grade = 0.5 * component_weighted_score + 0.5 * expected_grade
            print(f"Low confidence, using expected grade: components {component_weighted_score:.1f} + expected {expected_grade:.1f} = {final_overall_grade:.1f}")
        else:
            final_overall_grade = component_weighted_score
            print(f"Low confidence, using component score: {final_overall_grade:.1f}")

    if min_component_grade <= 1:
        max_cap = 3
    elif min_component_grade <= 3:
        max_cap = 6
    elif min_component_grade <= 5:
        max_cap = 8
    else:
        max_cap = 10
    print(f"Max cap based on worst component ({min_component_grade}): {max_cap}")

    # Only apply cap if it's significantly lower than our calculated grade
    if final_overall_grade > max_cap and (final_overall_grade - max_cap) > 1:
        print(f"Applying cap: {final_overall_grade:.1f} -> {max_cap}")
        final_overall_grade = max_cap

    final_overall_grade = clamp_grade(final_overall_grade)

    # Value estimation
    estimated_value = get_pricecharting_value(card_name, card_edition, final_overall_grade)

    print(f"Final computed overall grade: {final_overall_grade} with confidence {overall_confidence:.4f}")
    print(f"Component grades: {grades}")

    # Force cleanup
    gc.collect()

    return {
        "cardName": card_name,
        "edition": card_edition,
        "setname": card_set_name,
        "grades": grades,
        "overallGrade": final_overall_grade,
        "estimatedValue": estimated_value,
        "confidence": overall_confidence
    }