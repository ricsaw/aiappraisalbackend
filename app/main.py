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
    allow_origins=[
        "https://card-grader-tau.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
        "https://localhost:3000",
        "*"  # Allow all origins as fallback
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
    ],
    expose_headers=["*"],
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
    """Improved model loading with device consistency"""
    global grading_model, label_mapping
    if grading_model is None:
        print("Loading grading model...")
        grading_model, label_mapping = load_grading_model()
        
        # Ensure model is in eval mode and on correct device
        grading_model.eval()
        grading_model = grading_model.to(device)
        
        # Disable dropout and batch normalization updates
        for module in grading_model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
        
        print("✅ Grading model loaded and configured")
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
    """Improved grade prediction with better memory management"""
    model, _ = get_grading_model()
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"Could not read image: {str(e)}")

    # Ensure consistent preprocessing
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Ensure model is in eval mode
        model.eval()
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_model_output = torch.argmax(logits, dim=1).item()
        predicted_grade = model_output_to_grade(predicted_model_output)
        max_confidence = probabilities[0, predicted_model_output].item()
        
        # Improved uncertainty calculation
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1).item()
        max_entropy = np.log(NUM_CLASSES)
        normalized_uncertainty = entropy / max_entropy
        
        # Store probabilities before clearing tensors
        prob_numpy = probabilities.cpu().numpy()[0]
        
        # Expected grade calculation
        model_outputs = torch.arange(0, NUM_CLASSES, dtype=torch.float32).to(device)
        expected_model_output = torch.sum(probabilities * model_outputs).item()
        expected_grade = expected_model_output + 1

    # Clear tensors AFTER extracting all needed values
    del input_tensor, logits, probabilities, model_outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'predicted_grade': predicted_grade,
        'predicted_model_output': predicted_model_output,
        'expected_grade': expected_grade,
        'max_confidence': max_confidence,
        'uncertainty': normalized_uncertainty,
        'probabilities': prob_numpy
    }

def predict_multi_image_grade(image_bytes_list: list) -> dict:
    """Improved multi-image processing with better ensemble logic"""
    individual_results = []
    
    # Process images with consistent model state
    model, _ = get_grading_model()
    model.eval()  # Ensure consistent eval mode
    
    for i, img_bytes in enumerate(image_bytes_list):
        result = predict_card_grade_with_uncertainty(img_bytes)
        individual_results.append(result)
        print(f"Image {i}: Grade={result['predicted_grade']}, Confidence={result['max_confidence']:.3f}")

    # Extract probabilities and apply better weighting
    all_probabilities = np.array([r['probabilities'] for r in individual_results])
    
    # Adaptive weighting based on confidence
    confidences = np.array([r['max_confidence'] for r in individual_results])
    uncertainties = np.array([r['uncertainty'] for r in individual_results])
    
    # Weight by confidence (higher confidence = higher weight)
    confidence_weights = confidences / np.sum(confidences)
    
    # Alternative: Use predefined weights but adjust based on confidence
    base_weights = np.array([0.35, 0.35, 0.15, 0.15])  # front, back, topLeft, bottomRight
    
    # Blend base weights with confidence weights
    final_weights = 0.7 * base_weights + 0.3 * confidence_weights
    final_weights = final_weights / np.sum(final_weights)  # Normalize
    
    print(f"Confidence weights: {confidence_weights}")
    print(f"Final weights: {final_weights}")
    
    # Weighted ensemble
    weighted_probs = np.average(all_probabilities, axis=0, weights=final_weights)
    ensemble_model_output = np.argmax(weighted_probs)
    ensemble_predicted_grade = model_output_to_grade(ensemble_model_output)
    ensemble_confidence = weighted_probs[ensemble_model_output]
    
    # Improved expected grade calculation
    expected_grades = [r['expected_grade'] for r in individual_results]
    weighted_expected_grade = np.average(expected_grades, weights=final_weights)
    
    predicted_grades = [r['predicted_grade'] for r in individual_results]
    weighted_uncertainty = np.average(uncertainties, weights=final_weights)

    return {
        'overall_grade': ensemble_predicted_grade,
        'ensemble_confidence': ensemble_confidence,
        'weighted_expected_grade': weighted_expected_grade,
        'weighted_uncertainty': weighted_uncertainty,
        'individual_results': individual_results,
        'individual_grades': predicted_grades,
        'front_grade': predicted_grades[0],
        'back_grade': predicted_grades[1],
        'corner_grades': predicted_grades[2:4],
        'final_weights': final_weights.tolist()
    }

def calculate_component_grades_improved(grade_results: dict) -> tuple[dict, float]:
    """Improved component grade calculation"""
    individual_grades = grade_results['individual_grades']
    individual_results = grade_results['individual_results']
    
    front_grade = individual_grades[0]
    back_grade = individual_grades[1] 
    top_left_grade = individual_grades[2]
    bottom_right_grade = individual_grades[3]
    
    # Calculate average uncertainty for variation logic
    avg_uncertainty = np.mean([r['uncertainty'] for r in individual_results])
    print(f"Average uncertainty: {avg_uncertainty:.3f}")
    
    # Corner grade calculation
    corner_grade = int(round(np.mean([top_left_grade, bottom_right_grade])))
    
    # More sophisticated component mapping
    grades = {
        "centering": front_grade,  # Front image best shows centering
        "edges": back_grade,       # Back image for edge assessment
        "corners": corner_grade,   # Average of corner close-ups
        "surface": int(round(np.mean([front_grade, back_grade])))  # Surface from front/back
    }
    
    # Apply realistic constraints
    # Corners typically grade lower due to damage
    if corner_grade < min(front_grade, back_grade) - 2:
        # Severe corner damage affects other grades
        grades["centering"] = min(grades["centering"], corner_grade + 2)
        grades["edges"] = min(grades["edges"], corner_grade + 1)
        grades["surface"] = min(grades["surface"], corner_grade + 1)
    
    # Edge damage affects surface
    if grades["edges"] < grades["surface"] - 1:
        grades["surface"] = min(grades["surface"], grades["edges"] + 1)
    
    # Ensure all grades are within bounds
    for component in grades:
        grades[component] = max(1, min(10, grades[component]))
    
    return grades, avg_uncertainty

def calculate_final_overall_grade(grades: dict, grade_results: dict, avg_uncertainty: float) -> int:
    """Improved final grade calculation"""
    overall_grade_from_model = grade_results['overall_grade']
    overall_confidence = grade_results['ensemble_confidence']
    
    # Component-based grade calculation
    # Professional grading typically weights surface highest
    component_weights = {
        "centering": 0.15,  # Reduced from 0.25
        "edges": 0.20,      # Increased 
        "corners": 0.25,    # Increased (corners are critical)
        "surface": 0.40     # Increased (surface is most important)
    }
    
    component_weighted_score = sum(
        grades[component] * weight 
        for component, weight in component_weights.items()
    )
    
    print(f"Component weighted score: {component_weighted_score:.2f}")
    print(f"Model prediction: {overall_grade_from_model}")
    print(f"Model confidence: {overall_confidence:.3f}")
    
    # Improved blending logic
    model_component_diff = abs(overall_grade_from_model - component_weighted_score)
    
    if overall_confidence > 0.6 and model_component_diff <= 1.0:
        # High confidence and close agreement - use model
        final_grade = overall_grade_from_model
        print(f"Using model prediction (high confidence, close agreement)")
        
    elif overall_confidence > 0.4 and model_component_diff <= 2.0:
        # Moderate confidence - weighted blend favoring model
        blend_weight = 0.7  # Higher weight on model
        final_grade = blend_weight * overall_grade_from_model + (1 - blend_weight) * component_weighted_score
        print(f"Blending: {blend_weight} model + {1-blend_weight} components = {final_grade:.2f}")
        
    elif overall_confidence > 0.25:
        # Lower confidence - favor components
        blend_weight = 0.3  # Lower weight on model
        final_grade = blend_weight * overall_grade_from_model + (1 - blend_weight) * component_weighted_score
        print(f"Low confidence blending: {final_grade:.2f}")
        
    else:
        # Very low confidence - use components with expected grade as fallback
        if 'weighted_expected_grade' in grade_results:
            expected_grade = grade_results['weighted_expected_grade']
            final_grade = 0.6 * component_weighted_score + 0.4 * expected_grade
            print(f"Very low confidence: components + expected = {final_grade:.2f}")
        else:
            final_grade = component_weighted_score
            print(f"Using pure component score: {final_grade:.2f}")
    
    # Apply grade caps based on worst component (more conservative)
    min_component = min(grades.values())
    
    # Grade caps - more realistic
    if min_component == 1:
        max_cap = 2  # More restrictive
    elif min_component <= 2:
        max_cap = 4
    elif min_component <= 3:
        max_cap = 6
    elif min_component <= 5:
        max_cap = 8
    else:
        max_cap = 10
    
    print(f"Grade cap based on worst component ({min_component}): {max_cap}")
    
    # Apply cap if significantly higher
    if final_grade > max_cap and (final_grade - max_cap) > 0.5:
        print(f"Applying cap: {final_grade:.1f} -> {max_cap}")
        final_grade = max_cap
    
    return max(1, min(10, int(round(final_grade))))

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

@app.options("/appraise")
async def appraise_options():
    """Handle OPTIONS preflight request for CORS"""
    return {"message": "OK"}

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
    """Improved appraisal endpoint with fixed grading logic"""
    
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

    # Grade prediction with improved ensemble
    grade_results = predict_multi_image_grade(image_bytes_list)
    
    print(f"Individual image predictions: {grade_results['individual_grades']}")
    print(f"Ensemble prediction: {grade_results['overall_grade']} (confidence: {grade_results['ensemble_confidence']:.3f})")

    # Calculate component grades with improved logic
    grades, avg_uncertainty = calculate_component_grades_improved(grade_results)
    print(f"Component grades: {grades}")

    # Calculate final overall grade with improved blending
    final_overall_grade = calculate_final_overall_grade(grades, grade_results, avg_uncertainty)
    
    # Use ensemble confidence as final confidence
    overall_confidence = float(grade_results['ensemble_confidence'])

    # Value estimation
    estimated_value = get_pricecharting_value(card_name, card_edition, final_overall_grade)

    print(f"Final overall grade: {final_overall_grade} with confidence {overall_confidence:.4f}")
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