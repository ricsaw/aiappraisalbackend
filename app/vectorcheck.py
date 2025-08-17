from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from PIL import Image
import io
import os
import weaviate
from weaviate.classes.init import Auth

# Keep your OCR imports and functions as-is
from app.ocr_utils import extract_card_edition, extract_pokemon_name, enhance

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for prod
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
    edition: str  # New field
    setname: str  # New field
    grades: Grades
    overallGrade: int
    estimatedValue: float

# Initialize Weaviate client once (optional, or inside endpoint)
weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

@app.post("/appraise", response_model=AppraisalResult)
async def appraise(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    topLeft: UploadFile = File(...),
    bottomRight: UploadFile = File(...),
):
    # Read image bytes for Weaviate query
    front_bytes = await front.read()
    
    # Query Weaviate for closest card vector match using image bytes
    near_image = {"image": front_bytes}
    
    try:
        result = (
            client.query
            .get("PokemonCard", ["name", "set", "edition"])
            .with_near_image(near_image)
            .with_limit(1)
            .do()
        )
        card = result["data"]["Get"]["PokemonCard"][0]
        card_name = card.get("name", "Unknown")
        card_edition = card.get("edition", "Unknown")
        card_set_name = card.get("set", "Unknown")
    except Exception as e:
        print(f"Error querying Weaviate: {e}")
        # Fallback: use OCR if weaviate fails
        # front_img = Image.open(io.BytesIO(front_bytes))
        # enhanced_front = enhance(front_img)
        # card_name = extract_pokemon_name(enhanced_front)
        
        # bottom_right_img = Image.open(io.BytesIO(await bottomRight.read()))
        # enhanced_bottom_right = enhance(bottom_right_img)
        # card_edition = extract_card_edition(enhanced_bottom_right)


    
    # Calculate grades as before
    grades = {
        "centering": 9,
        "edges": 8,
        "corners": 8,
        "surface": 7,
    }
    overall = sum(grades.values()) // len(grades)
    estimated_value = overall * 13.5

    return {
        "cardName": card_name,
        "edition": card_edition,
        "setname": card_set_name,
        "grades": grades,
        "overallGrade": overall,
        "estimatedValue": estimated_value,
    }
