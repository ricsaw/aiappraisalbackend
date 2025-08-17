import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import re
from rapidfuzz import process, fuzz
import requests

def enhance(image):
    gray = image.convert("L")  # convert to grayscale
    contrast = ImageEnhance.Contrast(gray).enhance(2.0)  # increase contrast
    return contrast

def preprocess_image(image: Image.Image) -> Image.Image:
    # Convert to grayscale
    gray = image.convert("L")
    # Enhance contrast
    contrast = ImageEnhance.Contrast(gray).enhance(2.5)
    # Sharpen the image
    sharpened = contrast.filter(ImageFilter.SHARPEN)
    return sharpened


def fetch_pokemon_names(limit=2050):
    url = f"https://pokeapi.co/api/v2/pokemon?limit={limit}"
    response = requests.get(url)
    data = response.json()
    names = [pokemon['name'].capitalize() for pokemon in data['results']]
    return names

POKEMON_NAMES = fetch_pokemon_names()

def preprocess_image(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    gray = ImageOps.autocontrast(gray)
    # Upscale to improve OCR accuracy
    gray = gray.resize((gray.width * 3, gray.height * 3), Image.LANCZOS)
    enhancer = ImageEnhance.Sharpness(gray)
    gray = enhancer.enhance(3)  # stronger sharpening
    # Optional: add slight blur to reduce noise, but test to see effect
    # gray = gray.filter(ImageFilter.MedianFilter(size=3))
    return gray

def extract_pokemon_name(image: Image.Image) -> str:
    width, height = image.size
    crop_box = (0, 0, width, int(height * 0.25))  # focus on top quarter (usually name area)
    cropped = image.crop(crop_box)
    preprocessed = preprocess_image(cropped)

    data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT, config='--psm 6')

    candidates = []
    for i, word in enumerate(data['text']):
        w = word.strip()
        if len(w) < 3:
            continue
        h = data['height'][i]
        # Filter out known unwanted tokens that might appear (optional)
        if re.search(r'(stage|evolves|from|basic|level|hp|gx|v|x)', w, re.I):
            continue
        clean_word = re.sub(r'[^A-Za-z]', '', w)
        if clean_word:
            candidates.append((clean_word, h))

    if not candidates:
        return "Unknown"

    # Sort candidates by height descending (assuming bigger = name)
    candidates.sort(key=lambda x: x[1], reverse=True)

    for word, _ in candidates:
        result = process.extractOne(word, POKEMON_NAMES, scorer=fuzz.WRatio, score_cutoff=85)
        if result is not None:
            match, score, _ = result
            return match

    return "Unknown"


def extract_card_edition(image: Image.Image, debug=False) -> str:
    width, height = image.size
    if debug:
        print(f"[DEBUG] Image size: {width}x{height}")

    # Crop bottom right corner - adjust these ratios as needed
    left = int(width * 0.7)    # right 30% of width
    top = int(height * 0.85)   # bottom 15% of height
    crop_box = (left, top, width, height)
    cropped = image.crop(crop_box)

    if debug:
        print(f"[DEBUG] Cropping box (bottom right corner): {crop_box}")
        cropped.save("debug_edition_crop_bottom_right.png")
        print("[DEBUG] Saved cropped bottom right region to 'debug_edition_crop_bottom_right.png'")

    # Preprocess for better OCR
    processed = preprocess_image(cropped)
    if debug:
        processed.save("debug_edition_processed.png")
        print("[DEBUG] Saved preprocessed cropped region to 'debug_edition_processed.png'")

    # OCR
    text = pytesseract.image_to_string(processed)
    if debug:
        print(f"[DEBUG] OCR Output:\n{text}")

    # Regex patterns
    fraction_pattern = r'\b(\d{1,3})\s*/\s*(\d{1,3})\b'             # e.g. 087/132
    alphanum_pattern = r'\b([a-zA-Z]{2}\d{2})\b'                   # e.g. bv42, AB12

    # Try fraction pattern first
    fraction_match = re.search(fraction_pattern, text)
    if fraction_match:
        num, denom = int(fraction_match.group(1)), int(fraction_match.group(2))
        if num <= denom:
            edition = f"{num}/{denom}"
            if debug:
                print(f"[DEBUG] Edition (fraction) found: {edition}")
                print("[DEBUG] Returning edition from fraction pattern")
            return edition
        else:
            if debug:
                print(f"[DEBUG] Ignored fraction with numerator > denominator: {num}/{denom}")

    # Try alphanumeric pattern next
    alphanum_match = re.search(alphanum_pattern, text, re.IGNORECASE)
    if alphanum_match:
        edition = alphanum_match.group(1).lower()
        if debug:
            print(f"[DEBUG] Edition (alphanumeric) found: {edition}")
            print("[DEBUG] Returning edition from alphanumeric pattern")
        return edition

    if debug:
        print("[DEBUG] No valid edition found in OCR output.")
    return "Unknown"
