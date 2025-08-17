from difflib import get_close_matches
import requests

# Placeholder card list
known_cards = ["Charizard", "Blastoise", "Pikachu", "Bulbasaur"]

def extract_name(text: str):
    words = text.split()
    match = get_close_matches(' '.join(words), known_cards, n=1, cutoff=0.6)
    return match[0] if match else "Unknown Card"

def get_price(card_name: str, grade: str):
    # Dummy implementation – replace with real API calls like TCGPlayer or PriceCharting
    if card_name == "Unknown Card":
        return 0.0
    return round(100.0 / (int(grade) if grade.isdigit() else 1), 2)
