import re
import unicodedata
import spacy

# Load spaCy for NLP processing
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
    Cleans extracted text by:
    - Removing extra spaces and formatting issues
    - Preserving essential symbols (%, $, €, -)
    - Normalizing unicode characters
    - Keeping stop words for context

    :param text: Raw extracted text
    :return: Cleaned and structured text
    """
    # Convert to lowercase
    text = text.lower()

    # Normalize unicode characters (fix encoding issues)
    text = unicodedata.normalize("NFKD", text)

    # Remove unwanted characters but KEEP important symbols
    text = re.sub(r"[^a-zA-Z0-9.,!?$€%'-]", " ", text)  # Keeps $, €, %, and dashes

    # Remove extra spaces, tabs, and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and reassemble (removes only extreme noise)
    words = text.split()
    cleaned_text = " ".join(words)

    return cleaned_text

# ✅ Test the function
if __name__ == "__main__":
    sample_text = "Bitcoin is a decentralized digital currency, but it has been used in fraud cases! SEC's approval of ETFs was a game-changer."
    cleaned_sample = clean_text(sample_text)
    
    print("\n🔹 Original Text:\n", sample_text)
    print("\n✅ Cleaned Text:\n", cleaned_sample)
