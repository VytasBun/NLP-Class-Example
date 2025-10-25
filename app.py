from flask import Flask, render_template, request
import json
import re
import nltk
from nltk import word_tokenize, pos_tag

# -------------------- NLTK Setup -------------------- #
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

# -------------------- Load Database -------------------- #
with open("construction_database.json", "r") as f:
    database = json.load(f)

# -------------------- Helper Data -------------------- #
ORDINAL_MAP = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


# -------------------- Helper Functions -------------------- #
def stringify(value):
    """Ensure all values are converted to strings for text comparison."""
    if isinstance(value, list):
        return " ".join(map(str, value))
    return str(value)


def extract_nouns(text):
    """Return a list of meaningful nouns from the input text using NLTK."""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    nouns = [word.lower() for word, pos in tagged if pos.startswith("NN")]

    # Remove generic or structural nouns that add noise
    stop_nouns = {"week", "floor", "hazard", "site", "location", "area", "building"}
    clean_nouns = [n for n in nouns if n not in stop_nouns]
    return clean_nouns


def extract_week_number(text):
    """
    Extract week number from text like:
    'third week', 'week three', '2nd week', 'week 2'
    """
    week_pattern = r"(?:(first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\s+week|week\s+(one|two|three|four|five|\d+))"
    match = re.search(week_pattern, text, re.IGNORECASE)
    if not match:
        return None

    val = (match.group(1) or match.group(2)).lower()

    # safely remove ordinal suffixes only from digits
    val = re.sub(r'(\d+)(st|nd|rd|th)$', r'\1', val)

    # normalize to int if possible
    if val in ORDINAL_MAP:
        return ORDINAL_MAP[val]
    elif val.isdigit():
        return int(val)
    return val


def extract_location(text):
    """
    Extract and normalize location phrases like:
    'second floor', 'basement', 'roof', 'parking area', 'main entrance', 'main site'
    """
    pattern = r"((?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\s+floor|basement|roof|parking\s+area|main\s+(?:entrance|site))"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [m.strip().lower() for m in matches]  # lowercase for consistency


def extract_restrictions(text):
    """Combine regex extractions for week and location."""
    restrictions = {
        "location": extract_location(text),
        "week": extract_week_number(text)
    }
    return restrictions


def search_database(nouns, restrictions):
    """Search the database for matching hazards based on nouns and restrictions."""
    results = []

    for entry in database:
        entry_text = " ".join([
            stringify(entry.get("location", "")),
            stringify(entry.get("activity", "")),
            stringify(entry.get("hazard", "")),
            stringify(entry.get("hazardType", "")),
            stringify(entry.get("description", "")),
            stringify(entry.get("timeline", "")),
            stringify(entry.get("week", "")),
        ]).lower()

        # Step 1: Noun match (case-insensitive)
        noun_match = any(noun.lower() in entry_text for noun in nouns) if nouns else True

        # Step 2: Restriction matches
        loc_match = True
        week_match = True

        # --- Location Filtering ---
        if restrictions.get("location"):
            loc_text = stringify(entry.get("location", "")).lower()
            loc_match = any(loc in loc_text for loc in restrictions["location"])

        # --- Week Filtering ---
        if restrictions.get("week") is not None:
            entry_week = stringify(entry.get("week", "")).lower()
            try:
                week_match = int(entry_week) == int(restrictions["week"])
            except ValueError:
                week_match = str(restrictions["week"]).lower() in entry_week

        if noun_match and loc_match and week_match:
            results.append(entry)

    return results


# -------------------- Routes -------------------- #
@app.route("/", methods=["GET", "POST"])
def home():
    user_input = ""
    nouns = []
    restrictions = {}
    results = []

    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        if user_input.strip():
            nouns = extract_nouns(user_input)
            restrictions = extract_restrictions(user_input)
            results = search_database(nouns, restrictions)

    return render_template(
        "index.html",
        user_input=user_input,
        keywords=nouns,
        restrictions=restrictions,
        results=results
    )


@app.route("/database")
def database_view():
    return render_template("database.html", data=database)


if __name__ == "__main__":
    app.run(debug=True)
