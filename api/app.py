from flask import Flask, render_template, request
import os
import json
import re
import nltk
from nltk import word_tokenize, pos_tag

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

app = Flask(__name__)

with open("api/construction_database.json", "r") as f:
    database = json.load(f)

def get_keywords(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    keywords = [word.lower() for word, pos in tagged if pos.startswith("NN")]
    return keywords

def find_week(text):
    week_words = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5
    }

    match = re.search(r"(first|second|third|fourth|fifth|one|two|three|four|five|\d+)\s*(?:st|nd|rd|th)?\s*week|week\s*(first|second|third|fourth|fifth|one|two|three|four|five|\d+)", text, re.IGNORECASE)
    
    if match:
        word = (match.group(1) or match.group(2)).lower()
        word = re.sub(r'(st|nd|rd|th)$', '', word)
        
        if word in week_words:
            return week_words[word]
        elif word.isdigit():
            return int(word)
    
    return None

def find_location(text):
    pattern = r"(first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\s+floor|basement|roof|parking\s+area|main\s+entrance"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [m.lower() for m in matches]

def search(text):
    # Extract query info
    keywords = [k.lower() for k in get_keywords(text)]
    week = find_week(text)
    locations = [loc.lower() for loc in find_location(text)]

    results = []

    for item in database:
        item_text = " ".join([
            str(item.get("location", "")),
            str(item.get("activity", "")),
            str(item.get("hazard", "")),
            str(item.get("description", ""))
        ]).lower()

        item_week = str(item.get("week", ""))
        item_location = str(item.get("location", "")).lower()

        if week is None:
            week_match = True
        else:
            week_match = (item_week == str(week))

        if not locations:
            location_match = True
        else:
            location_match = any(loc in item_location for loc in locations)

        if not keywords:
            keyword_match = True
        else:
            keyword_match = any(k in item_text for k in keywords)

        if week or locations:
            if week_match and location_match:
                results.append(item)
        elif keyword_match:
            results.append(item)

    restrictions = {"week": week, "location": locations}
    return results, keywords, restrictions


@app.route("/", methods=["GET", "POST"])
def home():
    user_input = ""
    keywords = []
    restrictions = {}
    results = []
    
    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        if user_input:
            results, keywords, restrictions = search(user_input)
    
    return render_template(
        "index.html",
        user_input=user_input,
        keywords=keywords,
        restrictions=restrictions,
        results=results
    )


@app.route("/database")
def database_view():
    return render_template("database.html", data=database)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)