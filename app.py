from collections import defaultdict
from flask import Flask, render_template, request
import spacy
import nltk
from nltk import Tree
from io import StringIO
import svgwrite

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

NLP = spacy.load('en_core_web_sm')
app = Flask(__name__)

# ----------------------------
# Constants
# ----------------------------
POS_SCORES = {
    "PROPN": 5, "NOUN": 5,
    "VERB": 4, "ADJ": 4
}

# ----------------------------
# HTML Rendering Functions
# ----------------------------
def render_keywords_html(ranked):
    """Render keywords as HTML"""
    if not ranked:
        return ""
    
    cards = []
    for k in ranked[:12]:
        roles_html = "".join(f'<span class="role-badge">{role}</span>' for role in k.get("roles", []))
        constraints = k.get("constraints", [])[:3]
        constraints_html = ""
        
        if constraints:
            constraint_items = "".join(
                f'<div class="constraint-item"><span class="constraint-type">{c["type"]}</span><span>{c["text"]}</span></div>'
                for c in constraints
            )
            constraints_html = f'''<div class="keyword-constraints">
                <div class="constraint-label">Related Context:</div>
                <div class="constraint-list">{constraint_items}</div>
            </div>'''
        
        cards.append(f'''<div class="keyword-card">
            <div class="keyword-word">{k["word"]}</div>
            <div class="keyword-score">Score: {k["score"]:.2f}</div>
            {f'<div class="keyword-roles">{roles_html}</div>' if roles_html else ''}
            {constraints_html}
        </div>''')
    
    return f'''<div class="keyword-section">
        <h3>Keywords Found</h3>
        <div class="card-grid">{"".join(cards)}</div>
    </div>'''

def render_tokens_html(tokens):
    """Render tokens as HTML"""
    if not tokens:
        return ""
    
    token_items = []
    for t in tokens:
        token_items.append(f'''<div class="token-item">
            <div class="token-word">{t["text"]}</div>
            <div class="token-details">
                <div class="token-detail-row">
                    <span class="detail-label">Lemma:</span>
                    <span class="detail-value">{t["lemma"]}</span>
                </div>
                <div class="token-detail-row">
                    <span class="detail-label">POS:</span>
                    <span class="pos-badge">{t["pos"]}</span>
                </div>
                <div class="token-detail-row">
                    <span class="detail-label">Tag:</span>
                    <span class="detail-value">{t["tag"]}</span>
                </div>
                <div class="token-detail-row">
                    <span class="detail-label">Dep:</span>
                    <span class="dep-badge">{t["dep"]}</span>
                </div>
                <div class="token-detail-row">
                    <span class="detail-label">Head:</span>
                    <span class="detail-value">{t["head"]}</span>
                </div>
            </div>
        </div>''')
    
    return f'''<div class="hierarchy-section">
        <div class="hierarchy-header">
            <h3>Token Generated</h3>
            <span class="hierarchy-count">{len(tokens)}</span>
        </div>
        <div class="token-grid">{"".join(token_items)}</div>
    </div>'''

# ----------------------------
# Tree Building Functions
# ----------------------------
def _to_nltk_tree(token):
    """Convert spaCy dependency tree to NLTK Tree"""
    children = sorted(token.children, key=lambda x: x.i)
    if not children:
        return token.text
    
    tree_children = []
    for child in children:
        tree_children.append(_to_nltk_tree(child))
    
    return Tree(token.pos_, [token.text] + tree_children)

def _tree_to_svg(tree, x=0, y=0, level=0, dwg=None, parent_center=None):
    """Recursively convert NLTK tree to responsive SVG using svgwrite."""
    node_height = 40
    level_height = 80
    padding = 10

    # Initialize drawing on first call
    if dwg is None:
        dwg = svgwrite.Drawing(profile='tiny')
        dwg.attribs['preserveAspectRatio'] = 'xMidYMin meet'

    # Handle leaf node (word)
    if isinstance(tree, str):
        width = len(tree) * 8 + padding * 2
        center_x = x + width / 2
        center_y = y + node_height / 2

        # Node rectangle
        dwg.add(dwg.rect(
            insert=(x, y),
            size=(width, node_height),
            rx=5, ry=5,
            fill="white",
            stroke="#667eea",
            stroke_width=2
        ))

        # Node text
        dwg.add(dwg.text(
            tree,
            insert=(center_x, center_y + 5),
            text_anchor="middle",
            font_size="14px",
            fill="#2d3748"
        ))

        # Connect to parent if applicable
        if parent_center:
            dwg.add(dwg.line(
                start=parent_center,
                end=(center_x, y),
                stroke="#aaa",
                stroke_width=2
            ))

        return dwg, width, center_x

    # Internal node (phrase)
    label = tree.label()
    children = tree

    total_width = 0
    child_centers = []
    for child in children:
        dwg, child_width, child_center_x = _tree_to_svg(
            child,
            x + total_width,
            y + level_height,
            level + 1,
            dwg=dwg,
            parent_center=None  # connect after parent drawn
        )
        child_centers.append((child_center_x, y + level_height))
        total_width += child_width + padding

    total_width -= padding  # remove trailing padding

    # Parent node
    label_width = len(label) * 10 + padding * 2
    parent_x = x + (total_width - label_width) / 2
    parent_center_x = parent_x + label_width / 2
    parent_center_y = y + node_height / 2

    # Draw parent box
    dwg.add(dwg.rect(
        insert=(parent_x, y),
        size=(label_width, node_height),
        rx=5, ry=5,
        fill="#667eea",
        stroke="#5568d3",
        stroke_width=2
    ))

    dwg.add(dwg.text(
        label,
        insert=(parent_center_x, parent_center_y + 5),
        text_anchor="middle",
        font_size="14px",
        font_weight="bold",
        fill="white"
    ))

    # Draw connections
    for cx, cy in child_centers:
        dwg.add(dwg.line(
            start=(parent_center_x, y + node_height),
            end=(cx, cy),
            stroke="#aaa",
            stroke_width=2
        ))

    return dwg, total_width, parent_center_x

def build_constituent_trees(doc):
    """Generate responsive SVG constituent trees for each sentence."""
    trees = []

    for sent in doc.sents:
        # Find root of the sentence
        root = [token for token in sent if token.head == token][0]
        nltk_tree = Tree('S', [_to_nltk_tree(root)])

        # Generate SVG recursively
        dwg, width, _ = _tree_to_svg(nltk_tree, x=20, y=20)

        # Compute approximate height based on tree depth
        def get_depth(t):
            if isinstance(t, str):
                return 1
            return 1 + max([get_depth(child) for child in t] or [0])

        depth = get_depth(nltk_tree)
        height = depth * 80 + 100  # scalable height

        # Add responsive attributes (width 100%, no explicit height)
        dwg.viewbox(0, 0, width + 60, height)
        dwg.attribs['width'] = '100%'  # scales to container width
        # DO NOT set height here — CSS will handle it

        svg_html = dwg.tostring()

        trees.append({
            "sentence": sent.text,
            "html": svg_html
        })

    return trees

# ----------------------------
# Keyword Analysis
# ----------------------------
def identify_keywords(doc):
    """Extract and rank keywords from document"""
    keyword_map = defaultdict(lambda: {"score": 0, "roles": set(), "constraints": []})
    
    for token in doc:
        if token.is_punct or token.is_stop or not token.is_alpha:
            continue
        text = token.text.lower()
        score = POS_SCORES.get(token.pos_, 0)
        keyword_map[text]["score"] += score
    
    final_keywords = []
    for text, data in keyword_map.items():
        unique_constraints = list({c["text"]: c for c in data["constraints"]}.values())
        kw = {
            "word": text, 
            "score": data["score"], 
            "roles": list(data["roles"]), 
            "constraints": unique_constraints
        }
        final_keywords.append(kw)
    
    final_keywords.sort(key=lambda x: x["score"], reverse=True)
    return {"ranked_keywords": final_keywords}

# ----------------------------
# Main Parser
# ----------------------------
def parse_text(text):
    """Main parsing function"""
    if not text.strip():
        raise RuntimeError("No text provided.")
    
    doc = NLP(text)
    
    entities = [
        {
            "text": e.text, 
            "label": e.label_, 
            "description": spacy.explain(e.label_),
            "start": e.start_char, 
            "end": e.end_char
        } 
        for e in doc.ents
    ]
    
    tokens = [
        {
            "text": t.text, 
            "lemma": t.lemma_, 
            "pos": t.pos_, 
            "tag": t.tag_, 
            "dep": t.dep_, 
            "head": t.head.text
        } 
        for t in doc
    ]
    
    trees = build_constituent_trees(doc)
    keywords = identify_keywords(doc)
    
    return {
        "entities": entities,
        "tokens": tokens,
        "constituent_trees": trees,
        "keyword_analysis": keywords,
        "keywords_html": render_keywords_html(keywords["ranked_keywords"]),
        "tokens_html": render_tokens_html(tokens)
    }

# ----------------------------
# Flask Routes
# ----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resume_text = request.form.get('text', '').strip()
        
        if not resume_text:
            return render_template('index.html', error="No text provided.", input_text=resume_text)
        
        try:
            parsed_data = parse_text(resume_text)
            return render_template(
                'index.html', 
                parsed_data=parsed_data, 
                input_text=resume_text, 
                trees_html=parsed_data["constituent_trees"],
                keywords_html=parsed_data["keywords_html"],
                tokens_html=parsed_data["tokens_html"]
            )
        except Exception as e:
            return render_template('index.html', error=f"Parsing failed: {e}", input_text=resume_text)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
