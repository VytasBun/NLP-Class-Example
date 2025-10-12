from collections import defaultdict
from flask import Flask, render_template, request
import spacy

# Load spaCy model
try:
    NLP = spacy.load('en_core_web_sm')
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    NLP = spacy.load('en_core_web_sm')

app = Flask(__name__)

# ----------------------------
# Constants
# ----------------------------
POS_SCORES = {
    "PROPN": 5, "NOUN": 5,
    "VERB": 4, "ADJ": 4
}

# Simple stopwords list (no NLTK needed)
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
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
        cards.append(f'''<div class="keyword-card">
            <div class="keyword-word">{k["word"]}</div>
            <div class="keyword-score">Score: {k["score"]:.2f}</div>
            <div class="keyword-pos">POS: {k["pos"]}</div>
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
            <h3>Tokens Generated</h3>
            <span class="hierarchy-count">{len(tokens)}</span>
        </div>
        <div class="token-grid">{"".join(token_items)}</div>
    </div>'''

# ----------------------------
# Enhanced Tree Building
# ----------------------------
def build_dependency_tree_html(doc):
    """Generate beautiful HTML dependency trees for each sentence"""
    trees = []
    
    for sent in doc.sents:
        # Build simple nested list structure
        root = [token for token in sent if token.head == token]
        if not root:
            continue
        root = root[0]
        
        def get_dep_color(dep):
            """Assign colors based on dependency type"""
            colors = {
                'nsubj': '#667eea', 'ROOT': '#f093fb', 'dobj': '#4facfe',
                'prep': '#43e97b', 'pobj': '#38f9d7', 'det': '#fa709a',
                'amod': '#fee140', 'advmod': '#30cfd0', 'aux': '#a8edea',
                'compound': '#fbc2eb', 'conj': '#f6d365', 'cc': '#fda085'
            }
            return colors.get(dep, '#94a3b8')
        
        def build_tree_recursive(token, level=0):
            children = sorted([child for child in token.children], key=lambda x: x.i)
            
            dep_color = get_dep_color(token.dep_)
            is_root = token.dep_ == 'ROOT'
            
            # Build node with visual connector
            html = '<div class="tree-branch">'
            
            if level > 0:
                html += f'<div class="tree-connector" style="width: {level * 30}px;"></div>'
            
            html += f'''<div class="tree-node-card {'root-node' if is_root else ''}" style="border-left-color: {dep_color};">
                <div class="node-header">
                    <span class="node-token">{token.text}</span>
                    <span class="node-pos" style="background: {dep_color};">{token.pos_}</span>
                </div>
                <div class="node-meta">
                    <span class="node-dep" style="color: {dep_color};">
                        <svg width="12" height="12" viewBox="0 0 12 12" style="margin-right: 4px;">
                            <circle cx="6" cy="6" r="4" fill="{dep_color}"/>
                        </svg>
                        {token.dep_}
                    </span>
                    {f'<span class="node-head">→ {token.head.text}</span>' if not is_root else '<span class="node-head root-label">ROOT</span>'}
                </div>
            </div>'''
            
            html += '</div>'
            
            # Recursively add children
            if children:
                html += '<div class="tree-children">'
                for child in children:
                    html += build_tree_recursive(child, level + 1)
                html += '</div>'
            
            return html
        
        tree_html = f'''<div class="tree-wrapper">
            <div class="sentence-label">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                </svg>
                {sent.text}
            </div>
            <div class="tree-container">
                <div class="tree-content">{build_tree_recursive(root)}</div>
            </div>
        </div>'''
        
        trees.append({
            "sentence": sent.text,
            "html": tree_html
        })
    
    return trees

# ----------------------------
# Keyword Analysis
# ----------------------------
def identify_keywords(doc):
    """Extract and rank keywords from document"""
    keyword_map = defaultdict(lambda: {"score": 0, "pos": ""})
    
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        
        text = token.text.lower()
        
        # Skip stopwords
        if text in STOPWORDS:
            continue
            
        if not token.is_alpha:
            continue
            
        score = POS_SCORES.get(token.pos_, 0)
        if score > 0:
            keyword_map[text]["score"] += score
            keyword_map[text]["pos"] = token.pos_
    
    final_keywords = []
    for text, data in keyword_map.items():
        kw = {
            "word": text, 
            "score": data["score"], 
            "pos": data["pos"]
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
            "description": spacy.explain(e.label_) or e.label_,
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
    
    trees = build_dependency_tree_html(doc)
    keywords = identify_keywords(doc)
    
    return {
        "entities": entities,
        "tokens": tokens,
        "dependency_trees": trees,
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
        text_input = request.form.get('text', '').strip()
        
        if not text_input:
            return render_template('index.html', error="No text provided.", input_text=text_input)
        
        try:
            parsed_data = parse_text(text_input)
            return render_template(
                'index.html', 
                parsed_data=parsed_data, 
                input_text=text_input, 
                trees_html=parsed_data["dependency_trees"],
                keywords_html=parsed_data["keywords_html"],
                tokens_html=parsed_data["tokens_html"]
            )
        except Exception as e:
            return render_template('index.html', error=f"Parsing failed: {e}", input_text=text_input)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)