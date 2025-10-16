from collections import defaultdict
from flask import Flask, render_template, request
import spacy, os

# Load SPACY en_core_web_sm

try:
    NLP = spacy.load('en_core_web_sm')
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    NLP = spacy.load('en_core_web_sm')

app = Flask(__name__)

# Constants

POS_SCORES = {"PROPN": 5, "NOUN": 5, "VERB": 4, "ADJ": 4}
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself',
    'yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself',
    'they','them','their','theirs','themselves','what','which','who','whom','this','that',
    'these','those','am','is','are','was','were','be','been','being','have','has','had',
    'having','do','does','did','doing','a','an','the','and','but','if','or','because','as',
    'until','while','of','at','by','for','with','about','against','between','into','through',
    'during','before','after','above','below','to','from','up','down','in','out','on','off',
    'over','under','again','further','then','once'
}


# Keyword Extraction

def identify_keywords(doc):
    keywords = defaultdict(lambda: {"score": 0, "pos": ""})
    for token in doc:
        if token.is_alpha and token.text.lower() not in STOPWORDS:
            score = POS_SCORES.get(token.pos_, 0)
            if score:
                kw = keywords[token.text.lower()]
                kw["score"] += score
                kw["pos"] = token.pos_
    ranked = sorted(
        [{"word": k, "score": v["score"], "pos": v["pos"]} for k, v in keywords.items()],
        key=lambda x: x["score"], reverse=True
    )
    return ranked

# HTML Rendering

def render_cards(items, template):
    """Reusable helper for HTML sections"""
    return "".join(template(i) for i in items)

def render_keywords_html(ranked):
    if not ranked:
        return ""
    def card(k):
        return f"""
        <div class="keyword-card">
            <div class="keyword-word">{k['word']}</div>
            <div class="keyword-score">Score: {k['score']:.2f}</div>
            <div class="keyword-pos">POS: {k['pos']}</div>
        </div>"""
    return f"""
    <div class="keyword-section">
        <h3>Keywords Found</h3>
        <div class="card-grid">{render_cards(ranked[:12], card)}</div>
    </div>"""

def render_tokens_html(tokens):
    if not tokens:
        return ""
    def item(t):
        return f"""
        <div class="token-item">
            <div class="token-word">{t['text']}</div>
            <div class="token-details">
                <div class="token-detail-row"><span class="detail-label">Lemma:</span><span>{t['lemma']}</span></div>
                <div class="token-detail-row"><span class="detail-label">POS:</span><span class="pos-badge">{t['pos']}</span></div>
                <div class="token-detail-row"><span class="detail-label">Tag:</span><span>{t['tag']}</span></div>
                <div class="token-detail-row"><span class="detail-label">Dep:</span><span class="dep-badge">{t['dep']}</span></div>
                <div class="token-detail-row"><span class="detail-label">Head:</span><span>{t['head']}</span></div>
            </div>
        </div>"""
    return f"""
    <div class="hierarchy-section">
        <div class="hierarchy-header">
            <h3>Tokens Generated</h3><span class="hierarchy-count">{len(tokens)}</span>
        </div>
        <div class="token-grid">{render_cards(tokens, item)}</div>
    </div>"""

# Constituent Tree

def build_dependency_tree_html(doc):
    dep_colors = {
        'nsubj': '#667eea', 'ROOT': '#f093fb', 'dobj': '#4facfe', 'prep': '#43e97b',
        'pobj': '#38f9d7', 'det': '#fa709a', 'amod': '#fee140', 'advmod': '#30cfd0',
        'aux': '#a8edea', 'compound': '#fbc2eb', 'conj': '#f6d365', 'cc': '#fda085'
    }
    def color(dep): return dep_colors.get(dep, '#94a3b8')

    def build_tree(token, level=0):
        children = sorted(list(token.children), key=lambda x: x.i)
        dep_color = color(token.dep_)
        is_root = token.dep_ == "ROOT"
        connector = f'<div class="tree-connector" style="width:{level*30}px;"></div>' if level else ''
        node_html = f"""
        <div class="tree-branch">
            {connector}
            <div class="tree-node-card {'root-node' if is_root else ''}" style="border-left-color:{dep_color};">
                <div class="node-header">
                    <span class="node-token">{token.text}</span>
                    <span class="node-pos" style="background:{dep_color};">{token.pos_}</span>
                </div>
                <div class="node-meta">
                    <span class="node-dep" style="color:{dep_color};">
                        <svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="{dep_color}"/></svg>
                        {token.dep_}
                    </span>
                    <span class="node-head">{'ROOT' if is_root else f'→ {token.head.text}'}</span>
                </div>
            </div>
        </div>"""
        if children:
            node_html += f'<div class="tree-children">{"".join(build_tree(c, level+1) for c in children)}</div>'
        return node_html

    trees = []
    for sent in doc.sents:
        root = next((t for t in sent if t.head == t), None)
        if not root: continue
        html = f"""
        <div class="tree-wrapper">
            <div class="sentence-label">{sent.text}</div>
            <div class="tree-container"><div class="tree-content">{build_tree(root)}</div></div>
        </div>"""
        trees.append({"sentence": sent.text, "html": html})
    return trees


# Text Entry Parser

def parse_text(text):
    doc = NLP(text)
    entities = [{
        "text": e.text, "label": e.label_,
        "description": spacy.explain(e.label_) or e.label_,
        "start": e.start_char, "end": e.end_char
    } for e in doc.ents]

    tokens = [{
        "text": t.text, "lemma": t.lemma_, "pos": t.pos_,
        "tag": t.tag_, "dep": t.dep_, "head": t.head.text
    } for t in doc]

    ranked = identify_keywords(doc)
    trees = build_dependency_tree_html(doc)

    return {
        "entities": entities,
        "tokens": tokens,
        "dependency_trees": trees,
        "keyword_analysis": {"ranked_keywords": ranked},
        "keywords_html": render_keywords_html(ranked),
        "tokens_html": render_tokens_html(tokens)
    }

# Flask Route Handeling

@app.route('/', methods=['GET', 'POST'])
def index():
    text = request.form.get('text', '').strip() if request.method == 'POST' else ''
    if request.method == 'POST':
        if not text:
            return render_template('index.html', error="No text provided.", input_text=text)
        try:
            parsed = parse_text(text)
            return render_template('index.html',
                parsed_data=parsed, input_text=text,
                trees_html=parsed["dependency_trees"],
                keywords_html=parsed["keywords_html"],
                tokens_html=parsed["tokens_html"])
        except Exception as e:
            return render_template('index.html', error=f"Parsing failed: {e}", input_text=text)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
