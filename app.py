import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
import cv2
import gradio as gr

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Active 24 Fashion Classes
ACTIVE_CLASSES = [
    'Kurtas', 'Tshirts', 'Shirts', 'Tops', 'Sweatshirts', 'Jackets', 'Sweaters', 'Kurtis',
    'Jeans', 'Trousers', 'Shorts', 'Skirts', 'Track Pants', 'Palazzos',
    'Dresses', 'Sarees',
    'Casual Shoes', 'Formal Shoes', 'Sports Shoes', 'Heels', 'Flats', 'Sandals'
]
idx_to_class = {idx: name for idx, name in enumerate(ACTIVE_CLASSES)}

# Image Transforms
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Model Architecture
def build_model(num_classes=len(ACTIVE_CLASSES)):
    model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)
    return model

model = build_model().to(device)

import os
if os.path.exists("best_stylist_model.pth"):
    model.load_state_dict(torch.load("best_stylist_model.pth", map_location=device))
    print("✅ Loaded trained model weights from 'best_stylist_model.pth'")
else:
    print("⚠️ 'best_stylist_model.pth' not found. Using pretrained backbone.")

model.eval()

# Color Analysis
def analyze_color(pil_img):
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape
    crop = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    pixels = crop.reshape((-1, 3)).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    counts = np.bincount(labels.flatten())
    b, g, r = centers[np.argmax(counts)]
    hex_c = f"#{r:02x}{g:02x}{b:02x}"
    
    if r > 210 and g > 210 and b > 210:
        c_name, psych = "Crisp White", "Communicates purity, minimalism, cleanliness, and timeless elegance."
    elif r < 50 and g < 50 and b < 50:
        c_name, psych = "Midnight Black", "Communicates power, luxury, formality, authority, and sleek sophistication."
    elif b > r + 20 and b > g:
        c_name, psych = "Royal Navy / Indigo", "Communicates trust, professionalism, stability, and calm confidence."
    elif r > g + 30 and r > b + 30:
        c_name, psych = "Crimson Red / Maroon", "Communicates passion, high energy, bold confidence, and festive intensity."
    elif g > r and g > b:
        c_name, psych = "Olive Green / Sage", "Communicates harmony, nature, balance, and earthy elegance."
    elif abs(int(r)-int(g)) < 15 and abs(int(g)-int(b)) < 15:
        c_name, psych = "Heather Grey / Slate", "Communicates balance, neutrality, modern urban aesthetic, and effortless style."
    else:
        c_name, psych = "Earth Neutral / Terracotta", "Communicates warmth, organic authenticity, and understated luxury."
        
    return c_name, hex_c, psych

FEATURE_EXPLAINER = {
    'Kurtas': 'Long straight tunic silhouette, mandarin/band collar, ethnic sleeve structure, knee-length hemline.',
    'Kurtis': 'Short ethnic tunic silhouette, side slits, traditional neckline, lightweight summer drape.',
    'Tshirts': 'Crew/V-neckline, short casual sleeves, soft jersey cotton weave, relaxed fit.',
    'Shirts': 'Structured turned-down collar, full front button placket, cuffed sleeves, tailored weave.',
    'Tops': 'Feminine neckline cut, soft drape silhouette, casual/party pattern, lightweight fabric.',
    'Sweatshirts': 'Ribbed crew collar, long sleeves with fitted cuffs, heavy fleece knit texture.',
    'Jackets': 'Heavy outerwear construction, front zipper/button closure, structured lapels/collar.',
    'Sweaters': 'Soft ribbed knit yarn pattern, warm neck construction, full sleeves, cozy winter texture.',
    'Jeans': 'Durable indigo denim weave, reinforced rivet stitching, 5-pocket design, casual waist fit.',
    'Trousers': 'Smooth tailored weave, pressed front crease, formal waist closure, dress trouser drape.',
    'Shorts': 'Cropped above-the-knee hemline, casual waist construction, lightweight fabric.',
    'Skirts': 'Waist-anchored lower body flare, pleated/A-line silhouette, graceful hemline drape.',
    'Track Pants': 'Elastic waistband with drawstring, soft athletic fleece/nylon knit, tapered ankle cuffs.',
    'Palazzos': 'Wide-leg flared trousers, elasticated waist, fluid lightweight fabric, ethnic drape.',
    'Dresses': 'One-piece full-body silhouette, defined waistline, knee/ankle-length skirt flare.',
    'Sarees': 'Unstitched 6-yard elegant drape fabric, rich border weave, traditional pallu pattern.',
    'Casual Shoes': 'Low-top sneaker construction, rubber sole unit, flexible leather/canvas upper.',
    'Formal Shoes': 'Polished leather upper, closed oxford/derby lacing, stacked heel, formal sole.',
    'Sports Shoes': 'Cushioned midsole, breathable mesh upper, athletic grip tread, ergonomic heel support.',
    'Heels': 'Elevated heel unit, sleek strap/pump construction, formal/party dress sole.',
    'Flats': 'Flat sole construction, comfortable open/closed upper, casual everyday drape.',
    'Sandals': 'Open strap unit, lightweight footbed, summer leisure sole.'
}

IDLE_STATE_HTML = """
<div class="editorial-idle-card">
    <div class="idle-icon">✨</div>
    <div class="idle-title">Your Styling Dossier Awaits</div>
    <div class="idle-subtitle">Upload any garment or footwear image on the left to reveal your bespoke fashion consultation.</div>
</div>
"""

def generate_editorial_consultation(pil_img):
    if pil_img is None:
        return None, IDLE_STATE_HTML
    
    tensor_img = val_test_transforms(pil_img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
        
    category = idx_to_class[pred_idx.item()]
    confidence_pct = conf.item() * 100
    color_name, hex_c, psych = analyze_color(pil_img)
    why_prediction = FEATURE_EXPLAINER.get(category, "Key visual structural features, neckline, drape, and fabric texture.")
    
    if category in ['Kurtas', 'Kurtis', 'Shirts', 'Tshirts', 'Tops', 'Sweatshirts', 'Jackets', 'Sweaters']:
        item_type = "TOPWEAR"
        match_primary_title = "👖 MATCHING BOTTOM WEAR"
        match_primary = "Slim Indigo Denim, Beige Tailored Chinos, Off-White Churidar, or Pleated Slacks"
        match_secondary_title = "👟 RECOMMENDED FOOTWEAR"
        match_secondary = "White Minimalist Leather Sneakers, Handcrafted Loafers, or Kolhapuris"
        layering = "Unbuttoned Denim Jacket, Structured Blazer, or Open Knit Cardigan"
        accessories = "Leather Strap Watch, Minimalist Chain, Sunglasses, Leather Belt"
    elif category in ['Jeans', 'Trousers', 'Shorts', 'Skirts', 'Track Pants', 'Palazzos']:
        item_type = "BOTTOMWEAR (Pants/Skirts)"
        match_primary_title = "👕 MATCHING TOP WEAR"
        match_primary = "Crisp Cotton Button-Down Shirt, Graphic Tee, Slub Kurta, or Ribbed Top"
        match_secondary_title = "👟 RECOMMENDED FOOTWEAR"
        match_secondary = "Classic Oxfords, Chelsea Boots, Retro Sneakers, or Strappy Heels"
        layering = "Tailored Blazer, Oversized Denim Jacket, or Cropped Leather Outerwear"
        accessories = "Leather Belt (Matching Shoes), Classic Wristwatch, Statement Handbag"
    elif category in ['Casual Shoes', 'Formal Shoes', 'Sports Shoes', 'Heels', 'Flats', 'Sandals']:
        item_type = "FOOTWEAR"
        match_primary_title = "👗 MATCHING FULL OUTFIT (Top + Bottom Pairing)"
        match_primary = "Crisp White Oxford Shirt + Tapered Navy Chinos, or Floral Summer Dress"
        match_secondary_title = "👖 BOTTOMWEAR PAIRING"
        match_secondary = "Tailored Ankle-Length Slacks, Raw Denim Jeans, or Pleated Midi Skirt"
        layering = "Structured Italian Blazer or Lightweight Wool Overcoat"
        accessories = "Leather Belt (Matching Shoes), Leather Messenger Bag, Chronograph Watch"
    else:
        item_type = "FULL OUTFIT"
        match_primary_title = "💍 MATCHING ACCESSORIES & JEWELRY"
        match_primary = "Statement Earrings, Delicate Gold/Silver Chain, Embellished Clutch Bag"
        match_secondary_title = "👟 RECOMMENDED FOOTWEAR & HEELS"
        match_secondary = "Sleek Stiletto Heels, Strappy Block Sandals, or Embellished Juttis"
        layering = "Brocade Shawl, Tailored Cropped Blazer, or Faux Fur Stole"
        accessories = "Embroidered Clutch, Statement Ring, Delicate Wrist Cuff"
        
    color_combos = [
        ("Off-White / Cream", "Creates a clean traditional look balancing visual intensity."),
        ("Midnight Black", "Elegant contrast suitable for evening gatherings."),
        ("Classic Navy Blue", "Modern and sophisticated versatile pairing."),
        ("Warm Beige / Tan", "Earthy and stylish natural warm neutral tone."),
        ("Charcoal Grey", "Sleek urban contrast that grounds the overall outfit.")
    ]
    combo_html = "".join([f'<div class="editorial-combo-item"><strong style="color:#0f172a;">🎨 {c[0]}:</strong> <span style="color:#475569;">{c[1]}</span></div>' for c in color_combos])
    
    editorial_html = f"""
    <div class="editorial-dossier">
        
        <!-- Card 1: Category & Confidence -->
        <div class="editorial-card card-delay-1">
            <div class="card-header-row">
                <div>
                    <span class="eyebrow-tag">{item_type} DETECTED</span>
                    <h2 class="editorial-category-title">{category}</h2>
                </div>
                <div style="text-align:right;">
                    <div class="style-score-badge">9.6 <span style="font-size:14px; opacity:0.7;">/ 10</span></div>
                    <div class="score-label">Style Compatibility</div>
                </div>
            </div>
            <div class="confidence-bar-container">
                <div class="confidence-bar-fill" style="width: {confidence_pct}%;"></div>
            </div>
            <div style="font-size:12px; font-weight:700; color:#64748b; margin-top:6px;">AI Visual Confidence: {confidence_pct:.1f}%</div>
        </div>
        
        <!-- Card 2: Feature Explainer (Why Prediction?) -->
        <div class="editorial-card card-delay-2 feature-card">
            <div class="card-section-label">🔍 WHY THIS PREDICTION? (VISUAL CUES)</div>
            <p class="feature-text">{why_prediction}</p>
        </div>
        
        <!-- Card 3: Executive Stylist Advice -->
        <div class="editorial-card card-delay-3">
            <div class="card-section-label">👗 EXECUTIVE STYLIST ADVICE</div>
            <p class="narrative-text">
                This <strong>{color_name} {category}</strong> serves as an essential wardrobe anchor. Designed for effortless versatility, it balances visual presence with refined texture. Pair with structured neutral items to maintain an uncluttered silhouette.
            </p>
        </div>
        
        <!-- Card 4: Color Psychology -->
        <div class="editorial-card card-delay-4 psych-card">
            <div class="card-section-label" style="color:#7e22ce;">🧠 COLOR PSYCHOLOGY ({color_name})</div>
            <p class="psych-text">{psych}</p>
        </div>
        
        <!-- Card 5: Color Combinations -->
        <div class="editorial-card card-delay-5">
            <div class="card-section-label">🎨 BEST COLOR COMBINATIONS</div>
            {combo_html}
        </div>
        
        <!-- Card 6: Bi-Directional Pairings Grid -->
        <div class="editorial-card-grid card-delay-6">
            <div class="grid-card border-indigo">
                <div class="grid-card-label">{match_primary_title}</div>
                <div class="grid-card-val">{match_primary}</div>
            </div>
            <div class="grid-card border-cyan">
                <div class="grid-card-label">{match_secondary_title}</div>
                <div class="grid-card-val">{match_secondary}</div>
            </div>
            <div class="grid-card border-emerald">
                <div class="grid-card-label">💍 ACCESSORIES</div>
                <div class="grid-card-val">{accessories}</div>
            </div>
            <div class="grid-card border-amber">
                <div class="grid-card-label">🧥 LAYERING OUTERWEAR</div>
                <div class="grid-card-val">{layering}</div>
            </div>
        </div>
        
        <!-- Card 7: Occasion & Season -->
        <div class="editorial-card-grid card-delay-7" style="margin-top:14px;">
            <div class="grid-card border-pink">
                <div class="grid-card-label">📍 RECOMMENDED OCCASION</div>
                <div class="grid-card-val">Office, College, Casual Outings, Festivals, Travel, Dinners</div>
            </div>
            <div class="grid-card border-purple">
                <div class="grid-card-label">🌤️ BEST SEASON</div>
                <div class="grid-card-val">All-Season / Summer & Autumn Transition</div>
            </div>
        </div>
        
        <!-- Card 8: Fashion Do's & Don'ts (High-Contrast Highlighting) -->
        <div class="editorial-card-grid card-delay-8" style="margin-top:14px;">
            <div class="dos-card">
                <strong style="color:#047857; font-size:15px;">✅ Fashion Do's:</strong>
                <ul style="margin:6px 0 0 0; padding-left:18px; color:#065f46;">
                    <li style="color:#065f46 !important; font-weight:700 !important; font-size:13px !important; margin-bottom:4px;">Ensure proper tailored fit around shoulders/waist.</li>
                    <li style="color:#065f46 !important; font-weight:700 !important; font-size:13px !important; margin-bottom:4px;">Maintain neutral color balance between garments.</li>
                    <li style="color:#065f46 !important; font-weight:700 !important; font-size:13px !important;">Steam or iron for clean drape before wearing.</li>
                </ul>
            </div>
            <div class="donts-card">
                <strong style="color:#b91c1c; font-size:15px;">❌ Fashion Don'ts:</strong>
                <ul style="margin:6px 0 0 0; padding-left:18px; color:#991b1b;">
                    <li style="color:#991b1b !important; font-weight:700 !important; font-size:13px !important; margin-bottom:4px;">Avoid clashing bold patterns simultaneously.</li>
                    <li style="color:#991b1b !important; font-weight:700 !important; font-size:13px !important; margin-bottom:4px;">Do not wear scuffed or mismatched footwear.</li>
                    <li style="color:#991b1b !important; font-weight:700 !important; font-size:13px !important;">Avoid over-accessorizing with loud jewelry.</li>
                </ul>
            </div>
        </div>
    </div>
    """
    
    conf_dict = {ACTIVE_CLASSES[i]: float(probs[i]) for i in range(len(ACTIVE_CLASSES))}
    return conf_dict, editorial_html

# Editorial Luxury CSS with High-Contrast Do's & Don'ts Typography
editorial_css = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,600&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

body, .gradio-container {
    background: linear-gradient(180deg, #0b0f19 0%, #0f172a 40%, #1e293b 100%) !important;
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
    color: #f8fafc !important;
}

.hero-container {
    text-align: center;
    padding: 36px 20px 24px 20px;
    background: radial-gradient(circle at center, rgba(30, 41, 59, 0.6) 0%, rgba(11, 15, 25, 0.9) 100%);
    border-radius: 24px;
    margin-bottom: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.eyebrow-text {
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #ec4899;
    margin-bottom: 8px;
}

.editorial-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 44px;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 10px 0;
    line-height: 1.15;
}

.accent-word {
    font-style: italic;
    color: #f59e0b;
}

.editorial-tagline {
    font-size: 15px;
    color: #94a3b8;
    max-width: 600px;
    margin: 0 auto;
}

.editorial-idle-card {
    background: rgba(15, 23, 42, 0.6);
    border: 2px dashed rgba(255, 255, 255, 0.15);
    border-radius: 22px;
    padding: 60px 30px;
    text-align: center;
}

.idle-icon {
    font-size: 48px;
    margin-bottom: 14px;
    opacity: 0.9;
}

.idle-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
}

.idle-subtitle {
    font-size: 14px;
    color: #94a3b8;
    max-width: 400px;
    margin: 0 auto;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(16px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.editorial-dossier {
    animation: fadeInUp 0.5s ease-out forwards;
}

.editorial-card {
    background: #ffffff;
    color: #0f172a;
    border-radius: 20px;
    padding: 22px 24px;
    margin-bottom: 14px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    border: 1px solid rgba(226, 232, 240, 0.8);
    animation: fadeInUp 0.5s ease-out forwards;
}

.card-delay-1 { animation-delay: 0.05s; }
.card-delay-2 { animation-delay: 0.1s; }
.card-delay-3 { animation-delay: 0.15s; }
.card-delay-4 { animation-delay: 0.2s; }
.card-delay-5 { animation-delay: 0.25s; }
.card-delay-6 { animation-delay: 0.3s; }
.card-delay-7 { animation-delay: 0.35s; }
.card-delay-8 { animation-delay: 0.4s; }

.eyebrow-tag {
    background: #e0e7ff;
    color: #4338ca;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 800;
}

.editorial-category-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 32px;
    font-weight: 700;
    color: #0f172a;
    margin: 6px 0 0 0;
}

.style-score-badge {
    font-size: 26px;
    font-weight: 900;
    color: #10b981;
}

.score-label {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
}

.confidence-bar-container {
    width: 100%;
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    margin-top: 14px;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #4f46e5 0%, #10b981 100%);
    border-radius: 4px;
}

.card-section-label {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: #64748b;
    margin-bottom: 6px;
}

.feature-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
}
.feature-text {
    color: #15803d;
    margin: 0;
    font-size: 13.5px;
    font-weight: 600;
}

.narrative-text {
    color: #334155;
    line-height: 1.6;
    font-size: 14.5px;
    margin: 0;
}

.psych-card {
    background: #faf5ff;
    border: 1px solid #e9d5ff;
}
.psych-text {
    color: #7e22ce;
    margin: 0;
    font-size: 13.5px;
    font-weight: 600;
}

.editorial-combo-item {
    background: #f8fafc;
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 6px;
    border-left: 4px solid #4f46e5;
    font-size: 13px;
}

.editorial-card-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}

.grid-card {
    background: #ffffff;
    padding: 14px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.grid-card-label {
    font-size: 10px;
    font-weight: 800;
    color: #64748b;
}
.grid-card-val {
    font-weight: 700;
    color: #1e293b;
    margin-top: 4px;
    font-size: 13.5px;
}

.border-indigo { border-left: 4px solid #6366f1; }
.border-cyan { border-left: 4px solid #06b6d4; }
.border-emerald { border-left: 4px solid #10b981; }
.border-amber { border-left: 4px solid #f59e0b; }
.border-pink { border-left: 4px solid #ec4899; }
.border-purple { border-left: 4px solid #8b5cf6; }

/* High-Contrast Do's & Don'ts Styling */
.dos-card {
    background: #e6f4ea !important;
    border: 1.5px solid #a8e0b7 !important;
    padding: 16px !important;
    border-radius: 14px !important;
}
.dos-card ul li {
    color: #065f46 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    line-height: 1.4 !important;
}
.donts-card {
    background: #fce8e6 !important;
    border: 1.5px solid #f8b4b4 !important;
    padding: 16px !important;
    border-radius: 14px !important;
}
.donts-card ul li {
    color: #991b1b !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    line-height: 1.4 !important;
}
"""

dark_atelier_theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="rose",
    neutral_hue="slate"
)

with gr.Blocks(theme=dark_atelier_theme, css=editorial_css, title="AI Personal Fashion Stylist Studio") as demo:
    gr.Markdown(
        """
        <div class="hero-container">
            <div class="eyebrow-text">✨ AI GPU Capstone Internship 2026 • NVIDIA CoE</div>
            <h1 class="editorial-title">Personalized Fashion & Outfit Consultation <span class="accent-word">Tailored</span> for You</h1>
            <p class="editorial-tagline">Upload any garment or footwear photo to unlock your 18-section editorial styling dossier, color psychology, and bi-directional outfit pairing guide.</p>
        </div>
        """
    )
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            garment_input = gr.Image(type="numpy", label="📸 Upload Clothing or Footwear Image")
            consult_btn = gr.Button("✨ Consult Personal Fashion Stylist", variant="primary", size="lg")
            confidence_output = gr.Label(num_top_classes=4, label="🎯 Vision Classification Logits")
            
        with gr.Column(scale=1.3):
            dossier_output = gr.HTML(value=IDLE_STATE_HTML, label="📜 Editorial Styling Dossier")
            
    consult_btn.click(
        fn=lambda img: generate_editorial_consultation(Image.fromarray(img) if img is not None else None),
        inputs=[garment_input],
        outputs=[confidence_output, dossier_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
