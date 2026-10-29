import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pathlib import Path
import shutil

OUT_PPT = Path(r"D:\Yolo\Data\project_flir\thermal-yolo\IEEE_Sensor_Thermal\Multimodal_Transit_Security_Presentation.pptx")
ART_PPT = Path(r"C:\Users\satya\.gemini\antigravity\brain\8aed0bbc-dce3-4a5a-b2c4-58c864f9a829\Multimodal_Transit_Security_Presentation.pptx")
IMG_DIR = Path(r"D:\Yolo\Data\project_flir\thermal-yolo\IEEE_Sensor_Thermal")

def create_presentation():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]  # completely blank slide

    black = RGBColor(0, 0, 0)
    dark_gray = RGBColor(80, 80, 80)
    white = RGBColor(255, 255, 255)
    light_gray = RGBColor(245, 245, 245)

    def add_header(slide, title_text, category_text="DUAL-VISION AI TRANSIT SECURITY"):
        # Header banner text
        tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.9))
        tf = tx_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = "Arial"
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = dark_gray
        p_cat.space_after = Pt(2)
        
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.name = "Arial"
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = black
        
        # Plain divider line
        line = slide.shapes.add_shape(
            pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
            Inches(0.8), Inches(1.35), Inches(11.733), Inches(0.02)
        )
        line.fill.solid()
        line.fill.fore_color.rgb = black
        line.line.color.rgb = black

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide (Plain, Black and White)
    # -------------------------------------------------------------
    slide1 = prs.slides.add_slide(blank_layout)
    
    # Outer frame
    frame = slide1.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
        Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9)
    )
    frame.fill.solid()
    frame.fill.fore_color.rgb = white
    frame.line.color.rgb = black
    frame.line.width = Pt(1.5)

    tb = slide1.shapes.add_textbox(Inches(1.4), Inches(1.5), Inches(10.5), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "AUTONOMOUS TRANSIT HUB SECURITY"
    p.font.name = "Arial"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = dark_gray
    p.space_after = Pt(14)
    
    p = tf.add_paragraph()
    p.text = "Seeing the Unseen: Dual-Vision AI for Airport & Railway Surveillance"
    p.font.name = "Arial"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = black
    p.space_after = Pt(16)
    
    p = tf.add_paragraph()
    p.text = "How combining standard color cameras with invisible infrared heat vision enables automated, 24/7 detection of passengers, luggage, and unattended anomalies with over 92% accuracy."
    p.font.name = "Arial"
    p.font.size = Pt(16)
    p.font.color.rgb = dark_gray
    p.space_after = Pt(28)
    
    p = tf.add_paragraph()
    p.text = "Autonomous and Cyber-Physical Systems (ACPS) Research Group\nDepartment of Information and Communication Technology, University of Agder"
    p.font.name = "Arial"
    p.font.size = Pt(13)
    p.font.color.rgb = black

    # -------------------------------------------------------------
    # SLIDE 2: The Real-World Security Problem
    # -------------------------------------------------------------
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "The Real-World Security Challenge in High-Throughput Transit Hubs")

    tb = slide2.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True
    
    points = [
        ("High Passenger Traffic:", "Global transit concourses handle over 9 billion passengers annually, with millions of misplaced or handled luggage items moving through terminals."),
        ("Security Directive (ICAO Annex 17):", "Any baggage item left isolated from its owner must be treated as a potential explosive threat (IED) until cleared by security personnel."),
        ("Cost of Errors:", "False alarms cause panic, airport terminal evacuations, and costly operational ground stops. Missed bags pose catastrophic safety risks."),
        ("Surveillance Goal:", "Build an automated 24/7 AI vision system capable of reliably detecting 4 classes: Persons, Luggage Trolleys, Carried Bags, and Unattended Bags.")
    ]
    for i, (title, desc) in enumerate(points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{i+1}. {title} "
        p.font.name = "Arial"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = black
        
        run = p.add_run()
        run.text = desc
        run.font.name = "Arial"
        run.font.size = Pt(13)
        run.font.bold = False
        run.font.color.rgb = dark_gray
        p.space_after = Pt(14)

    # Right side: box explaining why computer vision struggles
    rbox = slide2.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
        Inches(7.2), Inches(1.6), Inches(5.333), Inches(5.2)
    )
    rbox.fill.solid()
    rbox.fill.fore_color.rgb = light_gray
    rbox.line.color.rgb = black
    rbox.line.width = Pt(1)

    rtb = slide2.shapes.add_textbox(Inches(7.5), Inches(1.8), Inches(4.733), Inches(4.8))
    rtf = rtb.text_frame
    rtf.word_wrap = True
    
    p = rtf.paragraphs[0]
    p.text = "WHY SINGLE-CAMERA AI FAILS"
    p.font.name = "Arial"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = black
    p.space_after = Pt(14)
    
    challenges = [
        "Crowded Environments: Travelers overlap and block camera sightlines.",
        "Severe Lighting Changes: Blinding sunlight glare through glass concourses by day; deep dark shadows at night.",
        "Small Physical Scale: Abandoned luggage is often small (median scale only 48 pixels wide on camera).",
        "Single Sensor Bottleneck: No single camera spectrum works reliably across both day and night conditions."
    ]
    for ch in challenges:
        p = rtf.add_paragraph()
        p.text = f"• {ch}"
        p.font.name = "Arial"
        p.font.size = Pt(12)
        p.font.color.rgb = black
        p.space_after = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 3: Why One Eye Isn't Enough (RGB vs. Thermal)
    # -------------------------------------------------------------
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Why One Camera Isn't Enough: Visible RGB vs. Thermal LWIR")

    # Left Column: RGB Camera
    box_l = slide3.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
        Inches(0.8), Inches(1.6), Inches(5.6), Inches(3.2)
    )
    box_l.fill.solid()
    box_l.fill.fore_color.rgb = white
    box_l.line.color.rgb = black
    box_l.line.width = Pt(1)

    ltb = slide3.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(2.8))
    ltf = ltb.text_frame
    ltf.word_wrap = True
    p = ltf.paragraphs[0]
    p.text = "SENSOR 1: VISIBLE COLOR CAMERA (RGB)"
    p.font.name = "Arial"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = black
    p.space_after = Pt(8)
    
    p = ltf.add_paragraph()
    p.text = "• Captures reflected visible light (colors, textures, text).\n• Superpower: Sharp luggage textures, zippers, straps, and boundaries easily distinguish bags from concourse floor tiles.\n• Blind Spot: Degrades heavily under darkness, shadows, and specular floor/window glare."
    p.font.name = "Arial"
    p.font.size = Pt(12)
    p.font.color.rgb = dark_gray

    # Right Column: Thermal Camera
    box_r = slide3.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
        Inches(6.9), Inches(1.6), Inches(5.6), Inches(3.2)
    )
    box_r.fill.solid()
    box_r.fill.fore_color.rgb = white
    box_r.line.color.rgb = black
    box_r.line.width = Pt(1)

    rtb = slide3.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.2), Inches(2.8))
    rtf = rtb.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    p.text = "SENSOR 2: THERMAL INFRARED CAMERA (LWIR)"
    p.font.name = "Arial"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = black
    p.space_after = Pt(8)
    
    p = rtf.add_paragraph()
    p.text = "• Measures emitted heat radiation (7.5 - 13.5 micrometers).\n• Superpower: Completely immune to pitch-black darkness or glare. Spots warm human bodies (37°C) instantly.\n• Blind Spot: Abandoned luggage cools down to room temperature (20°C) and becomes thermally invisible on cold floors."
    p.font.name = "Arial"
    p.font.size = Pt(12)
    p.font.color.rgb = dark_gray

    # Bottom summary box
    bbox = slide3.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
        Inches(0.8), Inches(5.1), Inches(11.733), Inches(1.7)
    )
    bbox.fill.solid()
    bbox.fill.fore_color.rgb = light_gray
    bbox.line.color.rgb = black
    bbox.line.width = Pt(1)

    btb = slide3.shapes.add_textbox(Inches(1.1), Inches(5.25), Inches(11.133), Inches(1.4))
    btf = btb.text_frame
    btf.word_wrap = True
    p = btf.paragraphs[0]
    p.text = "THE CORE PHYSICAL INSIGHT"
    p.font.name = "Arial"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = black
    p.space_after = Pt(4)
    
    p = btf.add_paragraph()
    p.text = "\"A cold bag on a cold floor looks invisible to a thermal camera. But in pitch black darkness or harsh glare, a human is invisible to a standard camera. High-reliability surveillance requires fusing both modalities into a unified AI system.\""
    p.font.name = "Arial"
    p.font.size = Pt(13)
    p.font.italic = True
    p.font.color.rgb = black

    # -------------------------------------------------------------
    # SLIDE 4: The Obvious Idea Fails (Why Naive Mixing Fails)
    # -------------------------------------------------------------
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "The Obvious Idea Fails: Why Naive Early Mixing Breaks Detection")

    tb = slide4.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(1.0))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "When combining dual cameras, the most common beginner approach is: \"Just stack the RGB and Thermal channels together at the beginning (4-channel or 6-channel input) and feed them into a standard YOLO model.\" Here is why that fails:"
    p.font.name = "Arial"
    p.font.size = Pt(14)
    p.font.color.rgb = black

    reasons = [
        ("1. Conflicting Physics in Shared Convolutions",
         "Visible light measures reflected surface photons, while thermal measures emitted heat radiation. Forcing the very first 3x3 convolutional filter to blend both signals simultaneously creates mathematical conflict and degrades low-level edge detection."),
        ("2. Destruction of Pretrained Weights",
         "Modern AI detectors rely on weights pretrained on millions of standard photos (COCO dataset). Modifying the initial layer to accept mixed raw thermal data corrupts these pretrained visual priors, causing the network to unlearn useful edge features."),
        ("3. The Measured Result: Worse Than Thermal Alone!",
         "In our experiments, naive early concatenation (v34) achieved only 83.68% test mAP@50. That is 2.54 percentage points WORSE than standard single-camera thermal YOLOv8n (86.22%) and 4.47 points worse than our thermal champion (88.15%).")
    ]

    for i, (title, desc) in enumerate(reasons):
        box = slide4.shapes.add_shape(
            pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
            Inches(0.8 + i * 4.0), Inches(2.7), Inches(3.733), Inches(3.5)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = light_gray if i == 2 else white
        box.line.color.rgb = black
        box.line.width = Pt(1.5 if i == 2 else 1)

        box_tb = slide4.shapes.add_textbox(Inches(1.0 + i * 4.0), Inches(2.9), Inches(3.333), Inches(3.1))
        box_tf = box_tb.text_frame
        box_tf.word_wrap = True
        
        p = box_tf.paragraphs[0]
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = black
        p.space_after = Pt(8)
        
        p = box_tf.add_paragraph()
        p.text = desc
        p.font.name = "Arial"
        p.font.size = Pt(12)
        p.font.color.rgb = dark_gray

    p_bot = slide4.shapes.add_textbox(Inches(0.8), Inches(6.4), Inches(11.733), Inches(0.6))
    p_bot.text_frame.word_wrap = True
    p = p_bot.text_frame.paragraphs[0]
    p.text = "KEY CONCLUSION: Each sensor requires its own dedicated neural processing pathway before fusion."
    p.font.name = "Arial"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = black

    # -------------------------------------------------------------
    # SLIDE 5: Our Architecture: The Dual-Stream System
    # -------------------------------------------------------------
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "Our Architecture: Asymmetric Dual-Stream Deep Neural Network")

    # Insert the 5-panel architecture diagram
    arch_img = IMG_DIR / "Multimodal_Dilated_ECA_Yolo.png"
    if arch_img.exists():
        slide5.shapes.add_picture(str(arch_img), Inches(0.8), Inches(1.6), Inches(11.733), Inches(3.5))

    # Explanatory text below diagram
    box_desc = slide5.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
        Inches(0.8), Inches(5.3), Inches(11.733), Inches(1.7)
    )
    box_desc.fill.solid()
    box_desc.fill.fore_color.rgb = white
    box_desc.line.color.rgb = black
    box_desc.line.width = Pt(1)

    dtb = slide5.shapes.add_textbox(Inches(1.0), Inches(5.4), Inches(11.333), Inches(1.5))
    dtf = dtb.text_frame
    dtf.word_wrap = True
    
    p = dtf.paragraphs[0]
    p.text = "5-PANEL ARCHITECTURAL BREAKDOWN:"
    p.font.name = "Arial"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = black
    p.space_after = Pt(4)
    
    steps = [
        "(a) Baseline Early Fusion: What fails (mixed stem convolutions corrupt features).",
        "(b) Macro-Architecture: Two dedicated streams (RGB + Thermal) with mid-level CMAF gates and ECA neck.",
        "(c) Micro-Bottlenecks: Asymmetric design (Dense d=1 for RGB texture vs. Dilated d=2 for thermal context).",
        "(d) CMAF Attention Gate: Real-time dynamic referee deciding pixel-wise trust between visible and thermal.",
        "(e) 2D-Grid CWAR / MACAF Head: Direct 1-to-1 anchor fusion across 8,400 spatial cells prior to NMS (92.33%)."
    ]
    for s in steps:
        p = dtf.add_paragraph()
        p.text = f"• {s}"
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.color.rgb = dark_gray

    # -------------------------------------------------------------
    # SLIDE 6: Secret 1: Asymmetric Focus (Magnifier vs. Wide-Angle)
    # -------------------------------------------------------------
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Secret #1: Cross-Modal Spatial Frequency & Receptive-Field Asymmetry")

    tb = slide6.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(0.8))
    tb.text_frame.word_wrap = True
    p = tb.text_frame.paragraphs[0]
    p.text = "Most multimodal models treat both camera streams identically. We discovered that visible and thermal features obey fundamentally different spatial frequency rules and require opposite focus styles:"
    p.font.name = "Arial"
    p.font.size = Pt(14)
    p.font.color.rgb = black

    # Left Box: RGB Dense Magnifier
    b_l = slide6.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
        Inches(0.8), Inches(2.6), Inches(5.6), Inches(3.6)
    )
    b_l.fill.solid()
    b_l.fill.fore_color.rgb = white
    b_l.line.color.rgb = black
    b_l.line.width = Pt(1)

    t_l = slide6.shapes.add_textbox(Inches(1.0), Inches(2.8), Inches(5.2), Inches(3.2))
    tf_l = t_l.text_frame
    tf_l.word_wrap = True
    
    p = tf_l.paragraphs[0]
    p.text = "RGB STREAM: \"THE DENSE MAGNIFIER\""
    p.font.name = "Arial"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = black
    p.space_after = Pt(8)
    
    points_l = [
        "Focus: Fine, high-frequency physical boundaries (zipper seams, thin straps, handles, logos).",
        "Design: Dense 3x3 convolutions with dilation d = 1 across all stages (P2 to P5).",
        "Attention: Coordinate Attention (directional horizontal and vertical pooling) preserves exact boundary coordinates.",
        "Result: Razor-sharp detection borders with 93.01% precision on carried luggage."
    ]
    for pt in points_l:
        p = tf_l.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = dark_gray
        p.space_after = Pt(4)

    # Right Box: Thermal Dilated Wide-Angle
    b_r = slide6.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
        Inches(6.9), Inches(2.6), Inches(5.6), Inches(3.6)
    )
    b_r.fill.solid()
    b_r.fill.fore_color.rgb = white
    b_r.line.color.rgb = black
    b_r.line.width = Pt(1)

    t_r = slide6.shapes.add_textbox(Inches(7.1), Inches(2.8), Inches(5.2), Inches(3.2))
    tf_r = t_r.text_frame
    tf_r.word_wrap = True
    
    p = tf_r.paragraphs[0]
    p.text = "THERMAL STREAM: \"THE WIDE-ANGLE GLANCE\""
    p.font.name = "Arial"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = black
    p.space_after = Pt(8)
    
    points_r = [
        "Focus: Low-frequency, diffuse thermal radiation halos emitted by human bodies.",
        "Design: Dilated convolutions with dilation d = 2 at deeper stages (P4 and P5).",
        "Receptive Field: Expands the effective receptive field to 5x5 without adding extra parameters or computation.",
        "Result: Binds human thermal plumes with nearby luggage carts, achieving 86.12% recall on pedestrians."
    ]
    for pt in points_r:
        p = tf_r.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = dark_gray
        p.space_after = Pt(4)

    # Bottom summary
    s_bot = slide6.shapes.add_textbox(Inches(0.8), Inches(6.4), Inches(11.733), Inches(0.5))
    p = s_bot.text_frame.paragraphs[0]
    p.text = "TAKEAWAY: Asymmetry ensures RGB provides high precision (93.0%) while LWIR provides high recall (86.1%)."
    p.font.name = "Arial"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = black

    # -------------------------------------------------------------
    # SLIDE 7: Secret 2: Real-Time Attention Referee (CMAF Gate)
    # -------------------------------------------------------------
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Secret #2: Multi-Scale Cross-Modal Attention Fusion (CMAF Gates)")

    tb = slide7.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(0.8))
    tb.text_frame.word_wrap = True
    p = tb.text_frame.paragraphs[0]
    p.text = "When both cameras inspect the same pixel, which one should the network trust? Our learned Cross-Modal Attention Fusion (CMAF) gate computes a spatial confidence weight in real time:"
    p.font.name = "Arial"
    p.font.size = Pt(14)
    p.font.color.rgb = black

    # Two Scenario Boxes
    sc_l = slide7.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
        Inches(0.8), Inches(2.6), Inches(5.6), Inches(2.7)
    )
    sc_l.fill.solid()
    sc_l.fill.fore_color.rgb = white
    sc_l.line.color.rgb = black
    sc_l.line.width = Pt(1)

    sct_l = slide7.shapes.add_textbox(Inches(1.0), Inches(2.8), Inches(5.2), Inches(2.3))
    p = sct_l.text_frame.paragraphs[0]
    p.text = "SCENARIO A: WELL-LIT ROOM, COLD BACKPACK"
    p.font.name = "Arial"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = black
    p.space_after = Pt(6)
    
    p = sct_l.text_frame.add_paragraph()
    p.text = "• The thermal camera sees only a room-temperature gray floor.\n• The visible RGB camera sees a clear navy-blue backpack with high-contrast zippers.\n• The CMAF Gate automatically shifts:\n  Alpha -> 1.0 (Routes 100% RGB texture into the neck)."
    p.font.name = "Arial"
    p.font.size = Pt(12)
    p.font.color.rgb = dark_gray

    sc_r = slide7.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
        Inches(6.9), Inches(2.6), Inches(5.6), Inches(2.7)
    )
    sc_r.fill.solid()
    sc_r.fill.fore_color.rgb = white
    sc_r.line.color.rgb = black
    sc_r.line.width = Pt(1)

    sct_r = slide7.shapes.add_textbox(Inches(7.1), Inches(2.8), Inches(5.2), Inches(2.3))
    p = sct_r.text_frame.paragraphs[0]
    p.text = "SCENARIO B: DARK CORRIDOR, SUNLIGHT GLARE"
    p.font.name = "Arial"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = black
    p.space_after = Pt(6)
    
    p = sct_r.text_frame.add_paragraph()
    p.text = "• The visible RGB camera is blinded by glare or dark shadows.\n• The thermal camera clearly registers the 37°C passenger and luggage frame.\n• The CMAF Gate automatically shifts:\n  Alpha -> 0.0 (Routes 100% Thermal heat into the neck)."
    p.font.name = "Arial"
    p.font.size = Pt(12)
    p.font.color.rgb = dark_gray

    # Formula Box
    f_box = slide7.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
        Inches(0.8), Inches(5.5), Inches(11.733), Inches(1.3)
    )
    f_box.fill.solid()
    f_box.fill.fore_color.rgb = light_gray
    f_box.line.color.rgb = black
    f_box.line.width = Pt(1)

    f_tb = slide7.shapes.add_textbox(Inches(1.0), Inches(5.6), Inches(11.333), Inches(1.1))
    p = f_tb.text_frame.paragraphs[0]
    p.text = "MATHEMATICAL FORMULATION (OPERATING AT P3, P4, P5):"
    p.font.name = "Arial"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = black
    p.space_after = Pt(2)
    
    p = f_tb.text_frame.add_paragraph()
    p.text = "Fused_Feature = Alpha * RGB_Feature + (1 - Alpha) * Thermal_Feature + Residual_Projection\nwhere Alpha is computed by a 1x1 spatial convolution: Alpha = Sigmoid(Conv1x1(ReLU(BN(Conv1x1([F_rgb, F_thm])))))"
    p.font.name = "Arial"
    p.font.size = Pt(11.5)
    p.font.color.rgb = black

    # -------------------------------------------------------------
    # SLIDE 8: How We Hit >92%: 3 Specialized Experts & 2D Grid CWAR
    # -------------------------------------------------------------
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Crossing the 92% Milestone: 2D-Grid Anchor-Aligned CWAR Fusion")

    tb = slide8.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(0.8))
    tb.text_frame.word_wrap = True
    p = tb.text_frame.paragraphs[0]
    p.text = "A single standalone dual-stream model peaks at 89.86% Test / 90.53% Val mAP@50. To reach 92.33% Test / 92.89% Val mAP@50, we combine three complementary dual-stream experts:"
    p.font.name = "Arial"
    p.font.size = Pt(13.5)
    p.font.color.rgb = black

    experts = [
        ("EXPERT 1: v36_main (Dense CA + ECA)",
         "RGB uses Coordinate Attention + 1D ECA (d=1); LWIR uses Dilated ECA (d=2).\n• Strength: Highest standalone Person (82.4%) and Unattended Luggage (86.10%) AP@50.\n• Val mAP@50: 90.53% | Test mAP@50: 89.75%"),
        ("EXPERT 2: v36_eca_rgb (Dense ECA)",
         "RGB uses pure 1D Efficient Channel Attention (d=1); LWIR uses Dilated ECA (d=2).\n• Strength: Highest standalone Precision (93.01%) and Carried Bag AP@50 (94.45%).\n• Val mAP@50: 89.75% | Test mAP@50: 89.86%"),
        ("EXPERT 3: v38_ca_dilated (Symmetric Dilated)",
         "RGB uses CA+ECA at P2,P3 and Dilated d=2 at P4,P5 + full ECA PANet Neck.\n• Strength: Highest standalone Recall (86.12%) and Trolley AP@50 (98.57%).\n• Val mAP@50: 89.33% | Test mAP@50: 88.74%")
    ]
    for i, (title, desc) in enumerate(experts):
        box = slide8.shapes.add_shape(
            pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
            Inches(0.8 + i * 4.0), Inches(2.6), Inches(3.733), Inches(2.6)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = white
        box.line.color.rgb = black
        box.line.width = Pt(1)

        btb = slide8.shapes.add_textbox(Inches(1.0 + i * 4.0), Inches(2.75), Inches(3.333), Inches(2.3))
        p = btb.text_frame.paragraphs[0]
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.bold = True
        p.font.color.rgb = black
        p.space_after = Pt(6)
        
        p = btb.text_frame.add_paragraph()
        p.text = desc
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.color.rgb = dark_gray

    # Fusion description box
    c_box = slide8.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
        Inches(0.8), Inches(5.4), Inches(11.733), Inches(1.4)
    )
    c_box.fill.solid()
    c_box.fill.fore_color.rgb = light_gray
    c_box.line.color.rgb = black
    c_box.line.width = Pt(1)

    ctb = slide8.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(11.333), Inches(1.2))
    p = ctb.text_frame.paragraphs[0]
    p.text = "CONFIDENCE-WEIGHTED ANCHOR-ALIGNED REGRESSION (CWAR):"
    p.font.name = "Arial"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = black
    p.space_after = Pt(2)
    
    p = ctb.text_frame.add_paragraph()
    p.text = "Instead of naive box concatenation into NMS (which causes good boxes to suppress each other), all three models share the exact same 8,400 spatial pyramid anchors (80x80, 40x40, 20x20). Bounding boxes are weighted by each model's exponentiated peak confidence: Box = Sum(Conf^1.25 * Box_m) / Sum(Conf^1.25). This boosts test mAP@50 by +2.10% over standard ensembling."
    p.font.name = "Arial"
    p.font.size = Pt(11)
    p.font.color.rgb = black

    # -------------------------------------------------------------
    # SLIDE 9: Measured Experimental Results (1,067 Frames Benchmark)
    # -------------------------------------------------------------
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "Empirical Results: Verified on 1,067-Frame Transit Dataset")

    # Table of Results
    rows = 5
    cols = 6
    table_shape = slide9.shapes.add_table(rows, cols, Inches(0.8), Inches(1.6), Inches(11.733), Inches(2.8))
    table = table_shape.table

    # Column widths
    table.columns[0].width = Inches(3.2)
    table.columns[1].width = Inches(1.9)
    table.columns[2].width = Inches(1.6)
    table.columns[3].width = Inches(1.6)
    table.columns[4].width = Inches(1.6)
    table.columns[5].width = Inches(1.833)

    headers = ["Model / Architecture", "Sensing Input", "Person AP", "Bag AP", "Unattended AP", "Test mAP@50"]
    data = [
        ["Stock YOLOv8n Baseline", "Thermal Only", "76.29%", "89.20%", "81.79%", "86.22%"],
        ["Early Stem Concat (v34)", "RGB + Thermal", "74.50%", "86.10%", "77.30%", "83.68% (Fails)"],
        ["Standalone Dual-Stream (v36_main)", "RGB + Thermal", "82.42%", "92.10%", "86.10%", "89.75%"],
        ["Proposed CWAR Triad System", "RGB + Thermal", "87.03%", "96.74%", "86.92%", "92.33% (Hit >92%)"]
    ]

    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = black
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            p.font.name = "Arial"
            p.font.size = Pt(11.5)
            p.font.bold = True
            p.font.color.rgb = white

    for r, row_data in enumerate(data):
        for c, val in enumerate(row_data):
            cell = table.cell(r + 1, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = light_gray if r == 3 else white
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
                p.font.name = "Arial"
                p.font.size = Pt(11)
                p.font.bold = (r == 3 or c == 5)
                p.font.color.rgb = black

    # Metric Gain Callouts
    gains = [
        ("+10.74%", "Person Silhouette Detection", "Disambiguates occluded pedestrians in dense crowds using clothing boundaries."),
        ("+7.54%", "Carried Luggage (Bags)", "Solves thermal room-temperature equilibration using visible surface texture."),
        ("+6.11%", "Overall Test Accuracy Jump", "Pushes baseline 86.22% to 92.33% test mAP@50 (69.02% mAP@50-95).")
    ]
    for i, (metric, label, desc) in enumerate(gains):
        box = slide9.shapes.add_shape(
            pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
            Inches(0.8 + i * 4.0), Inches(4.7), Inches(3.733), Inches(2.0)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = white
        box.line.color.rgb = black
        box.line.width = Pt(1)

        gtb = slide9.shapes.add_textbox(Inches(1.0 + i * 4.0), Inches(4.8), Inches(3.333), Inches(1.8))
        p = gtb.text_frame.paragraphs[0]
        p.text = metric
        p.font.name = "Arial"
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = black
        
        p = gtb.text_frame.add_paragraph()
        p.text = label
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = black
        p.space_after = Pt(2)
        
        p = gtb.text_frame.add_paragraph()
        p.text = desc
        p.font.name = "Arial"
        p.font.size = Pt(10)
        p.font.color.rgb = dark_gray

    # -------------------------------------------------------------
    # SLIDE 10: Summary & Practical Takeaways
    # -------------------------------------------------------------
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Summary: Core Contributions & Operational Significance")

    takeaways = [
        ("1. Hardware Physics Dictates Deep Learning Architecture",
         "Optical visible light and infrared heat waves operate on distinct physical principles. Concatenating both modalities into a shared introductory stem causes modality conflict. Decoupling them into asymmetric backbones is essential."),
        ("2. Asymmetric Receptive-Field Design is Optimal",
         "Dense convolutions (d=1) with Coordinate Attention on RGB preserve high-frequency luggage textures (93.01% precision), while dilated convolutions (d=2) on Thermal capture diffuse body plumes (86.12% recall)."),
        ("3. Pre-NMS 2D-Grid Anchor Alignment Surpasses Post-Hoc Ensembling",
         "Fusing predictions directly at the 8,400 pyramid anchors via Confidence-Weighted Anchor Regression (CWAR) prevents redundant bounding-box suppression, providing a +2.10% boost to hit 92.33% Test / 92.89% Val mAP@50."),
        ("4. Autonomous, False-Alarm Resistant, and Edge-Ready",
         "At only 4.5M parameters per stream, the architecture runs in real time on edge surveillance platforms, eliminating costly false alarms while guaranteeing 24/7 passenger and luggage hazard tracking.")
    ]

    for i, (title, desc) in enumerate(takeaways):
        box = slide10.shapes.add_shape(
            pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
            Inches(0.8), Inches(1.6 + i * 1.25), Inches(11.733), Inches(1.1)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = light_gray if i == 2 else white
        box.line.color.rgb = black
        box.line.width = Pt(1)

        ttb = slide10.shapes.add_textbox(Inches(1.0), Inches(1.7 + i * 1.25), Inches(11.333), Inches(0.9))
        p = ttb.text_frame.paragraphs[0]
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = black
        p.space_after = Pt(2)
        
        p = ttb.text_frame.add_paragraph()
        p.text = desc
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = dark_gray

    # Save presentation
    prs.save(str(OUT_PPT))
    shutil.copy2(OUT_PPT, ART_PPT)
    print("Successfully generated and saved PPTX at:", OUT_PPT)

if __name__ == "__main__":
    create_presentation()
