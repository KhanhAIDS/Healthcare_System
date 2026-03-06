# =========================================================================
# 1. LANDMARK INDICES (The "Truth Source")
# =========================================================================
LANDMARK_INDICES = {
    # --- GROUP 1: STRUCTURE ---
    'mid_face_top': 8,       # Glabella (Brow ridge center)
    'mid_face_bottom': 2,    # Subnasale (Nose base)
    'chin_bottom': 152,      # Menton
    'forehead_top': 10,      # Approximate hairline/forehead top
    'cheek_left': 234, 'cheek_right': 454,  # Bizygomatic width (Face edge)
    'jaw_left': 172,   'jaw_right': 397,    # Bigonial width (Frontal view)

    # --- GROUP 2: EYES & BROWS ---
    'left_eye':  {'in': 133, 'out': 33,  'top': 159, 'bot': 145},
    'right_eye': {'in': 362, 'out': 263, 'top': 386, 'bot': 374},
    
    'left_brow': {
        'head_top': 107, 'head_bot': 55,
        'arch_top': 105, 'arch_bot': 52,
        'tail_top': 70,  'tail_bot': 46
    },
    'right_brow': {
        'head_top': 336, 'head_bot': 285,
        'arch_top': 334, 'arch_bot': 282,
        'tail_top': 300, 'tail_bot': 276
    },

    # --- GROUP 3: NOSE ---
    'nose_wing_left': 129, 'nose_wing_right': 358,
    'nose_root': 168,      # Nasion (between eyes) - for Length
    'nose_tip': 1,         # Pronasale
    'nose_bridge_left': 196, 'nose_bridge_right': 419, # Root width

    # --- GROUP 4: LIPS ---
    'lip_top': 0,          # Cupid's bow
    'lip_bot': 17,         # Lower lip bottom
    'mouth_left': 61, 'mouth_right': 291,
    'upper_lip_inner': 13, 
    'lower_lip_inner': 14,

    # High cheekbone probabilities
    'cheek_bone_left': [117, 101, 50],
    'cheek_bone_right': [346, 330, 280] 
}

# =========================================================================
# 2. ROI POLYGONS (For Texture Analysis)
# =========================================================================
ROI_DEFINITIONS = {
    # S17/19 Skin Quality (Cheek Centers)
    'cheek_left':  [116, 121, 142, 207, 123],
    'cheek_right': [345, 350, 371, 427, 352],
    
    # S18 Skin Contrast (Outer Lip Ring)
    'lips_outer': [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 
                   17, 84, 181, 91, 146, 61, 185, 40, 39, 37],

    # A21 Nasolabial Folds (Triangles)
    'nasolabial_left':  [129, 216, 207, 100],
    'nasolabial_right': [358, 436, 427, 329],

    # A22 Periorbital Aging
    'eye_bags_left':  [130, 24, 22, 244, 121, 119, 111],
    'eye_bags_right': [359, 254, 252, 464, 350, 348, 340],
    'crows_feet_left':  [33, 156, 116],
    'crows_feet_right': [263, 383, 345],
    
    # Hair Occlusion Zones (Area above eyebrows) ---
    'hair_check_brow_left':  [70, 63, 105, 66, 107, 108, 69, 104, 68, 71],
    'hair_check_brow_right': [300, 293, 334, 296, 336, 337, 299, 333, 298, 301],
}