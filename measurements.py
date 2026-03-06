import numpy as np
import cv2
import math
import landmark_config as config

class FaceEvaluator:
    def __init__(self):
        self.idx = config.LANDMARK_INDICES
        self.rois = config.ROI_DEFINITIONS

    def calculate_features(self, img_bgr, landmarks_norm):
        """
        img_bgr: The image array (height, width, 3)
        landmarks_norm: List of normalized landmarks (x, y, z) from MediaPipe
        """
        if img_bgr is None or not landmarks_norm:
            return None

        h, w = img_bgr.shape[:2]
        
        # Convert landmarks to Pixel Coordinates for ROI processing
        lms_px = np.array([[lm.x * w, lm.y * h] for lm in landmarks_norm])

        # --- HELPER FUNCTIONS ---
        def dist(i1, i2): # Normalized Euclidean Distance (0-1 range approx)
            p1, p2 = landmarks_norm[i1], landmarks_norm[i2]
            return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

        def get_tilt(idx_in, idx_out):
            p_in, p_out = landmarks_norm[idx_in], landmarks_norm[idx_out]
            dy = (p_in.y - p_out.y) * h 
            dx = abs(p_out.x - p_in.x) * w
            return math.degrees(math.atan2(dy, dx))
        
        def get_edge_strength(indices):
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array([lms_px[i] for i in indices], np.int32)
            cv2.fillPoly(mask, [pts], 255)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            return cv2.mean(edges, mask=mask)[0]

        def get_roi_stats(indices):
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array([lms_px[i] for i in indices], np.int32)
            cv2.fillPoly(mask, [pts], 255)
            img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            mean, std = cv2.meanStdDev(img_lab, mask=mask)
            return np.mean(std), mean[0][0] # homogeneity, lightness

        def get_lum(indices):
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array([lms_px[i] for i in indices], np.int32)
            cv2.fillPoly(mask, [pts], 255)
            return cv2.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB), mask=mask)[0]

        def get_brow_angle(p_in, p_arch, p_out):
            a, b, c = lms_px[p_in], lms_px[p_arch], lms_px[p_out]
            ba = a - b
            bc = c - b
            denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
            if denom == 0: return 0
            cosine_angle = np.dot(ba, bc) / denom
            return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

        # --- FEATURES CALCULATION ---
        f = {}

        # GROUP 1: STRUCTURE
        mid_h = dist(self.idx['mid_face_top'], self.idx['mid_face_bottom'])
        low_h = dist(self.idx['mid_face_bottom'], self.idx['chin_bottom'])
        f['R1_Mid_Lower_Ratio'] = mid_h / low_h if low_h > 0 else 0

        cheek_w = dist(self.idx['cheek_left'], self.idx['cheek_right'])
        jaw_w = dist(self.idx['jaw_left'], self.idx['jaw_right'])
        f['R2_Cheek_Jaw_Ratio'] = cheek_w / jaw_w if jaw_w > 0 else 0

        face_h = dist(self.idx['forehead_top'], self.idx['chin_bottom'])
        f['R3_Face_Compactness'] = cheek_w / face_h if face_h > 0 else 0

        # GROUP 2: EYES
        f['E4_Canthal_Tilt'] = (get_tilt(self.idx['left_eye']['in'], self.idx['left_eye']['out']) + 
                                get_tilt(self.idx['right_eye']['in'], self.idx['right_eye']['out'])) / 2
        
        h_l = dist(self.idx['left_eye']['top'], self.idx['left_eye']['bot'])
        w_l = dist(self.idx['left_eye']['in'], self.idx['left_eye']['out'])
        h_r = dist(self.idx['right_eye']['top'], self.idx['right_eye']['bot'])
        w_r = dist(self.idx['right_eye']['in'], self.idx['right_eye']['out'])
        
        # Guard against division by zero
        if w_l > 0 and w_r > 0:
            f['E5_Eye_Aspect_Ratio'] = ((h_l/w_l) + (h_r/w_r)) / 2
        else:
            f['E5_Eye_Aspect_Ratio'] = 0

        icd = dist(self.idx['left_eye']['in'], self.idx['right_eye']['in'])
        avg_eye_w = (w_l + w_r) / 2
        f['E6_Interocular_Ratio'] = icd / avg_eye_w if avg_eye_w > 0 else 0

        # BROWS
        bh_l = dist(self.idx['left_eye']['top'], self.idx['left_brow']['arch_top'])
        bh_r = dist(self.idx['right_eye']['top'], self.idx['right_brow']['arch_top'])
        f['B7_Eyebrow_Height'] = ((bh_l + bh_r) / 2) / avg_eye_w

        angle_l = get_brow_angle(self.idx['left_brow']['head_top'], self.idx['left_brow']['arch_top'], self.idx['left_brow']['tail_top'])
        angle_r = get_brow_angle(self.idx['right_brow']['head_top'], self.idx['right_brow']['arch_top'], self.idx['right_brow']['tail_top'])
        f['B8_Eyebrow_Arch'] = (angle_l + angle_r) / 2

        lb = self.idx['left_brow']
        rb = self.idx['right_brow']
        thick_l = dist(lb['head_top'], lb['head_bot']) + dist(lb['arch_top'], lb['arch_bot']) + dist(lb['tail_top'], lb['tail_bot'])
        thick_r = dist(rb['head_top'], rb['head_bot']) + dist(rb['arch_top'], rb['arch_bot']) + dist(rb['tail_top'], rb['tail_bot'])
        avg_eye_h = (h_l + h_r) / 2
        f['B9_Eyebrow_Thickness'] = ((thick_l + thick_r) / 2) / avg_eye_h if avg_eye_h > 0 else 0

        # GROUP 3: NOSE
        nose_w = dist(self.idx['nose_wing_left'], self.idx['nose_wing_right'])
        f['N10_Nose_Width_Ratio'] = nose_w / icd if icd > 0 else 0
        
        nose_len = dist(self.idx['nose_root'], self.idx['nose_tip'])
        f['N11_Nose_Length_Ratio'] = nose_len / mid_h if mid_h > 0 else 0
        
        bridge_w = dist(self.idx['nose_bridge_left'], self.idx['nose_bridge_right'])
        f['N12_Nose_Bridge_Ratio'] = bridge_w / icd if icd > 0 else 0

        # GROUP 4: LIPS
        lip_h = dist(self.idx['lip_top'], self.idx['lip_bot'])
        mouth_w = dist(self.idx['mouth_left'], self.idx['mouth_right'])
        f['L13_Lip_Fullness'] = lip_h / mouth_w if mouth_w > 0 else 0
        
        up_thick = dist(self.idx['lip_top'], self.idx['upper_lip_inner'])
        low_thick = dist(self.idx['lower_lip_inner'], self.idx['lip_bot'])
        f['L14_Upper_Lower_Ratio'] = up_thick / low_thick if low_thick > 0 else 0
        f['L15_Mouth_Nose_Ratio'] = mouth_w / nose_w if nose_w > 0 else 0

        # GROUP 5: CHIN
        chin_h = dist(self.idx['lip_bot'], self.idx['chin_bottom'])
        f['J16_Chin_Compactness'] = chin_h / low_h if low_h > 0 else 0

        # GROUP 6: SKIN (UPDATED ORDER)
        h_l_skin, l_l_skin = get_roi_stats(self.rois['cheek_left'])
        h_r_skin, l_r_skin = get_roi_stats(self.rois['cheek_right'])
        
        f['S17_Skin_Homogeneity'] = (h_l_skin + h_r_skin) / 2

        f['S18_Skin_Lightness'] = (l_l_skin + l_r_skin) / 2


        mask_lip = np.zeros((h, w), dtype=np.uint8)
        pts_lip = np.array([lms_px[i] for i in self.rois['lips_outer']], np.int32)
        cv2.fillPoly(mask_lip, [pts_lip], 255)
        mean_lip, _ = cv2.meanStdDev(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB), mask=mask_lip)
        
        f['S19_Skin_Contrast'] = abs(f['S18_Skin_Lightness'] - mean_lip[0][0])

        # GROUP 7: AGING (UPDATED ORDER)
        philtrum = dist(self.idx['mid_face_bottom'], self.idx['lip_top'])
        f['A20_Philtrum_Ratio'] = philtrum / chin_h if chin_h > 0 else 0

        f['A21_Nasolabial_Fold'] = (get_edge_strength(self.rois['nasolabial_left']) + 
                                    get_edge_strength(self.rois['nasolabial_right'])) / 2

        f['A22_Crows_Feet'] = (get_edge_strength(self.rois['crows_feet_left']) + 
                               get_edge_strength(self.rois['crows_feet_right'])) / 2

        bag_l = get_lum(self.rois['eye_bags_left'])
        bag_r = get_lum(self.rois['eye_bags_right'])
        f['A23_Periorbital_Aging'] = ((l_l_skin - bag_l) + (l_r_skin - bag_r)) / 2

        # ⚖️ SYMMETRY (FULL 5-FACTOR RESTORED)
        # ---------------------------------------------------------
        
        # 1. Eyes (Width)
        sym_eye_w = abs(w_l - w_r) / avg_eye_w if avg_eye_w > 0 else 0
        
        # 2. Jaw (Shape/Length to Chin)
        d_jaw_l = dist(self.idx['jaw_left'], self.idx['chin_bottom'])
        d_jaw_r = dist(self.idx['jaw_right'], self.idx['chin_bottom'])
        sym_jaw = abs(d_jaw_l - d_jaw_r) / mid_h if mid_h > 0 else 0
        
        # 3. Cheek (Outer Width from Center Line)
        center_idx = self.idx['nose_root']
        d_cheek_l = dist(center_idx, self.idx['cheek_left'])
        d_cheek_r = dist(center_idx, self.idx['cheek_right'])
        
        avg_cheek_dist = (d_cheek_l + d_cheek_r) / 2
        sym_cheek_w = abs(d_cheek_l - d_cheek_r) / avg_cheek_dist if avg_cheek_dist > 0 else 0

        # 4. Cheek (Vertical Height) - ROBUST MULTI-POINT VERSION
        cb_l_indices = self.idx['cheek_bone_left']
        cb_r_indices = self.idx['cheek_bone_right']
        
        # Outer Eye Indices
        eye_out_l = self.idx['left_eye']['out']
        eye_out_r = self.idx['right_eye']['out']
        
        total_sym_score = 0
        
        for idx_l, idx_r in zip(cb_l_indices, cb_r_indices):
            h_l_c = dist(eye_out_l, idx_l)
            h_r_c = dist(eye_out_r, idx_r)
            
            avg_h_c = (h_l_c + h_r_c) / 2
            if avg_h_c > 0:
                total_sym_score += abs(h_l_c - h_r_c) / avg_h_c
        
        sym_cheek_h = total_sym_score / len(cb_l_indices) if len(cb_l_indices) > 0 else 0

        # 5. Mouth (Corner Height)
        nose_base_idx = self.idx['mid_face_bottom']
        h_mouth_l = dist(nose_base_idx, self.idx['mouth_left'])
        h_mouth_r = dist(nose_base_idx, self.idx['mouth_right'])
        
        avg_mouth_h = (h_mouth_l + h_mouth_r) / 2
        sym_mouth = abs(h_mouth_l - h_mouth_r) / avg_mouth_h if avg_mouth_h > 0 else 0
        
        # Average all 5 symmetry scores
        f['SYM_Symmetry_Index'] = (sym_eye_w + sym_jaw + sym_cheek_w + sym_cheek_h + sym_mouth) / 5

        return f