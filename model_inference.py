# model_inference.py (Modified for Debugging)

import cv2
import numpy as np
from PIL import Image
import io
import torch
from ultralytics import YOLO

class BlueChairDetector:
    def __init__(self, model_path='model/best.pt'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEBUG: Using device: {self.device}")

        self.model = self._load_model(model_path)

        if self.model:
            self.model.to(self.device)
            self.model.eval()
        else:
            print("CRITICAL: Model did not load successfully. Detection functions will fail.")
            # Consider raising an exception or more robust error handling for production

        self.target_class_name = "Blue lab Chair Detection - v3 2025-01-29 3-51pm" 
        
        self.blue_chair_class_id = -1
        if self.model and hasattr(self.model, 'names'):
            print(f"DEBUG: Model class names: {self.model.names}") # Print all detected class names
            for class_id, name in self.model.names.items():
                if name == self.target_class_name:
                    self.blue_chair_class_id = class_id
                    print(f"DEBUG: Found target class '{self.target_class_name}' with ID: {self.blue_chair_class_id}")
                    break
            if self.blue_chair_class_id == -1:
                print(f"WARNING: Could not find class ID for '{self.target_class_name}'. Please check model.names and your data.yaml.")
                if len(self.model.names) == 1:
                     self.blue_chair_class_id = list(self.model.names.keys())[0]
                     print(f"DEBUG: Defaulting to class ID {self.blue_chair_class_id} as fallback for single-class model.")
        else:
            print("WARNING: Model or its names attribute not available for class ID lookup during init.")


    def _load_model(self, model_path):
        print(f"DEBUG: Attempting to load YOLO11 model from: {model_path}")
        try:
            model = YOLO(model_path)
            print("DEBUG: YOLO11 model loaded successfully.")
            return model
        except Exception as e:
            print(f"ERROR: Error loading YOLO11 model from {model_path}: {e}")
            return None

    def _preprocess_image(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image. Is it a valid image file?")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"DEBUG: Image preprocessed. Shape: {img_rgb.shape}")
        return img_rgb

    def _postprocess_predictions(self, results, original_image):
        img_copy = original_image.copy()
        detected_blue_chairs_info = []

        print(f"DEBUG: Starting post-processing. Number of raw result objects: {len(results) if results else 0}")

        if results and len(results) > 0:
            for r in results:
                if r.boxes is None:
                    print("DEBUG: No boxes found in a result object.")
                    continue

                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy()

                print(f"DEBUG: Found {len(boxes)} raw detections in current result object.")

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    confidence = scores[i]
                    class_id = int(class_ids[i])
                    
                    current_class_name = self.model.names.get(class_id, 'Unknown')
                    print(f"DEBUG: Detection {i+1}: Class ID: {class_id} ({current_class_name}), Confidence: {confidence:.2f}, Box: [{x1},{y1},{x2},{y2}]")

                    # Check 1: Is it the target class?
                    if class_id == self.blue_chair_class_id:
                        print(f"DEBUG:   - Matches target class ('{self.target_class_name}').")
                        # Check 2: Does it meet confidence threshold?
                        if confidence > 0.4:
                            print(f"DEBUG:   - Meets confidence threshold ({confidence:.2f} > 0.4).")
                            chair_roi = original_image[y1:y2, x1:x2]
                            
                            if chair_roi.shape[0] == 0 or chair_roi.shape[1] == 0:
                                print(f"WARNING: ROI is empty for detection {i+1}. Skipping color analysis.")
                                continue

                            hsv_roi = cv2.cvtColor(chair_roi, cv2.COLOR_RGB2HSV)
                            
                            lower_blue = np.array([90, 50, 50]) # Tune these
                            upper_blue = np.array([130, 255, 255]) # Tune these
                            
                            blue_mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)
                            blue_pixel_count = cv2.countNonZero(blue_mask)
                            total_pixels_in_roi = chair_roi.shape[0] * chair_roi.shape[1]
                            
                            blue_percentage = 0
                            if total_pixels_in_roi > 0:
                                blue_percentage = (blue_pixel_count / total_pixels_in_roi) * 100
                            print(f"DEBUG:   - Blue percentage in ROI: {blue_percentage:.1f}% (Total pixels: {total_pixels_in_roi}, Blue pixels: {blue_pixel_count})")
                            
                            # Check 3: Does it meet blue percentage threshold?
                            if blue_percentage > 10: # Tune this
                                print(f"DEBUG:   - Meets blue percentage threshold ({blue_percentage:.1f}% > 10%). THIS IS A BLUE CHAIR!")
                                color = (0, 0, 255)
                                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                                text = f"{self.model.names.get(class_id, 'Unknown')}: {confidence:.2f} ({blue_percentage:.1f}% blue)"
                                cv2.putText(img_copy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                detected_blue_chairs_info.append({
                                    'box': [x1, y1, x2, y2],
                                    'confidence': float(confidence),
                                    'class_name': self.model.names.get(class_id, 'Unknown'),
                                    'blue_percentage': float(blue_percentage)
                                })
                            else:
                                print(f"DEBUG:   - Does NOT meet blue percentage threshold ({blue_percentage:.1f}% <= 10%).")
                        else:
                            print(f"DEBUG:   - Does NOT meet confidence threshold ({confidence:.2f} <= 0.4).")
                    else:
                        print(f"DEBUG:   - Does NOT match target class (ID: {class_id}, Name: '{current_class_name}'). Expected ID: {self.blue_chair_class_id}")
        else:
            print("DEBUG: No raw results from model inference.")

        print(f"DEBUG: Post-processing complete. Total blue chairs detected: {len(detected_blue_chairs_info)}")
        return img_copy, detected_blue_chairs_info

    def detect_blue_chairs(self, image_bytes):
        if self.model is None:
            return None, "Model not loaded. Cannot perform detection."
        
        try:
            original_image_rgb = self._preprocess_image(image_bytes)
        except ValueError as e:
            print(f"ERROR: Image preprocessing failed: {e}")
            return None, f"Image preprocessing error: {e}"

        print("DEBUG: Performing model inference...")
        try:
            # Running inference with the model
            results = self.model(original_image_rgb, stream=False, verbose=False, imgsz=640)
        except Exception as e:
            print(f"ERROR: Model inference failed: {e}")
            return None, f"Model inference error: {e}"

        processed_image, detections_info = self._postprocess_predictions(results, original_image_rgb)
        
        is_success, im_buf_arr = cv2.imencode(".jpg", cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        if not is_success:
            print("ERROR: Failed to encode processed image for display.")
            return None, "Failed to encode processed image for display."
        byte_im = im_buf_arr.tobytes()
        
        return byte_im, detections_info