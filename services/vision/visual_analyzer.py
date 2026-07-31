import os
import json
import cv2
import numpy as np
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image


class TemporalJerseyClassifier(nn.Module):
    """Reconstructs the PyTorch architecture matching prxkc/jersey-number-recognition"""
    def __init__(self):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Linear(1280, 256)
        
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.tens_head = nn.Linear(512, 10)
        self.units_head = nn.Linear(512, 10)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        
        feat = self.features(x)
        feat = self.pool(feat).flatten(1)
        feat = self.projector(feat)
        
        feat = feat.view(b, t, -1)
        lstm_out, _ = self.lstm(feat)
        
        rep = torch.cat([lstm_out[:, -1, :128], lstm_out[:, 0, 128:]], dim=1)
        
        tens_logits = self.tens_head(rep)
        units_logits = self.units_head(rep)
        return tens_logits, units_logits


class VisualAnalyzer:
    """Handles pixel-level feature extraction (temporal jersey recognition, color classification) 
    and resolves player metadata using match profiling data.
    """

    def __init__(self, reader=None, roster_path: str = None, profiling_path: str = None):
        self.reader = reader
        
        # macOS device selection (MPS for Apple Silicon, fallback to CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🚀 Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            print("💻 Using CPU")

        self.number_model = None
        
        # Buffer to keep track of historical crop sequences per player track ID for temporal smoothing
        self.track_crop_buffers: Dict[int, List[np.ndarray]] = {}

        # Preprocessing required by EfficientNet-B0
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load optional individual roster path if provided
        if roster_path and os.path.exists(roster_path):
            try:
                with open(roster_path, 'r', encoding='utf-8') as f:
                    pass
            except Exception as e:
                print(f"⚠️ Warning: Could not load roster path {roster_path}: {e}")

        # Initialize team rosters dictionary from profiling JSON
        self.rosters = self._load_profiling_data(profiling_path) if profiling_path else {}

        # Load the local temporal model weights from weights/jersey_number_recon
        model_path = "weights/jersey_number_recon/best_temporal.pt"
        if not os.path.exists(model_path):
            model_path = "weights/jersey_number_recon.pt"

        try:
            if os.path.exists(model_path):
                print(f"📦 Loading temporal jersey model from local path: {model_path}")
                self.number_model = TemporalJerseyClassifier().to(self.device)
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.number_model.load_state_dict(checkpoint["state_dict"], strict=False)
                elif isinstance(checkpoint, dict):
                    self.number_model.load_state_dict(checkpoint, strict=False)
                else:
                    self.number_model = checkpoint.to(self.device)
                    
                self.number_model.eval()
                print("✅ Successfully loaded local temporal jersey model!")
            else:
                print("⚠️ Warning: Model weights not found at 'weights/jersey_number_recon/'")
        except Exception as e:
            print(f"⚠️ Could not load temporal jersey model: {e}")
            self.number_model = None

    def _load_profiling_data(self, path: str) -> dict:
        """Loads and parses the match.json / profiling configuration file."""
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("rosters", data)
            except Exception as e:
                print(f"⚠️ Warning: Could not parse {path}: {e}")
        return {}

    def get_team_color(self, crop: np.ndarray) -> Optional[str]:
        """Classifies the team by comparing the chest crop against active match rosters."""
        if crop is None or crop.size == 0 or not self.rosters:
            return None

        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return None

        # Tighter upper-chest crop (avoids shorts, legs, skin, and floor reflections)
        y1, y2 = int(h * 0.12), int(h * 0.50)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        chest_crop = crop[y1:y2, x1:x2]
        if chest_crop.size == 0:
            chest_crop = crop[0 : int(h * 0.6), :]

        hsv = cv2.cvtColor(chest_crop, cv2.COLOR_BGR2HSV)
        total_pixels = chest_crop.shape[0] * chest_crop.shape[1]
        if total_pixels == 0:
            return None

        # HSV Mask definitions tuned specifically for indoor arena lighting
        color_masks = {
            # Low brightness + low saturation = true black/dark kit
            "Black": (hsv[:, :, 2] < 75) & (hsv[:, :, 1] < 90),
            # Low saturation + mid-range brightness = neutral grey (Referees)
            "Grey": (hsv[:, :, 1] < 45) & (hsv[:, :, 2] >= 75) & (hsv[:, :, 2] <= 165),
            "Gray": (hsv[:, :, 1] < 45) & (hsv[:, :, 2] >= 75) & (hsv[:, :, 2] <= 165),
            # Low saturation + high brightness = white
            "White": (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 165),
            # FC Porto Blue (requires minimum saturation and brightness so shadows don't flip it to black)
            "Blue": (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] >= 50),
            "Navy": (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] >= 35) & (hsv[:, :, 2] < 100),
            "Red": ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 160)) & (hsv[:, :, 1] > 60),
            "Green": (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 50),
            "Yellow": (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] > 60),
        }

        best_team = None
        highest_score = -1.0

        for team_name, team_data in self.rosters.items():
            kits = team_data.get("kits", {})
            if not kits:
                kits = {"default": team_data.get("colors", [])}

            team_score = 0.0
            for kit_name, kit_colors in kits.items():
                kit_pixels = 0
                for color_name in kit_colors:
                    c_clean = color_name.strip().title()
                    if c_clean in color_masks:
                        kit_pixels += np.sum(color_masks[c_clean])
                    elif c_clean == "Navy" and "Blue" in color_masks:
                        kit_pixels += np.sum(color_masks["Blue"])
                
                score = kit_pixels / total_pixels
                if score > team_score:
                    team_score = score

            if team_score > highest_score:
                highest_score = team_score
                best_team = team_name
                
        if highest_score >= 0.10:
            return best_team

        return None

    def get_shirt_number(self, crop: np.ndarray, track_id: int = 0, team_name: Optional[str] = None) -> str:
        """Extracts jersey numbers using the temporal sequence model."""
        if self.number_model is None or crop is None or crop.size == 0:
            return ""

        h, w = crop.shape[:2]
        if h < 40 or w < 40:
            return ""

        # Auto-classify team color if team_name was not explicitly passed
        if not team_name:
            team_name = self.get_team_color(crop)

        # Isolate upper torso area containing the jersey number
        ymin, ymax = int(h * 0.15), int(h * 0.65)
        xmin, xmax = int(w * 0.10), int(w * 0.90)
        torso_crop = crop[ymin:ymax, xmin:xmax]

        if torso_crop.size == 0:
            return ""

        # Prevent memory leaks by capping the tracking buffer size
        if len(self.track_crop_buffers) > 200:
            # Retain only the 50 most recent track IDs
            recent_keys = list(self.track_crop_buffers.keys())[-50:]
            self.track_crop_buffers = {k: self.track_crop_buffers[k] for k in recent_keys}

        # Buffer tracking frames per track_id
        if track_id not in self.track_crop_buffers:
            self.track_crop_buffers[track_id] = []

        self.track_crop_buffers[track_id].append(torso_crop)
        if len(self.track_crop_buffers[track_id]) > 8:
            self.track_crop_buffers[track_id].pop(0)

        # Pad buffer to 8 frames so inference works immediately on frame 1
        buffered_crops = list(self.track_crop_buffers[track_id])
        while len(buffered_crops) < 8:
            buffered_crops.append(torso_crop)

        # Prepare 8-frame sequence tensor stack
        tensor_list = []
        for c in buffered_crops[:8]:
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor_list.append(self.transform(pil_img))

        input_tensor = torch.stack(tensor_list).unsqueeze(0).to(self.device)

        with torch.no_grad():
            tens_logits, units_logits = self.number_model(input_tensor)
            tens_pred = torch.argmax(tens_logits, dim=1).item()
            units_pred = torch.argmax(units_logits, dim=1).item()

        raw_number = f"{tens_pred}{units_pred}"

        # Strictly validate against JSON match roster if team name is identified
        if team_name and team_name in self.rosters:
            team_data = self.rosters.get(team_name, {})
            raw_roster = team_data.get("roster", {})
            valid_roster_numbers = {str(k).strip() for k in raw_roster.keys() if not str(k).startswith("?")}

            if valid_roster_numbers:
                variants = [raw_number, str(int(raw_number))] if raw_number.isdigit() else [raw_number]
                for v in variants:
                    if v in valid_roster_numbers:
                        return v
                return ""

        return raw_number
    
    def resolve_player(self, team_name: str, shirt_num: str) -> Dict[str, str]:
        """Looks up player metadata from the profiling dictionary."""
        if team_name in self.rosters and shirt_num in self.rosters[team_name].get("roster", {}):
            player_info = self.rosters[team_name]["roster"][shirt_num]
            return {
                "name": player_info.get("name", "Unknown Player"),
                "position": player_info.get("position", "Unknown"),
                "shirt_num": shirt_num,
                "team": team_name,
            }

        return {
            "name": "Unknown Player",
            "position": "Unknown",
            "shirt_num": shirt_num,
            "team": team_name,
        }