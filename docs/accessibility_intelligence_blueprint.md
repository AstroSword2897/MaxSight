# MaxSight Accessibility Intelligence Blueprint
## Complete Engineering Specification

**Version:** 1.0  
**Date:** 2026  
**Status:** Implementation Ready

---

## Table of Contents

1. [Functional Vision Modeling](#1-functional-vision-modeling)
2. [Behavior-Level Impact Modeling](#2-behavior-level-impact-modeling)
3. [Environmental Context Awareness](#3-environmental-context-awareness)
4. [User Profile & Personalization Engine](#4-user-profile--personalization-engine)
5. [Cross-Sensory Substitution Systems](#5-cross-sensory-substitution-systems)
6. [Safety & Priority Reasoning](#6-safety--priority-reasoning)
7. [Social & Emotional Understanding](#7-social--emotional-understanding)
8. [Environmental Structuring & Spatial Mapping](#8-environmental-structuring--spatial-mapping)
9. [JSON Ontology Schema](#9-json-ontology-schema)
10. [Model Architecture Extensions](#10-model-architecture-extensions)
11. [Training Data Requirements](#11-training-data-requirements)
12. [Implementation Roadmap](#12-implementation-roadmap)

---

## 1. Functional Vision Modeling

### Model Output Fields (Add to MaxSightCNN)

```python
# New outputs in forward() return dict:
{
    'contrast_sensitivity': torch.Tensor,      # [B, 1] 0-1 score
    'field_of_view_usable': torch.Tensor,     # [B, 4] left, right, upper, lower regions
    'glare_risk_level': torch.Tensor,          # [B, 1] 0-3 integer
    'motion_perception_difficulty': torch.Tensor,  # [B, 1] 0-1 score
    'color_discrimination_demand': torch.Tensor,  # [B, num_locations] per-location demand
}
```

### Implementation Details

#### 1.1 Contrast Sensitivity Score
- **Input:** FPN features (P3, P4, P5)
- **Architecture:** 
  ```python
  self.contrast_head = nn.Sequential(
      nn.Conv2d(256, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),
      nn.Linear(128, 1),
      nn.Sigmoid()
  )
  ```
- **Training:** Supervised with synthetic low-contrast images
- **Evaluation:** Correlation with human contrast sensitivity tests

#### 1.2 Field-of-View Usability
- **Input:** FPN features at different spatial regions
- **Architecture:** 4 parallel heads (left, right, upper, lower)
- **Output:** Usability score per quadrant [0-1]
- **Usage:** Shift UI guidance to most usable region

#### 1.3 Glare Risk Level
- **Input:** Brightness histogram + spatial features
- **Architecture:**
  ```python
  self.glare_head = nn.Sequential(
      nn.Linear(256 + 64, 128),  # scene features + brightness stats
      nn.ReLU(),
      nn.Linear(128, 4),  # 0-3 levels
      nn.Softmax(dim=1)
  )
  ```
- **Training:** Labeled glare scenes (bright windows, reflective surfaces)

#### 1.4 Motion Perception Difficulty
- **Input:** Optical flow features (if available) or temporal features
- **Architecture:** Temporal CNN or LSTM
- **Output:** Difficulty score [0-1]
- **Usage:** Simplify narration in high-motion scenes

#### 1.5 Color Discrimination Demand
- **Input:** Per-location features
- **Architecture:** Per-location head (same as text_head)
- **Output:** [B, num_locations, 1] demand score
- **Usage:** Activate colorblind mode for high-demand regions

---

## 2. Behavior-Level Impact Modeling

### Model Output Fields

```python
{
    'object_findability': torch.Tensor,      # [B, num_locations] 0-1 score
    'recognizability': torch.Tensor,         # [B, num_locations] 0-1 score
    'distance_judgement_error': torch.Tensor,  # [B, num_locations] predicted error
    'navigation_difficulty': torch.Tensor,   # [B, 1] 0-1 score
}
```

### Implementation Details

#### 2.1 Object Findability Score
- **Inputs:** Clutter level, occlusion, contrast, size
- **Architecture:**
  ```python
  self.findability_head = nn.Sequential(
      nn.Conv2d(256, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 1, 1),
      nn.Sigmoid()
  )
  ```
- **Training:** User studies - "How easy is it to find X?"
- **Features:** Clutter density, occlusion ratio, contrast ratio, relative size

#### 2.2 Recognizability Score
- **Input:** Blurred/low-detail simulation of detection
- **Architecture:** Similar to findability, but trained on blurred versions
- **Usage:** Trigger magnification when recognizability < threshold

#### 2.3 Distance Judgement Assist
- **Input:** Box size + scene depth (if available) + object class
- **Architecture:**
  ```python
  self.distance_error_head = nn.Sequential(
      nn.Linear(256 + 4 + 1, 128),  # scene + box + class
      nn.ReLU(),
      nn.Linear(128, 1),
      nn.Sigmoid()
  )
  ```
- **Training:** Compare predicted distance vs. actual distance

#### 2.4 Navigation Difficulty Rating
- **Input:** Obstacles, lighting, path width, scene type
- **Architecture:** Scene-level head
- **Output:** Single difficulty score [0-1]
- **Usage:** Adjust guidance frequency and detail level

---

## 3. Environmental Context Awareness

### Model Output Fields

```python
{
    'lighting_class': torch.Tensor,          # [B, 5] bright, dim, backlit, mixed, flicker
    'hazard_density_heatmap': torch.Tensor,  # [B, H, W] pixel-wise danger
    'scene_type': torch.Tensor,              # [B, num_scene_types] probabilities
    'movement_patterns': torch.Tensor,       # [B, num_objects, 4] speed + direction
}
```

### Implementation Details

#### 3.1 Lighting-Class Detector
- **Input:** Brightness histogram + spatial distribution
- **Architecture:**
  ```python
  self.lighting_classifier = nn.Sequential(
      nn.Linear(256 + 64, 128),
      nn.ReLU(),
      nn.Linear(128, 5),
      nn.Softmax(dim=1)
  )
  ```
- **Classes:** bright, dim, backlit, mixed, fluorescent_flicker
- **Usage:** Apply silhouette enhancement for backlit scenes

#### 3.2 Hazard-Density Heatmap
- **Input:** FPN features + object detections
- **Architecture:** Pixel-wise segmentation head
- **Output:** [B, H, W] heatmap (0-1 danger score)
- **Usage:** Overlay vibrations when pointing at high-risk regions

#### 3.3 Scene Type Classifier
- **Input:** Scene embedding (already exists)
- **Architecture:** Extend scene_embedding head
- **Classes:** kitchen, bathroom, classroom, street, crosswalk, hallway, parking_lot, office, bedroom, living_room
- **Usage:** Prioritize objects by scene context

#### 3.4 Movement Pattern Analyzer
- **Input:** Temporal features (requires video or multi-frame)
- **Architecture:** Optical flow + LSTM or 3D CNN
- **Output:** Speed + direction per detected object
- **Usage:** "Person approaching from your right", "Car crossing in 2 seconds"

---

## 4. User Profile & Personalization Engine

### User Profile Schema (JSON)

```json
{
    "user_id": "string",
    "preferred_output_channel": "audio" | "haptic" | "visual" | "hybrid",
    "alert_frequency": "low" | "medium" | "high",
    "reaction_time_ms": 250,
    "blindness_profile": {
        "type": "low_vision" | "no_light_perception" | "tunnel_vision" | 
                "central_vision_only" | "colorblind" | "cvi",
        "severity": 0.0-1.0,
        "field_of_view_degrees": 20
    },
    "condition_modes": ["glaucoma", "amd"],
    "accessibility_preferences": {
        "audio_volume": 0.7,
        "haptic_intensity": 0.8,
        "text_size": "large",
        "contrast_mode": "high"
    }
}
```

### Implementation

#### 4.1 Profile Storage
- **Location:** `ml/utils/user_profiles.py`
- **Format:** JSON files in `user_profiles/` directory
- **API:**
  ```python
  class UserProfile:
      def load(user_id: str) -> Dict
      def save(user_id: str, profile: Dict)
      def get_output_channel(user_id: str) -> str
      def get_alert_frequency(user_id: str) -> str
  ```

#### 4.2 Reaction Time Calibration
- **Location:** `ml/utils/reaction_calibration.py`
- **Method:** Simple test - user presses button when they see/hear cue
- **Storage:** Saved to user profile

#### 4.3 Profile-Based Model Adaptation
- **Modify:** `MaxSightCNN.__init__()` to accept `user_profile: Optional[Dict]`
- **Usage:** Adjust condition_mode, output heads, thresholds based on profile

---

## 5. Cross-Sensory Substitution Systems

### Audio Components

#### 5.1 Spatial Audio Beacons
- **Location:** `ml/utils/audio_output.py`
- **Implementation:**
  ```python
  def generate_spatial_beacon(object_coords, screen_center):
      # Convert object position to stereo panning
      pan = (object_coords[0] - screen_center[0]) / screen_width
      # Generate tone with panning
      return stereo_tone(frequency=440, pan=pan, duration=0.1)
  ```

#### 5.2 Audio Tone Coding
- **Mapping:**
  - High pitch (800-1000 Hz) = hazard
  - Mid pitch (400-600 Hz) = object of interest
  - Low pitch (200-300 Hz) = environment feature
  - Fast rhythm (10 Hz) = urgency
  - Slow rhythm (2 Hz) = neutral

#### 5.3 Scene Summaries
- **Location:** `ml/utils/narration.py`
- **Format:** "3 people ahead, left hallway clear, stairs 2 meters ahead"
- **Implementation:** Template-based generation from detections

### Haptic Components

#### 5.4 Vibration Direction Encoding
- **Location:** `ml/utils/haptic_output.py`
- **Mapping:**
  - Right wrist buzz → object right
  - Left wrist buzz → object left
  - Long buzz → hazard head-on
  - Pulsing → moving object approaching

#### 5.5 Intensity Encoding
- **Mapping:**
  - Strong vibration = close object
  - Weak vibration = far object
  - Ramping frequency = approaching hazard

---

## 6. Safety & Priority Reasoning

### Priority Score System

```python
# Priority levels (0-100)
PRIORITY_IMMEDIATE_HAZARD = 90-100  # vehicle, drop-off, stove flame
PRIORITY_NAVIGATION = 70-89         # stairs, doors, signs
PRIORITY_USEFUL = 40-69             # chairs, handles, pathways
PRIORITY_NON_ESSENTIAL = 0-39        # plants, posters, sky
```

### Implementation

#### 6.1 Priority Head
- **Location:** Add to `MaxSightCNN`
- **Architecture:**
  ```python
  self.priority_head = nn.Sequential(
      nn.Conv2d(256, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 1, 1),
      nn.Sigmoid()  # Output 0-1, scale to 0-100
  )
  ```
- **Training:** Labeled priority scores from accessibility experts

#### 6.2 Priority-Based Filtering
- **Location:** `MaxSightCNN.get_detections()`
- **Logic:**
  ```python
  def filter_by_priority(detections, user_profile):
      alert_freq = user_profile['alert_frequency']
      if alert_freq == 'low':
          threshold = 70  # Only hazards + navigation
      elif alert_freq == 'medium':
          threshold = 40  # + useful objects
      else:
          threshold = 0   # All objects
      return [d for d in detections if d['priority'] >= threshold]
  ```

---

## 7. Social & Emotional Understanding

### Model Output Fields

```python
{
    'body_orientation': torch.Tensor,    # [B, num_people, 3] facing, away, sideways
    'gestures': torch.Tensor,            # [B, num_people, num_gestures] probabilities
    'interaction_intent': torch.Tensor,  # [B, num_people] approaching, looking, ignoring
    'crowd_flow': torch.Tensor,          # [B, 2] direction vector
}
```

### Implementation Details

#### 7.1 Body Orientation Detection
- **Input:** Person detections + pose estimation (if available)
- **Architecture:** Extend person detection with orientation head
- **Output:** facing_user, facing_away, sideways
- **Usage:** "Person facing you" vs "Person walking away"

#### 7.2 Gesture Classifier
- **Input:** Person bounding box + temporal features
- **Architecture:** Temporal CNN or LSTM
- **Classes:** waving, pointing, stop_gesture, beckoning, none
- **Training:** Labeled gesture videos

#### 7.3 Interaction Intent Detection
- **Input:** Person position + movement + gaze direction (if available)
- **Architecture:** Multi-task head
- **Output:** approaching, looking_at_user, ignoring
- **Usage:** "Cashier looking at you", "Person approaching"

#### 7.4 Crowd Flow Direction
- **Input:** Multiple person detections + movement vectors
- **Architecture:** Aggregate movement vectors
- **Output:** [direction_x, direction_y] normalized vector
- **Usage:** "Crowd moving left, align with flow"

---

## 8. Environmental Structuring & Spatial Mapping

### Model Output Fields

```python
{
    'semantic_segmentation': torch.Tensor,     # [B, num_classes, H, W] pixel-wise labels
    'room_zones': torch.Tensor,               # [B, 5] walkable, obstacles, walls, furniture, entry_exit
    'predicted_object_placement': torch.Tensor,  # [B, num_objects, 2] expected locations
    'text_regions': torch.Tensor,             # [B, num_locations] (already exists)
    'pathfinding_map': torch.Tensor,          # [B, H, W] walkability map
}
```

### Implementation Details

#### 8.1 Semantic Room Zones
- **Input:** FPN features
- **Architecture:** Segmentation head
- **Classes:** walkable_area, obstacles, walls, furniture, entry_exit_points
- **Usage:** Pathfinding, obstacle avoidance

#### 8.2 Predicted Object Placement
- **Input:** Scene type + room layout
- **Architecture:** Conditional generation (if kitchen → expect stove on counter)
- **Training:** Room layouts + object locations dataset

#### 8.3 Text Region Detection + OCR
- **Status:** Already implemented (`text_head`)
- **Enhancement:** Add OCR integration
- **Location:** `ml/utils/ocr_integration.py`
- **Features:** Magnification, high contrast mode, read aloud

#### 8.4 Pathfinding on Segmented Rooms
- **Input:** Semantic segmentation + current position
- **Algorithm:** A* or Dijkstra on walkability map
- **Location:** `ml/utils/pathfinding.py`
- **Output:** Recommended safe walking path
- **Usage:** "Follow path to door, 3 steps ahead"

---

## 9. JSON Ontology Schema

### Complete Output Schema

```json
{
    "frame_id": "string",
    "timestamp": 1234567890.0,
    "user_profile_id": "string",
    
    "detections": [
        {
            "class": 42,
            "class_name": "door",
            "confidence": 0.95,
            "box": [0.5, 0.5, 0.2, 0.15],
            "distance": "near",
            "urgency": 2,
            "priority": 75,
            "findability": 0.8,
            "recognizability": 0.9,
            "color_discrimination_demand": 0.3
        }
    ],
    
    "scene_analysis": {
        "scene_type": "kitchen",
        "lighting_class": "bright",
        "contrast_sensitivity": 0.85,
        "glare_risk_level": 1,
        "motion_perception_difficulty": 0.2,
        "navigation_difficulty": 0.3,
        "hazard_density_heatmap": "base64_encoded_image"
    },
    
    "functional_vision": {
        "field_of_view_usable": {
            "left": 0.9,
            "right": 0.8,
            "upper": 0.7,
            "lower": 0.95
        },
        "contrast_sensitivity": 0.85,
        "glare_risk_level": 1,
        "motion_perception_difficulty": 0.2
    },
    
    "social_context": {
        "people": [
            {
                "id": 0,
                "box": [0.3, 0.4, 0.1, 0.2],
                "body_orientation": "facing_user",
                "gesture": "waving",
                "interaction_intent": "approaching",
                "distance": "medium"
            }
        ],
        "crowd_flow": [0.1, -0.05]
    },
    
    "spatial_mapping": {
        "semantic_segmentation": "base64_encoded_image",
        "room_zones": {
            "walkable": 0.6,
            "obstacles": 0.1,
            "walls": 0.2,
            "furniture": 0.08,
            "entry_exit": 0.02
        },
        "pathfinding": {
            "recommended_path": [[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]],
            "chokepoints": [[0.4, 0.5]],
            "optimal_direction": [0.1, 0.05]
        }
    },
    
    "output_recommendations": {
        "audio": {
            "spatial_beacons": [
                {"object_id": 0, "pan": -0.3, "frequency": 600, "duration": 0.1}
            ],
            "scene_summary": "Door ahead, person on left, clear path forward"
        },
        "haptic": {
            "vibrations": [
                {"type": "right_wrist", "intensity": 0.7, "pattern": "single"},
                {"type": "left_wrist", "intensity": 0.3, "pattern": "pulsing"}
            ]
        },
        "visual": {
            "highlight_regions": [[0.5, 0.5, 0.2, 0.15]],
            "text_overlays": [
                {"text": "Door", "position": [0.5, 0.5], "size": "large"}
            ]
        }
    }
}
```

---

## 10. Model Architecture Extensions

### Extended MaxSightCNN Architecture

```python
class MaxSightCNN(nn.Module):
    def __init__(self, ..., enable_accessibility_features: bool = True):
        # ... existing code ...
        
        if enable_accessibility_features:
            # Functional Vision Modeling
            self.contrast_head = ...
            self.field_of_view_head = ...
            self.glare_head = ...
            self.motion_difficulty_head = ...
            self.color_demand_head = ...
            
            # Behavior-Level Impact
            self.findability_head = ...
            self.recognizability_head = ...
            self.distance_error_head = ...
            self.navigation_difficulty_head = ...
            
            # Environmental Context
            self.lighting_classifier = ...
            self.hazard_heatmap_head = ...
            self.scene_type_classifier = ...
            self.movement_analyzer = ...
            
            # Priority Reasoning
            self.priority_head = ...
            
            # Social Understanding
            self.body_orientation_head = ...
            self.gesture_classifier = ...
            self.interaction_intent_head = ...
            self.crowd_flow_head = ...
            
            # Spatial Mapping
            self.semantic_segmentation_head = ...
            self.pathfinding_head = ...
```

### Forward Pass Extensions

```python
def forward(self, images, audio_features=None, user_profile=None):
    # ... existing forward pass ...
    
    outputs = {
        # Existing outputs
        'classifications': cls_logits,
        'boxes': box_preds,
        'objectness': obj_scores,
        'text_regions': text_scores,
        'scene_embedding': scene_emb,
        'urgency_scores': urgency,
        'distance_zones': distances,
        
        # New accessibility outputs
        'contrast_sensitivity': self.contrast_head(scene_feats),
        'field_of_view_usable': self.field_of_view_head(scene_feats),
        'glare_risk_level': self.glare_head(scene_feats),
        'motion_perception_difficulty': self.motion_difficulty_head(scene_feats),
        'color_discrimination_demand': self.color_demand_head(det_feats),
        
        'object_findability': self.findability_head(det_feats),
        'recognizability': self.recognizability_head(det_feats),
        'distance_judgement_error': self.distance_error_head(det_feats),
        'navigation_difficulty': self.navigation_difficulty_head(scene_feats),
        
        'lighting_class': self.lighting_classifier(scene_feats),
        'hazard_density_heatmap': self.hazard_heatmap_head(fpn_features),
        'scene_type': self.scene_type_classifier(scene_feats),
        'movement_patterns': self.movement_analyzer(temporal_features),
        
        'priority_scores': self.priority_head(det_feats),
        
        'body_orientation': self.body_orientation_head(person_features),
        'gestures': self.gesture_classifier(person_features),
        'interaction_intent': self.interaction_intent_head(person_features),
        'crowd_flow': self.crowd_flow_head(person_features),
        
        'semantic_segmentation': self.semantic_segmentation_head(fpn_features),
        'room_zones': self.room_zones_head(scene_feats),
        'pathfinding_map': self.pathfinding_head(semantic_seg),
    }
    
    return outputs
```

---

## 11. Training Data Requirements

### Dataset Structure

```
datasets/
├── accessibility/
│   ├── functional_vision/
│   │   ├── contrast_sensitivity/
│   │   ├── glare_scenes/
│   │   ├── motion_difficulty/
│   │   └── color_discrimination/
│   ├── behavior_impact/
│   │   ├── findability_scores/
│   │   ├── recognizability_scores/
│   │   └── navigation_difficulty/
│   ├── environmental_context/
│   │   ├── lighting_classes/
│   │   ├── scene_types/
│   │   └── hazard_annotations/
│   ├── social_understanding/
│   │   ├── body_orientation/
│   │   ├── gestures/
│   │   └── interaction_intent/
│   └── spatial_mapping/
│       ├── semantic_segmentation/
│       └── pathfinding/
```

### Labeling Guidelines

#### Functional Vision
- **Contrast Sensitivity:** Score 0-1 based on edge detection tests
- **Glare Risk:** 0-3 scale (0=none, 1=low, 2=medium, 3=high)
- **Motion Difficulty:** Score based on optical flow magnitude

#### Behavior Impact
- **Findability:** User study ratings (0-1) - "How easy to find?"
- **Recognizability:** Blur simulation + recognition accuracy
- **Navigation Difficulty:** Expert ratings (0-1)

#### Environmental Context
- **Lighting Classes:** Manual labeling (bright, dim, backlit, mixed, flicker)
- **Scene Types:** COCO-style scene classification
- **Hazard Density:** Pixel-wise annotations

#### Social Understanding
- **Body Orientation:** Manual labeling (facing, away, sideways)
- **Gestures:** Video annotations with gesture classes
- **Interaction Intent:** Expert labeling (approaching, looking, ignoring)

---

## 12. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Add priority scoring head
- [ ] Implement user profile system
- [ ] Add scene type classifier
- [ ] Create JSON ontology schema

### Phase 2: Functional Vision (Week 3-4)
- [ ] Contrast sensitivity head
- [ ] Glare risk detection
- [ ] Field-of-view usability
- [ ] Motion perception difficulty

### Phase 3: Behavior Impact (Week 5-6)
- [ ] Object findability scoring
- [ ] Recognizability head
- [ ] Distance judgement assist
- [ ] Navigation difficulty rating

### Phase 4: Environmental Context (Week 7-8)
- [ ] Lighting classifier
- [ ] Hazard density heatmap
- [ ] Movement pattern analyzer
- [ ] Scene-based object prioritization

### Phase 5: Social Understanding (Week 9-10)
- [ ] Body orientation detection
- [ ] Gesture classifier
- [ ] Interaction intent detection
- [ ] Crowd flow analysis

### Phase 6: Spatial Mapping (Week 11-12)
- [ ] Semantic segmentation head
- [ ] Room zone classification
- [ ] Pathfinding algorithm
- [ ] OCR integration

### Phase 7: Cross-Sensory Output (Week 13-14)
- [ ] Spatial audio beacons
- [ ] Audio tone coding
- [ ] Haptic vibration system
- [ ] Scene narration generation

### Phase 8: Integration & Testing (Week 15-16)
- [ ] End-to-end testing
- [ ] User profile integration
- [ ] Performance optimization
- [ ] Documentation

---

## Next Steps

1. **Review this blueprint** - Confirm all requirements
2. **Prioritize features** - Which to implement first?
3. **Create implementation tasks** - Break into sprints
4. **Set up data collection** - Start gathering training data
5. **Begin Phase 1** - Foundation features

---

**Status:** Ready for implementation  
**Last Updated:** 2026  
**Maintainer:** MaxSight Team

