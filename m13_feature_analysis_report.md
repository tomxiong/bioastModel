# M13 Dataset Feature Analysis Report

## Overview
This report provides a comprehensive analysis of the m13.json dataset, focusing on samples with hole_number >= 25 (70x70 images) for training deep learning models.

**Dataset Statistics:**
- Total annotations: 1,824
- 70x70 images (hole_number >= 25): 1,824 (100%)
- Analysis date: August 31, 2025

## 1. Growth Level Analysis

### Unique Values (3 classes):
- **positive**: 953 samples (52.2%)
- **negative**: 782 samples (42.9%)
- **weak_growth**: 89 samples (4.9%)

### Key Insights:
- Relatively balanced binary classification (positive vs negative)
- Small weak_growth class may need data augmentation
- Imbalance ratio: 10.7:1 (weak_growth:positive)

## 2. Growth Pattern Analysis

### Unique Values (9 classes):
1. **clustered**: 823 samples (45.1%)
2. **clean**: 782 samples (42.9%)
3. **heavy_growth**: 123 samples (6.7%)
4. **small_dots**: 72 samples (3.9%)
5. **irregular_areas**: 11 samples (0.6%)
6. **scattered**: 6 samples (0.3%)
7. **light_gray**: 5 samples (0.3%)
8. **default_positive**: 1 sample (0.1%)
9. **default_weak_growth**: 1 sample (0.1%)

### Key Insights:
- Highly imbalanced distribution
- Top 2 patterns (clustered, clean) represent 88% of data
- Rare classes (irregular_areas, scattered, light_gray, defaults) have insufficient samples
- Severe imbalance ratio: 823:1

## 3. Microbe Type Analysis

### Unique Values (1 class):
- **bacteria**: 1,824 samples (100.0%)

### Key Insights:
- No variation in microbe type
- Not suitable for multi-class classification based on this feature

## 4. Interference Factors Analysis

### Unique Values (6 factors):
1. **pores**: 609 occurrences (33.4%)
2. **气孔 (air pores)**: 429 occurrences (23.5%)
3. **debris**: 96 occurrences (5.3%)
4. **气孔重叠 (overlapping pores)**: 86 occurrences (4.7%)
5. **杂质 (impurities)**: 52 occurrences (2.9%)
6. **noise**: 24 occurrences (1.3%)

### Common Combinations (11 unique combinations):
1. **pores only**: 584 samples (32.0%)
2. **no_interference**: 583 samples (32.0%)
3. **气孔 only**: 401 samples (22.0%)
4. **气孔重叠 only**: 83 samples (4.6%)
5. **debris only**: 71 samples (3.9%)
6. **杂质 + 气孔**: 27 samples (1.5%)
7. **debris + pores**: 25 samples (1.4%)
8. **noise only**: 24 samples (1.3%)
9. **杂质 only**: 23 samples (1.3%)
10. **杂质 + 气孔重叠**: 2 samples (0.1%)
11. **气孔 + 气孔重叠**: 1 sample (0.1%)

### Key Insights:
- 64% of samples have interference factors
- Most common interference: pores/气孔 (56.9% combined)
- Complex combinations are rare
- Good balance between interference vs no interference

## Classification Task Recommendations

### Recommended Binary Classification Tasks:

#### 1. **Growth Level Classification** (Highly Recommended)
- **Classes**: positive (953) vs negative (782)
- **Total samples**: 1,735
- **Balance**: Good (52.2% vs 42.9%)
- **Viability**: Excellent - well-balanced, sufficient samples

#### 2. **Interference Detection** (Recommended)
- **Classes**: with_interference (1,241) vs no_interference (583)
- **Total samples**: 1,824
- **Balance**: Acceptable (68.0% vs 32.0%)
- **Viability**: Good - clear distinction, sufficient samples

#### 3. **Major Growth Pattern Classification** (Conditional)
- **Classes**: clustered (823) vs clean (782)
- **Total samples**: 1,605
- **Balance**: Good (45.1% vs 42.9%)
- **Viability**: Good but excludes other patterns

### Multi-Class Classification Challenges:

#### **Growth Pattern Multi-Class** (Not Recommended)
- **Issue**: Severe class imbalance
- **Problem**: 7 classes have < 25 samples
- **Recommendation**: Group rare patterns or use binary approach

#### **Growth Level Multi-Class** (Possible with Augmentation)
- **Classes**: positive (953), negative (782), weak_growth (89)
- **Issue**: weak_growth class is small
- **Recommendation**: Use data augmentation for weak_growth

### Recommended Implementation Strategy:

#### Phase 1: Binary Classification (Immediate)
1. **Primary Task**: Growth Level (positive vs negative)
   - Use existing 1,735 samples
   - Target: 95%+ accuracy
   - Model: Enhanced MobileNetV3 or EfficientNet

2. **Secondary Task**: Interference Detection
   - Use all 1,824 samples
   - Target: 85%+ accuracy
   - Model: Same architecture as primary task

#### Phase 2: Enhanced Multi-Class (Future)
1. **Expanded Growth Level**: Include weak_growth with augmentation
   - Augment weak_growth samples to ~200-300
   - Target: 90%+ accuracy across 3 classes

2. **Growth Pattern Grouping**:
   - Group rare patterns: 
     - `other_positive` = irregular_areas + scattered + light_gray + default_positive
     - `other_weak` = default_weak_growth
   - Result: 5 classes with better balance

#### Phase 3: Multi-Task Learning (Advanced)
- **Combined Model**: Growth Level + Interference Detection
- **Benefits**: Shared feature extraction, improved efficiency
- **Architecture**: Multi-head CNN with shared backbone

### Data Preparation Recommendations:

#### Train/Validation/Test Split:
- **Training**: 70% (1,277 samples)
- **Validation**: 15% (274 samples)
- **Test**: 15% (273 samples)

#### Class Balancing:
- **For binary tasks**: Use weighted loss if imbalance > 60:40
- **For multi-class**: Apply oversampling to minority classes
- **Data augmentation**: Essential for weak_growth class

#### Feature Engineering:
1. **Primary features**: growth_level, interference_factors
2. **Secondary features**: growth_pattern (grouped)
3. **Auxiliary features**: hole_number, image_metadata

### Success Metrics:

#### Binary Classification:
- **Target Accuracy**: >95%
- **Target F1-Score**: >0.90
- **Minimum viable**: >85% accuracy, >0.80 F1

#### Multi-Class Classification:
- **Target Accuracy**: >85%
- **Target F1-Score**: >0.80
- **Minimum viable**: >75% accuracy, >0.70 F1

### Risk Assessment:

#### High Risk:
- Growth pattern multi-class (severe imbalance)
- Complex interference combinations (insufficient samples)

#### Medium Risk:
- Growth level 3-class (weak_growth too small)
- Multi-task learning complexity

#### Low Risk:
- Binary growth level classification
- Interference detection
- Model training stability

## Conclusion

The m13 dataset is well-suited for **binary classification tasks**, particularly:
1. **Growth Level Detection** (positive vs negative)
2. **Interference Detection** (with vs without interference)

Multi-class classification faces challenges due to severe class imbalance in growth patterns and limited microbe type variation. 

**Recommended approach**: Start with binary classification, then expand to grouped multi-class with data augmentation as a second phase.