# KAUST AI Stage 3 Exam — Grading Rubric

*February 2026*

**Total Points:** 35 points (+ 13 bonus points)

---

## Question 1: Handwritten Character Classification using Transfer Learning

**Total:** 12 points (+ 4 bonus)

### Part 1: Load and Prepare Data (3 points)

| Task | Points |
|------|-------:|
| Task 1: Complete transforms (Resize, ToTensor, Normalize) | 1.5 |
| Task 2: Create DataLoaders | 0.5 |
| Task 3: Display sample images | 1.0 |
| **Part 1 Total** | **3.0** |

### Part 2: Load Pretrained Model and Adapt (3 points)

| Task | Points |
|------|-------:|
| Task 1: Load pretrained EfficientNetV2-Small | 0.5 |
| Task 2: Freeze the backbone (feature extractor) | 1.5 |
| Task 3: Replace classifier head for 26 classes | 1.0 |
| **Part 2 Total** | **3.0** |

### Part 3: Training and Validation Functions (3.5 points)

| Task | Points |
|------|-------:|
| Correct training loop | 0.5 |
| Correct validation loop | 1.0 |
| Label Fix: Fixed labels (subtract 1 for 0-indexed) | +2.0 |
| Penalty: Did NOT fix labels (1–26 instead of 0–25) | −2.0 |
| **Part 3 Total** | **3.5** |

### Part 4: Training (2.5 points)

| Task | Points |
|------|-------:|
| Task 1: Set up device, model, loss, optimizer | 1.0 |
| Task 2: Train the model | 0.5 |
| Task 3: Plot training and validation losses | 0.5 |
| Task 4: Plot training and validation accuracy | 0.5 |
| **Part 4 Total** | **2.5** |

### Part 5: Bonus — Test Time Augmentation (4 points)

| Bonus Task | Points |
|------------|-------:|
| Defined 3 predictions correctly (original, h flip, v flip) | 3.0 |
| Averaged predictions correctly | 1.0 |
| **Part 5 Bonus Total** | **4.0** |

---

## Question 2: Potato Disease Classification using CNN

**Total:** 11 points (+ 4 bonus)

### Part 1: Load and Prepare Data (4 points)

| Task | Points |
|------|-------:|
| Task 1: Dataset class or ImageFolder | 1.0 |
| Task 2: Create train/test datasets and DataLoaders | 0.5 |
| Task 3: Transforms (RandomRotation, Resize to 32x32) | 1.5 |
| Task 4: Display sample images with labels | 1.0 |
| **Part 1 Total** | **4.0** |

### Part 2: Build the CNN Model (4 points)

| Task | Points |
|------|-------:|
| Task 1: CNN model class with 5 convolutional layers | 2.0 |
| Task 2: Added Batch Normalization | 2.0 |
| **Part 2 Total** | **4.0** |

### Part 3: Training and Validation Functions (1 point)

| Task | Points |
|------|-------:|
| Task 1: Training loop function | 0.5 |
| Task 2: Validation loop function | 0.5 |
| **Part 3 Total** | **1.0** |

### Part 4: Training (2 points)

| Task | Points |
|------|-------:|
| Task 1: Set up device, model, loss, optimizer | 1.0 |
| Task 2: Train the model | 0.5 |
| Task 3: Plot losses and accuracy | 0.5 |
| **Part 4 Total** | **2.0** |

### Part 5: Bonus — Residual Connection (4 points)

| Bonus Task | Points |
|------------|-------:|
| Defined residual connection correctly (even if it is not from 2nd to 4th layer) | 3.0 |
| Retrain + Plot losses and accuracy | 1.0 |
| **Part 5 Bonus Total** | **4.0** |

---

## Question 3: Multi-Class Segmentation for Underwater Imagery

**Total:** 12 points (no bonus)

### Task 1: Dataset Class (3 points)

| Task | Points |
|------|-------:|
| Build custom dataset class for images and masks | 2.0 |
| Create DataLoaders | 0.5 |
| Display sample images and masks | 0.5 |
| **Task 1 Total** | **3.0** |

### Task 2: Model Class (2 points)

| Task | Points |
|------|-------:|
| Define pretrained U-Net with efficientnet-b1 encoder | 2.0 |
| **Task 2 Total** | **2.0** |

### Task 3: Training and Validation Loops (1 point)

| Task | Points |
|------|-------:|
| Training loop | 0.5 |
| Validation loop | 0.5 |
| **Task 3 Total** | **1.0** |

### Task 4: Running Training (3 points)

| Task | Points |
|------|-------:|
| Define loss and optimizer | 1.5 |
| Train the model | 0.5 |
| Print training and validation losses | 0.5 |
| Plot loss curve | 0.5 |
| **Task 4 Total** | **3.0** |

### Task 5: Visualizing Predictions (3 points)

| Task | Points |
|------|-------:|
| Visualize predictions vs ground truth for multiple images | 3.0 |
| **Task 5 Total** | **3.0** |

---

## Bonus Question 4: Building a Custom Dataset for Image Colorization

**Total:** 5 points

| Task | Points |
|------|-------:|
| Correctly implemented `ColorizationDataset` class:<br>• `__init__`: Store dataset<br>• `__len__`: Return length<br>• `rgb_to_grayscale`: Correct formula and dimensions<br>• `__getitem__`: Return (grayscale, color) tuple<br>• Correct visualization | 5.0 |
| Any mistake in implementation | 0.0 |
| **Question 4 Total** | **5.0** |

> **Note:** This is an all-or-nothing question. The implementation must be fully correct to receive points.

> **Note:** Accuracy should not affect your scoring as long as task is implemented correctly.
