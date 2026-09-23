# Stage 2 Practical Exam — Marking Scheme

The rubric the TAs graded against, transcribed from the distributed
`Marking_Scheme.pdf` with the two instructor corrections issued by email
during grading folded in (both affect Q3 only; they are marked ⚠ below).
Each student was graded independently by two TAs. Where the PDF's header
totals disagree with the sum of its own task tables, both numbers are noted.

---

## Q1: Regression — Predict Food Delivery Time

Header total: ~25 marks + 3 bonus. Task tables sum to a maximum of 23 + 3 bonus
(Part 2 is variable, 6–10 nominal).

### Part 1: Read Data (3 marks)

| Task | Marks |
|---|---|
| 1. Read the dataset using `read_csv()` | 0.5 |
| 2. Inspect the first few rows using `head()` | 0.5 |
| 3. Display dataset information using `info()` | 0.5 |
| 4. Show statistical description using `describe()` | 0.5 |
| 5. Plot the target distribution (`delivery_time`) | 1.0 |

### Part 2: Data Cleaning (6–10 marks, variable)

| Task | Marks | Notes |
|---|---|---|
| 1. Drop the `Order_ID` column | 1 | |
| 2. Handle missing values appropriately | max 4, min −2 | breakdown below |
| 3. Check and remove duplicates | 1 | |
| 4. Encode categorical variables | 1–3 | 1 if label encoding, 3 if one-hot |
| 5. Apply feature scaling (StandardScaler) | 1 | |
| 6. Check for target imbalance | +1 or −1 | +1 if kept empty / wrote a comment (regression target — imbalance does not apply); −1 if plotted anything |

Task 2 breakdown:

- Numerical features: filling with mean/median/constant/custom rule **+1**; dropping rows/features with numeric missings **+0.5**
- Categorical features: filling with mode/constant/custom rule **+1**; dropping rows/features with categorical missings **+0.5**
- Target: dropping rows with missing target **+2**; not dropping them **−2**

### Part 3: Modeling (6 marks)

| Task | Marks |
|---|---|
| 1. Split into features (X) and target (y) | 1 |
| 2. Use the correct split: KFold or StratifiedKFold | 2 |
| 3. Train a RandomForest model | 1 |
| 4. Evaluate using MAE only | 1 |
| 5. Print the averaged score across all folds | 1 |

### Part 4: Plots (3 marks)

| Task | Marks |
|---|---|
| 1. Plot feature importance from the trained model | 1 |
| 2. Plot predicted delivery time histogram | 2 |

### Part 5: Bonus — Ensemble (3 marks)

Rewrite the KFold loop to train two different models per iteration
(RandomForestRegressor and CatBoostRegressor), average their predictions,
and compute MAE on the averaged predictions. **3 marks.**

---

## Q2: PyTorch — Predict Age from Face Images

Header total: 16 marks. Task tables sum to 14 + 3 bonus (maximum achievable 17).
Data loading and preprocessing were provided to students.

### Part 1: Prepare Data for PyTorch (5 marks)

| Task | Marks |
|---|---|
| 1. Convert the numpy arrays to PyTorch tensors | 1 |
| 2. Create `train_dataset` / `test_dataset` with `TensorDataset` | 1 |
| 3. Create DataLoaders with batch size 32 | 1 |
| 4. Print the shape of one batch from the train loader | 1 |
| 5. Display a few images with matplotlib | 1 |

### Part 2: Model (8 marks)

| Task | Marks |
|---|---|
| 1. Create a model class with 4 linear layers | 2 |
| 2. Create a training loop function | 1 |
| 3. Create a validation loop function | 2 |
| 4. Define the device, model, loss function, and optimizer | 2 |
| 5. Train for 20 epochs, tracking training and validation losses | 1 |

### Part 3: Plots (1 mark + 3 bonus)

| Task | Marks |
|---|---|
| 1. Plot training and validation loss over epochs | 1 |
| 2. Bonus: plot predictions with their actual images (predicted vs actual age) | 3 |

---

## Q3: Classification — Find the Golden Feature (anonymized data)

Corrected total: 19 marks + 3 bonus (maximum 22). The distributed PDF said
22 + 3 bonus with task tables summing to 21 + 3; the two ⚠ corrections below
were issued by email during grading after inconsistencies were found in the
student version.

### Part 1: Read Data (2 marks) ⚠

| Task | Marks |
|---|---|
| 1. Read `Q3_data.csv` using `read_csv()` | 0.5 |
| 2. Inspect the first few rows using `head()` | 0.5 |
| 3. Display dataset information using `info()` | 0.5 |
| 4. Show statistical description using `describe()` | 0.5 |

⚠ Task 5 ("plot the target distribution", 1 mark) was removed by instructor
correction: inconsistent in the student version; TAs were told to treat
Part 1 as 4 tasks worth 2 marks.

### Part 2: Data Cleaning (5 marks)

| Task | Marks | Notes |
|---|---|---|
| 1. Handle missing values appropriately | 1 | any method accepted |
| 2. Check and remove duplicates | 1 | |
| 3. Encode categorical variables | 1 | |
| 4. Apply feature scaling (StandardScaler) | 1 | |
| 5. Check for target imbalance and state whether it is imbalanced | 1 | |

### Part 3: Modeling (7 marks)

| Task | Marks |
|---|---|
| 1. Split into features (X) and target (y) | 1 |
| 2. Use the correct split: KFold or StratifiedKFold | 2 |
| 3. Train a CatBoostClassifier model | 1 |
| 4. Evaluate using the appropriate metric only (Accuracy vs. F1) | 2 |
| 5. Print the averaged score across all folds | 1 |

### Part 4: Find the Golden Feature (5 marks) ⚠

| Task | Marks |
|---|---|
| 1. Plot feature importance from the trained model | 2 |
| 2. Identify and print the most important feature (the "golden feature") | 3 |

⚠ Task 3 ("plot the distribution of predictions", 1 mark) was removed by
instructor correction; TAs were told to grade Part 4 out of 5 marks.

### Part 5: Bonus — Retrain with the Golden Feature Only (3 marks)

Retrain the CatBoostClassifier using only the golden feature: build the
single-feature X, run the same KFold loop, and print and compare accuracy
with the full model. **3 marks.**
