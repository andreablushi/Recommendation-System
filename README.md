# Recommendation System for Process Monitoring

Implementation of **predictive process monitoring** and **prescriptive recommendations** using Decision Trees with Boolean encoding.

This project was developed as part of the course "Process Mining and Management", held by Prof. Chiara Di Francescomarino at the University of Trento, within the Master's Degree in Computer Science.

[View Full Report (PDF)](docs/report.pdf)

## Table of Contents

* [1. Overview](#1-overview)
* [2. Prerequisites](#2-prerequisites)
* [3. Repository Structure](#3-repository-structure)
* [4. Installation and Setup](#4-installation-and-setup)
* [5. Usage](#5-usage)
* [6. Troubleshooting](#6-troubleshooting)

## 1. Overview

This recommendation system analyzes process execution logs to: 
- **Predict outcomes** of ongoing process instances using Decision Tree classifiers
- **Provide recommendations** for traces predicted to have negative outcomes
- **Evaluate performance** using accuracy, precision, recall, and F1-score metrics
- **Visualize decision trees** and confusion matrices

The system uses Boolean encoding of process activities and hyperparameter optimization to train effective classifiers on XES event logs.

## 2. Prerequisites

### Required Software
- **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
- **pip**: Package installer for Python (usually included with Python)
- **Jupyter Notebook**: Installed via `requirements.txt`
- **Git**: [Download Git](https://git-scm.com/downloads) (optional, for cloning)

## 3. Repository Structure

```
Recommendation-System/
├── src/                          # Source code modules
│   ├── utils.py                  # Core utilities (log processing, encoding, optimization)
│   └── plotting.py               # Visualization functions (trees, matrices, metrics)
├── logs/                         # Compressed log files (XES format)
├── docs/                         # Documentation and outputs
│   ├── media/                    # Generated visualizations
│   ├── LaTeX/                    # LaTeX report files
│   └── report.pdf                # Final project report
├── app.log                       # Application log file
├── notebook.ipynb                # Main Jupyter Notebook for experiments
├── requirements.txt              # Python dependencies
└── README.md                     
```

## 4. Installation and Setup

### Step 1: Clone the Repository

**Option A: Using Git (Recommended)**
```bash
git clone https://github.com/andreablushi/Recommendation-System.git
cd Recommendation-System
```

**Option B: Download ZIP**
1. Go to: https://github.com/andreablushi/Recommendation-System
2. Click the green **Code** button → **Download ZIP**
3. Extract the ZIP file to your desired location
4. Open a terminal/command prompt and navigate to the extracted folder

### Step 2: Verify Python Installation

**Check Python version:**

**Linux/macOS:**
```bash
python3 --version
```

**Windows (Command Prompt or PowerShell):**
```cmd
python --version
```

You should see Python 3.8 or higher. If not, [download and install Python](https://www.python.org/downloads/).

### Step 3: Create a Virtual Environment (Recommended)

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Verify activation:** Your terminal prompt should now show `(venv)` at the beginning.

### Step 4: Install Required Packages

```bash
pip install -r requirements.txt
```

This will install all dependencies including:
- `pm4py` (process mining library)
- `scikit-learn` (machine learning)
- `pandas`, `numpy` (data processing)
- `matplotlib`, `seaborn`, `plotly` (visualization)
- `hyperopt` (hyperparameter optimization)
- `jupyter` (notebook environment)

### Step 5: Extract Event Logs

**Linux/macOS:**
```bash
cd logs
unzip '*.zip'
cd ..
```

**Manual extraction (all platforms):**
1. Navigate to the `logs/` folder
2. Right-click each `.zip` file
3. Select **Extract Here** or **Extract All**

After extraction, you should have:
- `logs/Production_avg_dur_training_0-80.xes`
- `logs/Production_avg_dur_testing_80-100.xes`

## 5. Usage

### Starting Jupyter Notebook

From the project root directory (with the virtual environment activated):

```bash
jupyter notebook
```

This will: 
1. Start the Jupyter server
2. Open your default web browser
3. Display the file browser at `http://localhost:8888`

### Running the Main Notebook

1. In the Jupyter file browser, click `notebook.ipynb`.
2. The notebook contains the complete pipeline with the following sections.

#### Notebook Sections: 

**1. Data Loading and Preprocessing**
- Import XES event logs using `pm4py`
- Create trace prefixes of a specified length
- Apply Boolean encoding to activities

**2. Decision Tree Classifier**
- Hyperparameter optimization using `hyperopt`
- Train a decision tree on Boolean-encoded prefixes
- Visualize the decision tree structure

**3. Model Evaluation**
- Load and process test logs
- Generate predictions
- Display the confusion matrix
- Calculate metrics: accuracy, precision, recall, and F1-score

**4. Recommendations**
- Extract positive paths from the Decision Tree
- Generate recommendations for negative predictions
- Visualize recommendations on the tree
- Evaluate recommendation quality

**5. Prefix Length Analysis**
   - Test different prefix lengths (1-12)
   - Compare performance metrics
   - Generate comparative visualizations

### Running the Notebook

**Option A: Run all cells at once**
- Click **Kernel** → **Restart & Run All**
- Wait for all cells to complete (this may take several minutes)

**Option B: Run cells step-by-step**
- Click the first code cell
- Press `Shift + Enter` to run and move to the next cell
- Review outputs and visualizations after each cell

### Expected Outputs

The notebook will generate:
- **Decision tree visualizations** showing the classification logic
- **Confusion matrices** for prediction evaluation
- **Performance metrics** (accuracy, precision, recall, F1-score)
- **Recommendation analysis** with trace-specific suggestions
- **Prefix length comparison plots**

---  
**Authors**:
- Davide Donà - [GitHub](https://github.com/446f6e6e79) - [Email](mailto:davidedona03@gmail.com)
- Andrea Blushi - [GitHub](https://github.com/andreablushi) - [Email](mailto:andreablushi@gmail.com)

**Course**: Process Mining and Management, University of Trento 