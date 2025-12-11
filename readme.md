# Process Mining and Management Project

Implementation of **predictive process monitoring** and **prescriptive recommendations** using DecisionTrees with boolean encoding.

The project was developed as part of the course "Process Mining and Management", held by Prof. Chiara Di Francescomarino at the University of Trento, Master's Degree in Computer Science

[View Full Report (PDF)](docs/report.pdf)


## Table of Contents

* [1. Repository Structure](#1-repository-contents)
* [2. Installation and setup](#2-installation-and-setup)

## 1. Repository Contents

- **src/**:  Source code modules
   - **utils.py**: Core utilities (log processing, encoding, hyperparameter optimization)
   - **plotting.py**: Visualization functions (decision trees, confusion matrices, metrics)
- **logs/**: Event log  compressed files (XES format)
- **docs/**: Documentation and outputs
   - **media/**: Generated visualizations
   - **LaTeX/**: LaTeX report files
   - **report.pdf**: Final report of the project (PDF)
- **app.log**: Application log file
- **notebook.ipynb**: Main Jupyter Notebook for experiments
- **requirements.txt**: Python dependencies

## 2. Installation and setup
To install and set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/andreablushi/Process-Mining-Management-Project
   cd Process-Mining-Management-Project
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Extract the event logs from the `logs/` directory:
   ```bash
      unzip 'logs/*.zip' -d logs/
   ```
5. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
6. Open `notebook.ipynb` in your browser to start working with the project.