## README

This repo is the code for reproducing the paper：Domain Generalization and Adaptation Using Low Rank Exemplar SVMs’ in ARIN7404A, 2025, HKU.

The code is implemented by Python, and the models are saved in 'model' directory.

To run the code, you should install the requirements by `pip install -r requirements.txt`

If you want to run the project, you can open and run the 'Main.py' for office_caltech_10 dataset, or 'Main_ixams.py' for ixams dataset.

The baseline model we use in our project is SVM, and you can run the code in 'baseline.py' or 'baseline_ixams.py'.

If you want to train the models by yourselves, you can simply set the 'save_model_flag' to 1 (for Main.py).

