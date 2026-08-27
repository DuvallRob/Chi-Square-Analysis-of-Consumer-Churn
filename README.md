# Chi Square Analysis of Consumer Churn

## Project Overview
This repository contains a hybrid R and Python data analysis pipeline designed to evaluate consumer churn[cite: 1]. The primary objective is to determine if financial institutions can utilize predictive modeling to estimate customer attrition and deploy targeted interventions based on consumer complaint product categories[cite: 1]. 

For a detailed breakdown of the statistical methods and findings, please refer to the included "Capstone_ Data Analysis Report.pdf".

## Research Question & Hypotheses
The core research question asks: Do consumer complaint product categories have a statistically significant association with customer churn signals in the CFPB Churn Signal Golden Set?[cite: 1]

* **Null Hypothesis (H0):** There is no statistically significant association between complaint product category and churn signal.[cite: 1]
* **Alternative Hypothesis (Ha):** There is a statistically significant association between complaint product category and churn signal.[cite: 1]
* **Decision Rule:** At a significance level of α=0.05, the null hypothesis is rejected when the p-value is less than 0.05.[cite: 1]

## Data Source & Preparation
* **Dataset:** The project utilizes the "CFPB Churn Signal Golden Set," containing 1,001 hand-verified consumer complaint narratives mapping categorical variables against a binary churn signal.[cite: 1]
* **Synthetic Expansion:** Due to severe initial class imbalance, the SMOTE synthetic data generation technique was applied.[cite: 1] Using a duplicate size of 135, the dataset was expanded from 1,001 rows to 7,481 rows (953 negative and 6,528 positive signals) prior to cross-validation splitting.[cite: 1]
* **NLP Processing:** Unstructured text narratives were transformed into numerical features via Hugging Face DistilBERT, generating a 1001 x 768 embedding matrix.[cite: 1]

## Methodology
The pipeline bridges Python's robust transformer tooling with R's statistical rigor[cite: 1]. The analysis relies on three primary techniques:

1. **Chi-Square Test of Independence:** Used to determine the statistical association between the categorical product type and the churn signal.[cite: 1]
2. **K-Means Clustering:** An unsupervised technique used to group complaint text patterns based on PCA-reduced DistilBERT embeddings.[cite: 1] Silhouette scoring identified K=8 as the optimal number of clusters for the training set.[cite: 1]
3. **Gradient Boosting (XGBoost):** Selected to classify the final churn signal, optimized via k-fold cross-validation.[cite: 1]

## Repository Structure
    ├── data\
    │   ├── golden_set.csv         # Original 1,001-row CFPB dataset
    │   └── cv_folds.csv           # Cross-validation mapping
    ├── python\
    │   ├── requirements.txt       # Python dependencies (e.g., Hugging Face)
    │   └── embed_narratives.py    # DistilBERT NLP embedding script
    ├── R\
    │   ├── 01_load_data.R         # Extracts and merges CSV data
    │   ├── 02_get_embeddings.R    # Bridges to Python via reticulate
    │   ├── 03_smote_expand.R      # Applies SMOTE for class imbalance
    │   ├── 04_chi_square.R        # Performs Chi-Square Test of Independence
    │   ├── 05_train_test_split.R  # Splits the 7,481-row expanded dataset
    │   ├── 06_pca.R               # Reduces dimensionality of embeddings
    │   ├── 07_kmeans_clustering.R # K-Means clustering (K=8)
    │   ├── 08_assemble_features.R # Prepares final model frame
    │   ├── 09_hyperparam_cv.R     # K-fold cross-validation
    │   ├── 10_final_train.R       # Trains the XGBoost model
    │   └── 11_evaluate.R          # Generates test metrics and confusion matrix
    ├── app.R                      # Interactive Shiny dashboard frontend
    ├── install_packages.R         # R environment setup
    ├── run_pipeline.R             # Master execution script
    └── Dockerfile                 # Containerization for reproducibility

## Results & Performance
The results of the analysis strongly support the alternative hypothesis:
* **Chi-Square Results:** The test yielded an X-squared output value of 437.93 with a p-value < 2.2e-16, falling well below the 0.05 threshold and rejecting the null hypothesis.[cite: 1]
* **XGBoost Metrics (Held-out Fold 4):** 
    * **Accuracy:** 0.9886[cite: 1]
    * **Precision:** 0.9870[cite: 1]
    * **Recall:** 1.00[cite: 1]
    * **F1 Score:** 0.9934[cite: 1]
    * **AUC:** 1.00[cite: 1]

## Usage & Deployment
To run the full end-to-end pipeline locally:
1. Ensure both R and Python environments are configured.
2. Install required Python packages via `pip install -r python/requirements.txt`.
3. Install required R packages via `Rscript install_packages.R`.
4. Execute the master pipeline script using `Rscript run_pipeline.R`.
5. To view the findings interactively, launch the Shiny application via `app.R`. The Shiny dashboard is recommended for customer service representatives to immediately flag and target high-risk complaints for retention interventions.[cite: 1]

## Docker Quickstart
To ensure complete reproducibility and avoid cross-language environment configuration issues between R and Python, you can run this project using Docker. 

### 1. Build the Docker Image
From the root directory of the project (where the `Dockerfile` is located), run the following command to build the image. This will install all necessary R packages, Python dependencies (including Hugging Face and PyTorch), and system libraries:
    docker build -t churn-analysis-pipeline .

### 2. Run the Full Pipeline
To execute the data preparation, SMOTE expansion, model training, and evaluation scripts (`run_pipeline.R`) completely inside the container:
    docker run --rm churn-analysis-pipeline Rscript run_pipeline.R

*(Note: The `--rm` flag automatically cleans up the container once the pipeline finishes executing).*

### 3. Launch the Shiny Dashboard
To run the interactive Shiny frontend (`app.R`) and expose it to your local machine, run:
    docker run --rm -p 3838:3838 churn-analysis-pipeline Rscript -e "shiny::runApp('app.R', host='0.0.0.0', port=3838)"

Once running, open your web browser and navigate to `http://localhost:3838` to view the interactive churn dashboard. *(If your `Dockerfile` exposes a different port by default, simply adjust the `3838:3838` mapping accordingly).*
