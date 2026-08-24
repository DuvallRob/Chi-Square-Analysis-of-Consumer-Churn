# Shiny dashboard. Run from the project root: shiny::runApp(".")

library(shiny)
library(ggplot2)
library(dplyr)
library(xgboost)
library(pROC)
library(tibble)
library(scales)

# 1. FIXED: Correct file name for chi-square results
chi_results  <- readRDS("data/processed/chi_square_results_expanded.rds")
model_frame  <- readRDS("data/processed/model_frame.rds")
kmeans_fit   <- readRDS("data/processed/kmeans_fit.rds")
eval_results <- readRDS("data/processed/test_evaluation.rds")
pca_fit      <- readRDS("data/processed/pca_fit.rds")
final_model  <- xgb.load("models/final_xgb_model.model")
feature_cols <- readRDS("models/feature_cols.rds")
train_pcs    <- readRDS("data/processed/train_pcs.rds")

churn_by_product <- model_frame %>%
  group_by(product) %>%
  summarise(churn_rate = mean(churn_signal), n = n(), .groups = "drop")

ui <- navbarPage("Consumer Churn Signal — CFPB Complaints",
                 
                 tabPanel("Chi-Square: Product vs Churn",
                          fluidRow(
                            column(6, h4("Contingency Table"), tableOutput("contingency_table"),
                                   h4("Test Result"), verbatimTextOutput("chi_summary")),
                            column(6, plotOutput("churn_rate_plot"))
                          )
                 ),
                 
                 tabPanel("Complaint Clusters",
                          p(paste("K-Means on PCA-reduced DistilBERT embeddings, k =", kmeans_fit$k,
                                  "(selected by silhouette score)")),
                          plotOutput("cluster_plot", height = "500px")
                 ),
                 
                 tabPanel("Model Performance",
                          fluidRow(
                            column(6, h4("Held-out Test Metrics (fold 4)"),
                                   tableOutput("metrics_table"),
                                   h4("Confusion Matrix"), tableOutput("confusion_matrix")),
                            column(6, plotOutput("roc_plot"))
                          )
                 )
                 # 2. FIXED: Removed "Try a Narrative" tab which requires local Python/Reticulate
)

server <- function(input, output, session) {
  
  output$contingency_table <- renderTable({
    as.data.frame.matrix(chi_results$contingency_table) %>% rownames_to_column("product")
  })
  output$chi_summary <- renderPrint({ chi_results$chisq })
  
  output$churn_rate_plot <- renderPlot({
    ggplot(churn_by_product, aes(product, churn_rate, fill = product)) +
      geom_col() +
      geom_text(aes(label = percent(churn_rate, accuracy = 0.1)), vjust = -0.5) +
      scale_y_continuous(labels = percent) +
      labs(title = "Churn Rate by Product Category", x = NULL, y = "Churn Rate") +
      theme_minimal() + theme(legend.position = "none")
  })
  
  output$cluster_plot <- renderPlot({
    ggplot(train_pcs, aes(PC1, PC2, color = cluster)) +
      geom_point(alpha = 0.6, size = 2) +
      labs(title = "Complaint Narrative Clusters (Training Set)") +
      theme_minimal()
  })
  
  # 3. FIXED: Extracted metrics from the nested `full` list saved in 11_evaluate.R
  output$metrics_table <- renderTable({
    data.frame(Metric = c("Accuracy", "Precision", "Recall", "F1", "AUC"),
               Value = round(c(eval_results$full$accuracy, eval_results$full$precision,
                               eval_results$full$recall, eval_results$full$f1, eval_results$full$auc), 4))
  })
  
  output$confusion_matrix <- renderTable({
    as.data.frame.matrix(eval_results$full$cm) %>% rownames_to_column("predicted \\ actual")
  })
  
  output$roc_plot <- renderPlot({
    roc_obj <- roc(eval_results$full$actual, eval_results$full$pred_prob, quiet = TRUE)
    plot(roc_obj, main = paste("ROC Curve (AUC =", round(eval_results$full$auc, 3), ")"))
  })
}

shinyApp(ui, server)