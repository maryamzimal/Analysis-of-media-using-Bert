#  Social Media Sentiment Analysis using BERT

## 📌 Project Overview
This project provides a complete, end-to-end implementation of Sentiment Analysis on social media text (specifically Twitter data) using **Bidirectional Encoder Representations from Transformers (BERT)**. The pipeline covers everything from data collection and social-media-specific text preprocessing to fine-tuning, benchmarking against traditional models, and generating evaluation graphs.

##  Key Features & Reviewer Justifications
Based on recent review feedback, this project has been optimized with the following implementations:

1. **Dynamic Padding & Truncation Optimization:** 
   Instead of static padding (which wastes GPU memory), we utilize Hugging Face's `DataCollatorWithPadding`. This dynamically pads sequences only to the maximum length within each specific batch. Truncation is set to `max_length=128`, which is optimal for tweets (280 characters).
2. **Class Imbalance Justification:**
   Social media datasets often suffer from class imbalance. To prevent the model from favoring the majority class, we calculate class weights using `sklearn` and pass them to a custom `WeightedTrainer` (using `CrossEntropyLoss`). We also prioritize **Weighted F1-Score** over standard accuracy for evaluation.
3. **Model Selection:**
   The pipeline supports both standard **`bert-base-uncased`** as a baseline and **`vinai/bertweet-base`** (BERTweet), which is specifically pre-trained on 850M tweets and uses a custom tokenizer optimized for social media slang, emojis, and hashtags.

## 📊 Dataset
We use the **TweetEval (Sentiment)** dataset, available directly via the Hugging Face `datasets` library. It contains ~45,000 tweets labeled into 3 classes:
* `0` = Negative
* `1` = Neutral
* `2` = Positive

## 📂 Project Structure
```text
bert-sentiment-project/
│
├── bert-sentiment/          # Saved fine-tuned BERT model & tokenizer
├── results/                 # Output folder for graphs, logs, and .npy files
│
├── train.py                 # Script to preprocess and fine-tune BERT
├── benchmark.py             # Script to evaluate BERT vs Baselines (TF-IDF, VADER)
├── graphs.py                # Script to generate ROC, Confusion Matrix, etc.
└── README.md
🛠️ Installation & Requirements
Ensure you have Python 3.8+ installed. Install the required dependencies:

bash
pip install -U transformers datasets torch accelerate scikit-learn emoji wordninja
pip install vaderSentiment matplotlib seaborn tqdm
🚀 How to Run
Step 1: Preprocess & Train the Model
Run the training script to clean the tweets and fine-tune BERT.

bash
python train.py
This will save the trained model weights and tokenizer into the ./bert-sentiment directory.

Step 2: Benchmark the Model
Evaluate the fine-tuned BERT against traditional baselines (TF-IDF + Logistic Regression, TF-IDF + SVM, and VADER lexicon).

bash
python benchmark.py
Outputs: Saves benchmark_table.csv and prediction arrays (.npy files) in the results/ folder.

Step 3: Generate Evaluation Graphs
Create visualizations for your paper or presentation.

bash
python graphs.py
Outputs: Generates graph1_model_comparison.png, graph2_confusion_matrix.png, graph4_roc_curves.png, and training curves in the results/ folder.

📈 Text Preprocessing Pipeline
Social media text is inherently noisy. Our clean_tweet() function handles:

URLs: Replaced with HTTPURL

Mentions: Replaced with @USER

Hashtags: Split into words (e.g., #GoodDay -> Good Day) using wordninja.

Emojis: Converted to descriptive text (e.g., 😍 -> smiling face) using the emoji library.

📊 Evaluation Metrics
The model is evaluated using:

Accuracy

Weighted Precision, Recall, and F1-Score (Crucial for imbalanced classes)

Confusion Matrix (To identify class confusion, e.g., Neutral vs. Positive)

ROC-AUC Curves (One-vs-Rest)
