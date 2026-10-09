# SMS Spam Classifier

An NLP classification project that detects spam SMS messages using TF-IDF text features and a tuned Linear SVM.

## Dataset
SMS Spam Collection: 5,572 messages labeled `ham` (normal) or `spam`. After removing 403 exact duplicates, 5,169 messages remained (about 13% spam, so the target is imbalanced).

## Exploratory Data Analysis
- **Class imbalance:** about 87% ham vs 13% spam. A model that always predicts "ham" would get ~87% accuracy, so accuracy alone is misleading. I evaluated with precision, recall and F1 on the spam class.
- **Message length:** spam messages averaged 138 characters vs 71 for ham, nearly double, likely because spam packs in offers, deadlines and contact details.
- **Duplicates:** spam was duplicated about twice as often as ham (~12.6% vs ~6%), consistent with spam templates being sent to many numbers. Duplicates were dropped so the same message could not appear in both train and test.

## Text Preprocessing
- Lowercased text and removed punctuation and digits
- Split into train (4,135) and test (1,034) sets with stratification to preserve the spam ratio
- Converted text to numbers with **TF-IDF** (English stopwords removed), giving 7,282 features
- Vectorizer was fit on training data only, to avoid data leakage

## Models Compared (5-fold Cross-Validation, F1 on spam class)

| Model | CV F1 |
|---|---|
| **Linear SVM** | **0.889** |
| Naive Bayes | 0.790 |
| Logistic Regression | 0.694 |

Linear SVM was tuned with GridSearchCV. Best parameter: `C=10`, tuned CV F1: 0.892.

## Final Test Set Results

| Metric (spam class) | Score |
|---|---|
| Precision | 0.94 |
| Recall | 0.88 |
| F1 | 0.91 |
| Overall accuracy | 0.98 |

Confusion matrix: 896 ham correctly kept, 115 spam caught, 7 real messages wrongly flagged as spam, 16 spam messages missed.

Test F1 (0.91) is consistent with the cross-validation score (0.89), suggesting the result is stable and not a lucky split. Only 7 of 903 ham messages (under 1%) were wrongly flagged, which matters because blocking a real message is usually worse than letting a spam through.

## Sanity Checks on Custom Messages
The model correctly flagged "Congratulations! You won a free prize, claim now" as spam, and correctly kept "Are we meeting at 5 today?" and "I'm free tomorrow, let's meet" as ham, showing it does not rely on the word "free" alone.

## Limitations and What I'd Improve Next
- Text cleaning removed digits and symbols (phone numbers, prices, £ signs), which are strong spam signals. Adding features like "contains a number" or digit count could help.
- Try n-grams (word pairs) to capture phrases like "claim now"
- Tune the decision threshold to trade off missed spam against false positives
- Test on legitimate-but-spammy-looking messages (OTPs, delivery alerts)
- Try transformer models (e.g. BERT) as a stronger baseline
- Wrap the model in a Streamlit app for live predictions

## Tools Used
Python, Pandas, Scikit-learn, TF-IDF, Matplotlib, Seaborn
