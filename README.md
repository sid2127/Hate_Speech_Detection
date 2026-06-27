Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer  #feature extraction
from sklearn.model_selection import train_test_split    #splitting dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB           #classifier models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  #model accurracy


# New Section



## Reading dataset



df_jigsaw = pd.read_csv('train.csv.zip')
df_yt = pd.read_csv('archive (2).zip')
df_hinglish = pd.read_csv("combined_hate_speech_dataset.csv")


cols = [
    "IsToxic","IsAbusive","IsThreat","IsProvocative","IsObscene",
    "IsHatespeech","IsRacist","IsSexist","IsHomophobic",
    "IsReligiousHate","IsRadicalism","IsNationalist"
]

df_yt[cols] = df_yt[cols].astype(int)

df_yt["label"] = df_yt[cols].max(axis=1)

df_yt.rename(columns={"Text": "tweet"}, inplace=True)

df_yt = df_yt[["tweet", "label"]]

df_yt = df_yt.dropna(subset=["tweet"])
df_yt = df_yt[df_yt["tweet"].str.strip() != ""]

df_yt.reset_index(drop=True, inplace=True)

df_yt.head()



# 2. Create Binary Label
# =========================
cols = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

df_jigsaw["label"] = df_jigsaw[cols].max(axis=1)

# 3. Rename Column

df_jigsaw.rename(columns={"comment_text": "tweet"}, inplace=True)

# =========================
# 4. Keep Required Columns
# =========================
df_jigsaw = df_jigsaw[["tweet", "label"]]

# =========================
# 5. Remove Null / Empty Rows
# =========================
df_jigsaw = df_jigsaw.dropna(subset=["tweet"])
df_jigsaw = df_jigsaw[df_jigsaw["tweet"].str.strip() != ""]

# =========================
# 6. Reset Index
# =========================
df_jigsaw.reset_index(drop=True, inplace=True)

# =========================
# 7. Check Output
# =========================

df_jigsaw.head()

df_hinglish.head()

df_hinglish = df_hinglish[["text", "hate_label"]]

df_hinglish.rename(columns={
    "text": "tweet",
    "hate_label": "label"
}, inplace=True)

df_hinglish = df_hinglish.dropna()

df_hinglish.reset_index(drop=True, inplace=True)


df_hinglish.head()

###Merge Both Datasets

df_final = pd.concat(
    [df_jigsaw, df_yt, df_hinglish],
    ignore_index=True
)

# Shuffle
df_final = df_final.sample(
    frac=1,
    random_state=42
)

# Reset index for df_final
df_final.reset_index(drop=True, inplace=True)

# Check counts
print(df_final["label"].value_counts())

df_final[df_final['label'] == 0].count()

df_final[df_final['label'] == 1].count()

##OverSampling(Data Level Fix)

from sklearn.utils import resample

df_majority = df_final[df_final.label == 0]
df_minority = df_final[df_final.label == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42)

df_balanced.head()

## Normalization



import re

def clean_text(t):
    t = str(t)

    t = re.sub(r"http\S+|www\S+", "", t)   # remove URLs
    t = re.sub(r"@\w+", "", t)             # remove mentions
    t = re.sub(r"#\w+", "", t)             # remove hashtags
    t = re.sub(r'[^\x00-\x7F]+', '', t)    # remove weird encoding
    t = re.sub(r"[^a-zA-Z0-9 ]", " ", t)   # remove special chars
    t = re.sub(r'\s+', ' ', t)             # remove extra spaces

    return t.lower().strip()


# Apply on correct dataframe
df_balanced["clean_text"] = df_balanced["tweet"].apply(clean_text)

# Remove empty rows after cleaning
df_balanced = df_balanced[df_balanced["clean_text"].str.strip() != ""]

df_balanced.head()

## Features Extraction

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)

# Use correct dataframe
X = tfidf_vectorizer.fit_transform(df_balanced["clean_text"])
y = df_balanced["label"]

print(X.shape)
print(y.shape)

print(tfidf_vectorizer.get_feature_names_out()[:20])


## Spilt Datasets(Train & Test)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)


# Model Training

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(class_weight='balanced')
}

results = []

for name, model in models.items():
    print(f"\n========== {name} ==========")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))

    print("Accuracy:", round(acc, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
results_df = results_df.sort_values(by="Accuracy", ascending=False)

print("\n===== Final Model Comparison =====")
print(results_df)

# Confusion Matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

labels = [0, 1]
names = ['Clean (0)', 'Toxic (1)']

# Create figure
plt.figure(figsize=(18, 5))

# Loop through models
for i, (name, model) in enumerate(models.items(), 1):

    # Predict
    y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Create subplot
    plt.subplot(1, len(models), i)

    # Heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=names,
        yticklabels=names
    )

    plt.title(f'{name}\nConfusion Matrix')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

# Adjust spacing
plt.tight_layout()

# Show plots
plt.show()

###General test

from scipy.special import expit
import numpy as np
import pandas as pd

# =========================
# Confidence Function
# =========================
def get_confidence(model, vec, pred):

    # Models with predict_proba
    if hasattr(model, "predict_proba"):

        prob = model.predict_proba(vec)[0]

        return float(prob[pred])

    # Models like LinearSVC
    elif hasattr(model, "decision_function"):

        score = model.decision_function(vec)

        score = float(score[0]) if len(score.shape) == 1 else float(score[0][pred])

        prob = expit(score)

        return prob if pred == 1 else 1 - prob

    return None


# =========================
# Final Testing Function
# =========================
def test_all_models(text):

    cleaned = clean_text(text)

    vec = tfidf_vectorizer.transform([cleaned])

    results = []

    for name, model in models.items():

        # Prediction
        pred = int(model.predict(vec)[0])

        # Confidence
        conf = get_confidence(model, vec, pred)

        conf_percent = round(conf * 100, 2)

        # =========================
        # Final Logic
        # =========================
        if pred == 1 and conf_percent >= 60:

            final_prediction = "Toxic 🚨"

        else:

            final_prediction = "Clean ✅"

        results.append({

            "Model": name,

            "Input": text,

            "Cleaned Text": cleaned,

            "Prediction": final_prediction,

            "Confidence": conf_percent
        })

    return pd.DataFrame(results)


test_all_models("sid")



good_comments = [

    "You did a great job",

    "I really liked your article",

    "This video was very helpful",

    "Amazing content bro",

    "Thank you for sharing this",

    "This is a killer product",

    "I respect your opinion",

    "You are very talented",

    "This blog is informative",

    "I love this community"
]

mid_comments = [

    "I don't like your attitude",

    "This video is boring",

    "You talk too much",

    "There is something wrong with you",

    "Your explanation is confusing",

    "I expected better from you",

    "You sound annoying sometimes",

    "This content is not good",

    "I disagree with your opinion",

    "You should improve your work"
]

bad_comments = [

    "You are stupid",

    "I hate you",

    "Shut up idiot",

    "You are worthless",

    "I will kill you",

    "You are a dumb person",

    "Nobody likes you",

    "Go to hell",

    "You are disgusting",

    "You racist idiot"
]



all_tests = {
    "GOOD COMMENTS": good_comments,
    "MID COMMENTS": mid_comments,
    "TOXIC COMMENTS": bad_comments
}

for category, comments in all_tests.items():

    print("\n")
    print("=" * 60)
    print(category)
    print("=" * 60)

    for text in comments:

        print("\nINPUT:", text)

        display(test_all_models(text))

import joblib

# =========================
# Select Best Model
# =========================
best_model = models["Logistic Regression"]

# =========================
# Save TF-IDF Vectorizer
# =========================
joblib.dump(
    tfidf_vectorizer,
    "tfidf_vectorizer.pkl"
)

# =========================
# Save Logistic Regression Model
# =========================
joblib.dump(
    best_model,
    "logistic_regression_model.pkl"
)

print("Logistic Regression model saved successfully ✅")

# Downloading Offline Mode

import joblib

#Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

#Save the models
for name, model in models.items():
    filename = f'{name.lower().replace(" " , "_")}_model.joblib'
    joblib.dump(model, filename)
