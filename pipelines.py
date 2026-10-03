from transformers import pipeline
import pandas as pd
import os

os.makedirs("dataset", exist_ok=True)

# =========================================================
# LOAD ARTICLE
# =========================================================

with open("dataset/article_text.txt", "r", encoding="utf-8") as file:
    article_text = file.read()

print("Article loaded successfully!")
print("Word count:", len(article_text.split()))

# Use a safe-length portion for models with token limits
model_text = article_text[:1800]

# =========================================================
# 1. SENTIMENT ANALYSIS
# =========================================================

sentiment_pipeline = pipeline(
    "sentiment-analysis"
)

sentiment_result = sentiment_pipeline(model_text)

print("\nSentiment Result:")
print(sentiment_result)

sentiment_df = pd.DataFrame([{
    "Original Text": article_text,
    "Sentiment": sentiment_result[0]["label"],
    "Score": sentiment_result[0]["score"]
}])

sentiment_df.to_csv(
    "dataset/home_sentiment.csv",
    index=False
)

print("Sentiment result saved!")

# =========================================================
# 2. SUMMARIZATION - DIFFERENT MAX_LENGTH VALUES
# =========================================================

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

summary_50 = summarizer(
    model_text,
    max_length=50,
    min_length=20,
    do_sample=False
)[0]["summary_text"]

summary_100 = summarizer(
    model_text,
    max_length=100,
    min_length=40,
    do_sample=False
)[0]["summary_text"]

summary_150 = summarizer(
    model_text,
    max_length=150,
    min_length=60,
    do_sample=False
)[0]["summary_text"]

print("\nSummary - max_length 50:")
print(summary_50)

print("\nSummary - max_length 100:")
print(summary_100)

print("\nSummary - max_length 150:")
print(summary_150)

summary_df = pd.DataFrame([
    {"Max Length": 50, "Summary": summary_50},
    {"Max Length": 100, "Summary": summary_100},
    {"Max Length": 150, "Summary": summary_150}
])

summary_df.to_csv(
    "dataset/home_summaries_comparison.csv",
    index=False
)

print("Summary comparison saved!")

# =========================================================
# 3. TRANSLATION - FRENCH
# =========================================================

translator_fr = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-fr"
)

french_text = translator_fr(
    model_text,
    max_length=512
)[0]["translation_text"]

print("\nFrench Translation:")
print(french_text)

# =========================================================
# 4. TRANSLATION - SPANISH
# =========================================================

translator_es = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-es"
)

spanish_text = translator_es(
    model_text,
    max_length=512
)[0]["translation_text"]

print("\nSpanish Translation:")
print(spanish_text)

translation_df = pd.DataFrame([{
    "Original Text": article_text,
    "French Translation": french_text,
    "Spanish Translation": spanish_text
}])

translation_df.to_csv(
    "dataset/home_translations.csv",
    index=False
)

print("Translation results saved!")

# =========================================================
# 5. QUESTION ANSWERING - 5 QUESTIONS
# =========================================================

qa_pipeline = pipeline(
    "question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad"
)

questions = [
    "What is the main topic of the article?",
    "When was the article published?",
    "What technology is discussed in the article?",
    "What are reasoning models?",
    "What is the article about?"
]

qa_results = []

for question in questions:

    result = qa_pipeline(
        question=question,
        context=model_text
    )

    qa_results.append({
        "Question": question,
        "Answer": result["answer"],
        "Score": result["score"]
    })

    print("\nQuestion:", question)
    print("Answer:", result["answer"])
    print("Score:", result["score"])

qa_df = pd.DataFrame(qa_results)

qa_df.to_csv(
    "dataset/home_qa_results.csv",
    index=False
)

print("QA results saved!")

# =========================================================
# 6. FINAL COMPARISON TABLE
# =========================================================

comparison_df = pd.DataFrame([{
    "Original Text": article_text,
    "Summary (50)": summary_50,
    "Summary (100)": summary_100,
    "Summary (150)": summary_150,
    "French Translation": french_text,
    "Spanish Translation": spanish_text,
    "Sentiment": sentiment_result[0]["label"],
    "Sentiment Score": sentiment_result[0]["score"],
    "QA Answers": "Saved in home_qa_results.csv"
}])

comparison_df.to_csv(
    "dataset/home_comparison.csv",
    index=False
)

print("\nFinal comparison saved!")

print("\n===================================")
print("HOME ACTIVITY COMPLETED SUCCESSFULLY!")
print("===================================")