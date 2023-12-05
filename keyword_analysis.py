import re
from bs4 import BeautifulSoup
from collections import Counter
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')


def is_valid_keyword(keyword):
    common_words = {'can', 'an', 'at', 'but', 'the', 'and', 'a', 'is', 'in', 'it', 'of', 'to', 'for', 'on', 'with', 'as', 'by', 'that', 'be', 'am', 'are'}
    # Split the phrase into individual words
    words = keyword.split()
    # Check if each word is alphabetic and not a common word
    valid_words = [word for word in words if word.isalpha() and word.lower() not in common_words]
    # The phrase is valid if it has the same number of valid words as the original
    return len(valid_words) == len(words)


def get_webpage_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Error occurred: {req_err}")


def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text


def get_first_n_words(text, n=50):
    words = text.split()[:n]
    return ' '.join(words) + '...'


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return words


def get_high_frequency_words(text, num=10):
    words = clean_text(text)
    words = [word for word in words if is_valid_keyword(word)]
    counter = Counter(words)
    most_common = counter.most_common(num)
    return most_common


def get_high_frequency_phrases(text, num=10):
    words = clean_text(text)
    # Construct candidate phrases from adjacent cleaned words
    phrases = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    # Filter out any non-valid phrases
    valid_phrases = [phrase for phrase in phrases if is_valid_keyword(phrase)]
    counter = Counter(valid_phrases)
    most_common = counter.most_common(num)
    return most_common


def get_lsi_keywords(text, num=10):
    documents = re.split(r'(?<=[.!?])\s*', text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    average_similarities = cosine_similarities.mean(axis=0)
    indices = average_similarities.argsort()[-num:]
    feature_names = vectorizer.get_feature_names_out()

    keywords = [feature_names[index] for index in indices[::-1] if is_valid_keyword(feature_names[index])]
    return keywords[:num]


def best_practices_for_blog(article_length):
    h1_count = 1
    h2_count = max(2, article_length // 500)
    h3_count = max(2, article_length // 250)
    h4_count = max(2, article_length // 125)
    image_count = max(2, article_length // 250)
    tips = [
        "Use descriptive and engaging title.",
        "Include target keyword in the title.",
        "Use subheadings to break the content.",
        "Include internal and external links.",
        "Optimize images with alt text containing target keyword.",
        "Use bullet points or numbered lists.",
        "Include a call to action.",
        "Keep paragraphs short and easy to read.",
        "Include social share buttons."
    ]
    return {
        "h1_count": h1_count,
        "h2_count": h2_count,
        "h3_count": h3_count,
        "h4_count": h4_count,
        "image_count": image_count,
        "tips": tips
    }


def main():
    num_urls = int(input("How many webpage URLs do you want to analyze (3 max)? "))
    while num_urls < 1 or num_urls > 3:
        print("Please enter a number between 1 and 3.")
        num_urls = int(input("How many webpage URLs do you want to analyze (3 max)? "))

    article_length = 0
    if num_urls > 1:
        article_length = max(int(input("Enter the expected length of your article in words (minimum 1000 words): ")), 1000)

    all_results = {}

    for _ in range(num_urls):
        url = input(f"Please enter webpage URL {_ + 1}: ")
        html_content = get_webpage_content(url)
        text = extract_text_from_html(html_content)

        lsi_keywords = get_lsi_keywords(text)
        high_freq_keywords = get_high_frequency_words(text)
        high_freq_phrases = get_high_frequency_phrases(text)

        all_results[text] = {
            'url': url,
            'preview': get_first_n_words(text),
            'lsi_keywords': lsi_keywords,
            'hf_keywords': high_freq_keywords,
            'hf_phrases': high_freq_phrases,
        }

    with open("kw-analysis.txt", "w", encoding='utf-8') as file:
        for text, results in all_results.items():
            file.write(f"Analysis for URL: {results['url']}\n")
            file.write(f"Text Preview: {results['preview']}\n\n")

            lsi_keywords_list = ", ".join(results['lsi_keywords'][:10])
            hf_keywords_list = ", ".join([f"{kw[0]} ({kw[1]})" for kw in results['hf_keywords'][:10]])
            hf_phrases_list = ", ".join([f"{phrase[0]} ({phrase[1]})" for phrase in results['hf_phrases'][:10]])

            file.write(f"LSI Keywords:\n{lsi_keywords_list}\n")
            file.write(f"\nHigh Frequency Keywords:\n{hf_keywords_list}\n")
            file.write(f"\nHigh Frequency Two-Word Phrases:\n{hf_phrases_list}\n")
            file.write("=" * 50 + "\n\n")  # Separator

    if num_urls >= 2:
        aggregate_lsi = Counter()
        aggregate_hf_normalized = Counter()
        aggregate_phrase_normalized = Counter()

        for text, results in all_results.items():
                total_words = len(text.split())
                for kw in results['lsi_keywords']:
                    aggregate_lsi[kw] += text.split().count(kw) / total_words
        for kw, freq in results['hf_keywords']:
                normalized_count = freq / total_words
                aggregate_hf_normalized[kw] += normalized_count
        for phrase, freq in results['hf_phrases']:
                normalized_count = freq / total_words
                aggregate_phrase_normalized[phrase] += normalized_count

        top_10_lsi = [item[0] for item in aggregate_lsi.most_common(10)]
        lsi_keywords_list = ", ".join(top_10_lsi)

        top_10_hf = aggregate_hf_normalized.most_common(10)
        hf_keywords_list = ", ".join([
            f"{kw[0]} ({max(1, int(kw[1] * article_length) - int(kw[1] * article_length * 0.1))}-{max(1, int(kw[1] * article_length) + int(kw[1] * article_length * 0.1))})"
            for kw in top_10_hf
        ])

        top_10_phrase = aggregate_phrase_normalized.most_common(10)
        hf_phrases_list = ", ".join([
            f"{phrase[0]} ({int(round(phrase[1] * article_length))})"
            for phrase in top_10_phrase
        ])


        with open("kw-analysis.txt", "a", encoding='utf-8') as file:
            file.write("Aggregated Analysis:\n")
            file.write(f"LSI Keywords:\n{lsi_keywords_list}\n")
            file.write(f"\nHigh Frequency Keywords:\n{hf_keywords_list}\n")
            file.write(f"\nHigh Frequency Two-Word Phrases:\n{hf_phrases_list}\n")
            file.write("=" * 50 + "\n\n")

        best_practices = best_practices_for_blog(article_length)
        with open("kw-analysis.txt", "a", encoding='utf-8') as file:
            file.write(f"Best Practices for {article_length}-word Blog Articles:\n")
            file.write(f"Expected H1 count: {best_practices['h1_count']}\n")
            file.write(f"Expected H2 count: {best_practices['h2_count']}\n")
            file.write(f"Expected H3 count: {best_practices['h3_count']}\n")
            file.write(f"Expected H4 count: {best_practices['h4_count']}\n")
            file.write(f"Expected Image count: {best_practices['image_count']}\n\n")

            file.write("Tips:\n")
            for tip in best_practices['tips']:
                file.write(f"- {tip}\n")


if __name__ == "__main__":
    main()
