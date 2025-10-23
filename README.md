# ☕ Cafe Delight FAQ Chatbot

This is a simple, FAQ-based chatbot for a fictional cafe called "Cafe Delight." It's built using Streamlit for the user interface and scikit-learn's TF-IDF and Cosine Similarity for natural language processing to match user questions with the best possible answer from a predefined JSON file.

<img width="914" height="610" alt="image" src="https://github.com/user-attachments/assets/e8bd376c-511c-4c07-885f-89635c8a0648" />


*(Note: You will need to take a screenshot of your running application and save it as `screenshot.png` in your repository for the image above to display.)*

---

## Features

* **Web-based Chat Interface**: Built with Streamlit, providing a clean and interactive UI.
* **NLP-powered Matching**: Uses TF-IDF vectorization to understand the content of a user's question.
* **Smart Responses**: Employs Cosine Similarity to find the most relevant answer from the FAQ database.
* **Graceful Fallback**: If no question matches with a high-enough similarity (threshold set at 0.3), it provides a "Sorry, I couldn't find an answer" response.
* **Custom Styling**: Includes custom CSS to create a branded, chat-like appearance and hide default Streamlit UI elements.

---

## Technologies Used

* **Python**
* **Streamlit**: For the web application and user interface.
* **Scikit-learn**: For TF-IDF vectorization (`TfidfVectorizer`) and `cosine_similarity`.
* **NLTK (Natural Language Toolkit)**: For text preprocessing (tokenization and stopword removal).

---

## Setup and Installation

Follow these steps to get the chatbot running on your local machine.

### 1. Clone the Repository

```bash
https://github.com/huzaifaakmal/CodeAlpha_ChatbotForFAQs
