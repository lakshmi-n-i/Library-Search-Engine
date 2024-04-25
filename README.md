# Library-Search-Engine
The library search engine project aims to recommend books to users based on their interests and preferences using machine learning and latent similarity indexing.

# Functionalities:

1) Text Preprocessing: The system employs advanced text preprocessing techniques to clean and tokenize book descriptions, ensuring accurate analysis of book content. It removes distractions such as single quotes, digits, and special characters, and proceeds to lemmatizes words to their base form for standardized representation.

2) Feature Extraction: Leveraging the Spacy library, the system extracts relevant features from book descriptions, including keywords, themes, and topics, to facilitate comprehensive analysis and retrieval of books based on user queries.

3) Model Training: The project utilizes the Gensim library to train machine learning models for text analysis, including TF-IDF and LSI models. These models enable the system to understand the underlying patterns and relationships within the book descriptions, enhancing the accuracy of search results.

4) Similarity Search: Using the trained models, the system performs similarity searches to recommend books that closely match the user's query. By analyzing the semantic similarity between book descriptions, the system identifies relevant titles and presents them to the user in order of relevance.

5) User Interface: The project features a user-friendly interface built with Flask and HTML, providing users with an intuitive platform to interact with the search engine. The interface includes input fields for users to enter search queries and displays search results in a visually appealing manner. The results include the title of the book, its description, front page image, and a relevance score.

6) Relevance Score: The search engine displays a relevance score alongside each recommended book, providing users with valuable insights into the similarity between their query and the suggested titles. This relevance score is calculated based on the semantic similarity between the user's search query and the content of each book description. A higher relevance score indicates a closer match between the user's interests and the content of the recommended book. This helps users make informed decisions about which books to explore further, enabling them to quickly identify titles that are most likely to meet their preferences and requirements.

7) Dynamic Book Recommendation: The system continuously learns from user interactions and dynamically adjusts its recommendations based on user feedback. This adaptive approach ensures that the search engine provides increasingly accurate and personalized book recommendations over time.

#Technology used
* Flask - A lightweight web application used to build web applications.
* Gensim - Python library used for topic modelling and document similarity analysis.
* Pickle - Used for serializing and deserializing Python objects. 
* Pandas - Used to manipulate and analyze structured data mainly used for data pre-processing and cleaning.
* Spacy - NLP library for Python which provides tools for NLP tasks such as tokenization used for text processing.
