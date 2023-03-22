# NLP-Project: Scientific Research Papers Recommendation System using Content-Based Filtering
Flow of the Recommendation System code:
1. The necessary libraries are imported - pandas, Streamlit, CountVectorizer and TruncatedSVD from scikit-learn, cosine_similarity from metrics.pairwise, and rake_nltk and nltk for keyword extraction.

2. A CSV file containing the dataset is loaded into a Pandas DataFrame.

3. Keywords are extracted from the abstracts using RAKE (Rapid Automatic Keyword Extraction) from rake_nltk. RAKE is a simple algorithm for extracting keywords from text documents. For each abstract in the dataset, the RAKE algorithm extracts the keywords and returns them as a list.

4. The extracted keywords are then converted into strings.

5. CountVectorizer is used to vectorize the keywords. CountVectorizer is a scikit-learn class that is used to transform a list of strings into a matrix of token counts. In this case, it is used to transform the list of keyword strings into a sparse matrix of token counts.

6. TruncatedSVD is applied to the sparse matrix of token counts. TruncatedSVD is a scikit-learn class that performs truncated singular value decomposition on a sparse matrix. The output of this step is a reduced-dimensionality matrix that can be used to represent the keywords.

7. A function named "get_recommendations" is defined. This function takes a user input (in this case, the abstract of a research paper), the dataset, the reduced-dimensionality matrix of the keywords, and a number of recommendations to return. It vectorizes the user input abstract, reduces it to the same dimensionality as the research paper abstracts, calculates the cosine similarity between the user input and the research paper abstracts, and returns the indices of the most similar papers. Finally, it returns the top n recommendations.

8. Another function named "get_abst" is defined. This function takes the title of a research paper as input and returns its abstract.

9. A Streamlit app is defined. It displays a dropdown menu of research paper titles. When the user selects a title and clicks the "Submit" button, the app calls the "get_abst" function to retrieve the abstract of the selected paper. It then calls the "get_recommendations" function to retrieve the top 5 recommended papers based on the selected paper's abstract. Finally, it displays the top 5 recommended papers along with their titles and abstracts.
