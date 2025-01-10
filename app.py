
import requests
import csv
from bs4 import BeautifulSoup
import pandas as pd

# List to store course data
courses = []

# Define the base URL for the courses
base_url = 'https://courses.analyticsvidhya.com'

# Start with the first page of courses
pages = [
    base_url + '/courses',  # Page 1
    base_url + '/collections?page=2',  # Page 2
    base_url + '/collections?page=3',  # Page 3
    base_url + '/collections?page=4',  # Page 4
    base_url + '/collections?page=5',  # Page 5
    base_url + '/collections?page=6',  # Page 6
    base_url + '/collections?page=7',  # Page 7
    base_url + '/collections?page=8',  # Page 8
]

# Loop through each page in the list
for current_page_url in pages:
    print(f"Fetching courses from: {current_page_url}...")
    
    # Fetch the current page
    page = requests.get(current_page_url)
    soup = BeautifulSoup(page.text, 'html.parser')

    # Extracting course title, image, and course link from the current page
    for course_card in soup.find_all('header', class_='course-card__img-container'):
        img_tag = course_card.find('img', class_='course-card__img')
        
        if img_tag:
            title = img_tag.get('alt')
            image_url = img_tag.get('src')
            
            link_tag = course_card.find_previous('a')
            if link_tag:
                course_link = link_tag.get('href')
                if not course_link.startswith('http'):
                    course_link = base_url + course_link

                # Sending a request to the individual course page to get the description
                course_page = requests.get(course_link)
                course_soup = BeautifulSoup(course_page.text, 'html.parser')
                # Extracting the course description from the individual course page
                fr_view = course_soup.find('div', class_='fr-view')

                if fr_view:
                    description_paragraph = fr_view.find('p')
                    if description_paragraph:
                        description = description_paragraph.text.strip()
                    else:
                        description = 'No description available'
                else:
                    description = 'No description available'

                # Storing course details
                courses.append({
                    'title': title,
                    'image_url': image_url,
                    'course_link': course_link,
                    'description': description
                })

# Save to a DataFrame
df = pd.DataFrame(courses)
df.to_csv("courses2.csv", index=False)
print("Course data saved to courses.csv")


import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import gradio as gr
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

# Load course data
df = pd.read_csv("courses2.csv")
# Load pre-trained BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')


# def get_bert_embedding(text):
#     if not isinstance(text, str):  # Handle missing or invalid data
#         text = ""
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).numpy()


# # Combine title and description for embedding generation
# df['combined_text'] = df['title'] + " " + df['description']

# # Create embeddings for course titles and descriptions
# df['embedding'] = df['combined_text'].apply(lambda x: get_bert_embedding(x))

model = SentenceTransformer("all-mpnet-base-v2")
# Function to generate embeddings using SentenceTransformer
def get_sentence_embedding(text):
    if not isinstance(text, str):  # Handle missing or invalid data
        text = ""
    return model.encode(text, convert_to_tensor=True)

# Combine title and description for embedding generation
df['combined_text'] = df['title'] + " " + df['description']

# Create embeddings for course titles and descriptions
df['embedding'] = df['combined_text'].apply(lambda x: get_sentence_embedding(x))


# Function to perform search using BERT-based similarity
def search_courses(query):
    query_embedding = get_sentence_embedding(query).reshape(1, -1)
    course_embeddings = np.vstack(df['embedding'].values)
    
    # Compute cosine similarity between query embedding and course embeddings
    similarities = cosine_similarity(query_embedding, course_embeddings).flatten()
    
    # Add the similarity scores to the DataFrame
    df['score'] = similarities
    
    # Sort by similarity score in descending order and return top results
    top_results = df.sort_values(by='score', ascending=False).head(10)
    return top_results[['title', 'description', 'image_url', 'course_link', 'score']].to_dict(orient='records')


# Function to simulate autocomplete by updating search results live
def autocomplete(query):
    matching_courses = df[df['title'].str.contains(query, case=False, na=False)]
    return matching_courses['title'].tolist()[:3]  # Show top 3 matching course titles

def gradio_search(query):
    result_list = search_courses(query)
    
    if result_list:
        html_output = '<div class="results-container">'
        for item in result_list:
            course_title = item['title']
            course_description = item['description']
            course_image = item['image_url']
            course_link = item['course_link']
            relevance_score = round(item['score'] * 100, 2)
            
            html_output += f'''
            <div class="course-card">
                <img src="{course_image}" alt="{course_title}" class="course-image"/>
                <div class="course-info">
                    <h3>{course_title}</h3>
                    <p>{course_description}</p>
                    <p>Relevance: {relevance_score}%</p>
                    <a href="{course_link}" target="_blank" class="course-link">View Course</a>
                </div>
            </div>'''
        html_output += '</div>'
        return html_output
    else:
        return '<p class="no-results">No results found.</p>'

# Custom CSS for the Gradio interface
custom_css = """
body {
    font-family: Arial, sans-serif;
    background-color: #2f3435;
}
.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
.results-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
}
.course-card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
    overflow: hidden;
    width: 48%;
    transition: transform 0.2s;
}
.course-card:hover {
    transform: translateY(-5px);
}
.course-image {
    width: 100%;
    height: 150px;
    object-fit: cover;
}
.course-info {
    padding: 15px;
}
.course-info h3 {
    margin-top: 0;
    font-size: 18px;
    color: #2b0d0d;
}
.course-info p {
    color: #666;
    font-size: 14px;
    margin-bottom: 10px;
}
.course-link {
    display: inline-block;
    background-color: #007bff;
    color: white;
    padding: 8px 12px;
    text-decoration: none;
    border-radius: 4px;
    font-size: 14px;
    transition: background-color 0.2s;
}
.course-link:hover {
    background-color: #0056b3;
}
.no-results {
    text-align: center;
    color: #666;
    font-style: italic;
}
"""

# Gradio interface
iface = gr.Interface(
    fn=gradio_search,
    inputs=gr.Textbox(label="Enter your search query", placeholder="e.g., machine learning, data science, python"),
    outputs=gr.HTML(label="Search Results"),
    title="Analytics Vidhya Smart Course Search",
    description="Find the most relevant courses from Analytics Vidhya based on your query.",
    theme="huggingface",
    css=custom_css,
    examples=[
        ["machine learning for beginners"],
        ["advanced data visualization techniques"],
        ["python programming basics"], 
        ["Business Analytics"]
    ]
)

if __name__ == "__main__":
    iface.launch()