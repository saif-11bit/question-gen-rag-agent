import streamlit as st
import requests
import json
from newspaper import Article, ArticleException
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
# from dotenv import load_dotenv

# Load environment variables from .env file
# load_dotenv()

# Retrieve API keys from environment variables
# open_ai_api_key = os.getenv("OPENAI_API_KEY")
# google_api_key = os.getenv("GOOGLE_API_KEY")
# google_cx = os.getenv("GOOGLE_CX")



# Function to load the SentenceTransformer model once
@st.cache_resource
def load_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

# Function to extract JSON from the generated text
def to_json(string):
    string = string.strip().replace('<json>', '').replace('</json>', '')
    start_index = string.find('{')
    end_index = string.rfind('}')

    if start_index == -1 or end_index == -1:
        st.error("Failed to extract JSON from the response.")
        return None

    json_string = string[start_index:end_index+1]

    try:
        data = json.loads(json_string, strict=False)
        return data
    except json.JSONDecodeError:
        st.error("Invalid JSON format.")
        return None

# Function to fetch articles using Google Custom Search API
def fetch_articles(query, google_api_key, google_cx):
    articles = []
    payload = {
        "key": google_api_key,
        "q": f"{query}",
        "cx": google_cx,
        "start": 1,
        "num": 10
    }
    try:
        resp = requests.get("https://customsearch.googleapis.com/customsearch/v1", params=payload)
        resp.raise_for_status()
        resp_json = resp.json()
        resp_links = [i["link"] for i in resp_json.get("items", [])]
        st.write(f"Found {len(resp_links)} articles.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching articles: {e}")
        return articles

    for link in resp_links:
        article = Article(link)
        try:
            article.download()
            article.parse()
            articles.append(article.text)
        except ArticleException:
            st.warning(f"Failed to parse article: {link}")
            continue
    return articles

# Function to split articles into chunks
def chunk_fetched_articles(articles, text_splitter):
    documents = []
    for article in articles:
        texts = text_splitter.split_text(article)
        documents.extend(texts)
    return documents

# Function to generate FAISS index from chunks
def generate_embedding_chunks(all_chunks, model):
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings_np = np.array(embeddings)

    embedding_dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)
    return index

# Function to retrieve relevant chunks based on query
def retrieve_chunks(index, query, all_chunks, model, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = [all_chunks[i] for i in indices[0]]
    return retrieved_chunks

# Function to generate questions using OpenAI's GPT-4
def generate_questions(retrieved_chunks, prompt_template, open_ai_api_key):
    headers = {
        "Authorization": f"Bearer {open_ai_api_key}",
        "Content-Type": "application/json"
    }
    context_string = "\n".join(retrieved_chunks)
    data = {
        "model": "gpt-4",
        'temperature':0.2,
        "messages":[
            {"role": "user", "content": "Here is the context: " + context_string},
            {"role": "user", "content": prompt_template},
        ],
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        generated_text = response_json['choices'][0]['message']['content']
        return generated_text
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating questions: {e}")
        return None
    except KeyError:
        st.error("Unexpected response format from OpenAI.")
        return None

# RAG Pipeline integrating all steps
def rag_pipeline(query, google_api_key, google_cx, open_ai_api_key, model, text_splitter):
    articles = fetch_articles(query, google_api_key, google_cx)
    if not articles:
        st.warning("No articles fetched. Please check your query and API keys.")
        return None

    all_chunks = chunk_fetched_articles(articles, text_splitter)
    st.write(f"Total chunks: {len(all_chunks)}")

    with st.spinner("Generating embeddings and building index..."):
        index = generate_embedding_chunks(all_chunks, model)

    retrieved_chunks = retrieve_chunks(index, query, all_chunks, model, k=5)
    st.write("Retrieved relevant chunks from articles.")

    prompt_template = f"""
You are an expert in generating insightful and relevant interview questions. Your task is to create 10 high-quality interview questions related to {{query}} based on the provided context. Follow these steps:

1. Carefully read and analyze the context provided below:
{{context}}

2. Identify key themes, concepts, and skills related to {{query}} from the context.

3. Formulate 10 interview questions that:
   - Are directly relevant to {{query}}
   - Cover a range of difficulty levels (easy, medium, hard)
   - Include a mix of question types (e.g., technical, behavioral, problem-solving)
   - Are clear, concise, and unambiguous

4. Review your questions to ensure they are unique and non-repetitive.

An example json should look like this.
<json>
{{
    "questions": [
        {{
            "question_number": 1,
            "question": "First interview question here",
            "difficulty": "easy|medium|hard",
            "type": "technical|behavioral|problem-solving"
        }},
        {{
            "question_number": 2,
            "question": "Second interview question here",
            "difficulty": "easy|medium|hard",
            "type": "technical|behavioral|problem-solving"
        }},
        // Add additional questions up to number 10
        {{
            "question_number": 10,
            "question": "Tenth interview question here",
            "difficulty": "easy|medium|hard",
            "type": "technical|behavioral|problem-solving"
        }}
    ]
}}
</json>
Ensure that your response contains exactly 10 questions and follows the specified JSON format. Wrap the JSON in <json> tags.
"""

    with st.spinner("Generating interview questions..."):
        questions = generate_questions(retrieved_chunks, prompt_template, open_ai_api_key)
        if not questions:
            return None

    response = to_json(questions)
    return response

# Main Streamlit App
def main():
    st.set_page_config(page_title="Interview Questions Generator", layout="wide")
    st.title("📝 Interview Questions Generator")

    st.markdown("""
    This application fetches relevant articles based on your query, processes them, and generates insightful interview questions using OpenAI's GPT-4.
    """)

    st.header("Generate Interview Questions")
    query = st.text_input("Enter your query:", value="sales interview questions")

    if st.button("Generate Questions"):
        try:
            open_ai_api_key = st.secrets["OPENAI_API_KEY"]
            google_api_key = st.secrets["GOOGLE_API_KEY"]
            google_cx = st.secrets["GOOGLE_CX"]
        except KeyError as e:
            st.error(f"Missing secret key: {e}. Please check your secrets configuration.")
            st.stop()
        model = load_model()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        with st.spinner("Running pipeline..."):
            response = rag_pipeline(query, google_api_key, google_cx, open_ai_api_key, model, text_splitter)

        if response:
            st.success("Interview questions generated successfully!")

            st.subheader("Generated Questions (JSON)")
            st.json(response)

            st.subheader("Formatted Questions")
            for q in response.get("questions", []):
                st.markdown(f"**Question {q['question_number']}** ({q['difficulty'].capitalize()}, {q['type'].capitalize()}): {q['question']}")

if __name__ == "__main__":
    main()
