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

load_dotenv()

# Retrieve API keys from environment variables
open_ai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cx = os.getenv("GOOGLE_CX")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)


def to_json(string):
    string = string.strip().replace('<json>', '').replace('</json>', '')
    start_index = string.find('{')
    end_index = string.rfind('}')

    # Extract the JSON substring
    json_string = string[start_index:end_index+1]

    # Parse the JSON
    data = json.loads(json_string,strict=False)
    return data

def fetch_articles(query):
    articles = []
    payload = {
        "key": google_api_key,
        "q": f"{query}",
        "cx": google_cx,
        "start": 1,
        "num": 10
    }
    resp = requests.get("https://customsearch.googleapis.com/customsearch/v1", params=payload)
    print(resp)
    resp_links = [i["link"] for i in resp.json()["items"]]
    print(resp_links)
    for i in resp_links:
        article = Article(i)
        try:
            article.download()
            article.parse()
        except ArticleException:
            continue
        article_text = article.text
        articles.append(article_text)
    return articles


def chunk_fetched_articles(articles):
    documents = []
    for article in articles:
        texts = text_splitter.split_text(article)
        documents.extend(texts)
    return documents



def generate_embedding_chunks(all_chunks):
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings_np = np.array(embeddings)

    embedding_dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)
    return index

# Step 4: Implement the retriever
def retrieve_chunks(index, query, all_chunks ,k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = [all_chunks[i] for i in indices[0]]
    return retrieved_chunks



def generate_questions(retrieved_chunks, prompt_template):
    headers = {
        "Authorization": f"Bearer {open_ai_api_key}",
        "Content-Type": "application/json"
    }
    context_string = "\n".join(retrieved_chunks)
    data = {
        "model": "gpt-4o",
        'temperature':0.2,
        "messages":[
            {"role": "user", "content": "Here is the context: " + context_string},
            {"role": "user", "content": prompt_template},
        ],
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data).json()
    # Extract the assistant's reply
    generated_text = response['choices'][0]['message']['content']
    return generated_text

def rag_pipeline(query):
    articles = fetch_articles(query)
    all_chunks = chunk_fetched_articles(articles)
    print(f"Total chunks: {len(all_chunks)}")
    index = generate_embedding_chunks(all_chunks)
    retrieved_chunks = retrieve_chunks(index, query, all_chunks, k=5)
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
Ensure that your response contains exactly 10 questions and follows the specified JSON format.  Wrap the JSON in <json> tags.
"""
    questions = generate_questions(retrieved_chunks, prompt_template)
    return questions

def main():
    query = "sales interview questions"
    extracted_questions = rag_pipeline(query)

    response = to_json(extracted_questions)
    return response