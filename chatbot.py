import cohere
from newspaper import Article
import requests
from bs4 import BeautifulSoup

# Initialize Cohere client with your API key
cohere_client = cohere.Client("136nQ8QaflaJnEheQ187ypmGaDz6vECXt7ruZAem")

def fetch_article_text(url):
    # Custom headers to mimic a real browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Send a request to the URL with custom headers
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = "\n".join([para.get_text() for para in paragraphs])
        return article_text
    else:
        raise Exception(f"Failed to fetch article: {response.status_code}")

url = input("Enter the news article URL: ")

try:
    article_text = fetch_article_text(url)
    print("Article fetched successfully.")
except Exception as e:
    print("Failed to fetch article:", e)

# Ask a question
question = input("\nAsk a question (or type 'exit' to quit): ")

if question.lower() == "exit":
    print("Exiting chatbot.")

prompt = f"Use the following news article to answer the question:\n\n{article_text[:8000]}\n\nQuestion: {question}"

response = cohere_client.generate(
    prompt=prompt,
    max_tokens=200,
    temperature=0.7
)

print("Chatbot:", response.generations[0].text.strip())
