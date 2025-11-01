import os
import requests
from bs4 import BeautifulSoup
from openai import AzureOpenAI

# Azure settings
AZURE_OPENAI_ENDPOINT = "https://<your-resource-name>.openai.azure.com/"
AZURE_OPENAI_API_KEY = "<your-api-key>"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o" 
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

def get_article_text(url: str) -> str:
    """Fetch and extract main article text from a webpage."""
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted elements
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Get main visible text
    text = " ".join(soup.stripped_strings)
    return text

def translate_text(text: str, target_language: str) -> str:
    """Use Azure OpenAI to translate text into the target language."""
    prompt = f"Translate the following text into {target_language}:\n\n{text}"

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()

def main():
    article_url = input("Enter the article URL: ").strip()
    target_lang = input("Enter the target language (e.g. 'Portuguese'): ").strip()

    print("\nFetching article...")
    article_text = get_article_text(article_url)

    print(f"\nTranslating article to {target_lang}...")
    translated_text = translate_text(article_text, target_lang)

    # Save translation to a file
    output_filename = f"translated_article_{target_lang.lower()}.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(translated_text)

    print(f"\n Translation saved to {output_filename}")

if __name__ == "__main__":
    main()
