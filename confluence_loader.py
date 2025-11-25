import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
from langchain_core.documents import Document  # Import Document from langchain

load_dotenv()

def fetch_confluence_pages():
    url = os.getenv("CONFLUENCE_URL")
    space_key = os.getenv("CONFLUENCE_SPACE")
    email = os.getenv("CONFLUENCE_EMAIL")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")

    if not all([url, space_key, email, api_token]):
        raise ValueError("Missing one or more Confluence environment variables.")

    pages = []
    start = 0
    limit = 25

    while True:
        response = requests.get(
            f"{url}?spaceKey={space_key}&expand=body.storage&limit={limit}&start={start}",
            auth=HTTPBasicAuth(email, api_token)
        )

        if response.status_code != 200:
            print("Failed to fetch from Confluence:", response.status_code, response.text)
            break

        data = response.json()
        results = data.get("results", [])
        if not results:
            break

        for page in results:
            title = page.get("title", "Untitled")
            html_content = page["body"]["storage"]["value"]

            # Convert HTML to plain text using BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            text_content = soup.get_text(separator="\n").strip()

            if text_content:
                # Convert to Document object
                doc = Document(
                    page_content=text_content,
                    metadata={
                        "title": title,
                        "source": f"Confluence:{title}",
                        "content_type": "Confluence"
                    }
                )
                pages.append(doc)

        start += limit

    print(f"Loaded {len(pages)} pages from Confluence")
    return pages
