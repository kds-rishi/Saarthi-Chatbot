import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import pandas as pd
import time

start_url = "https://keydynamicssolutions.com/"
domain = "keydynamicssolutions.com"
max_pages = 3 #preventing infinite crawling
delay = 1 #seconds

visited = set()
to_visit = [start_url]
cleaned_data = []


def clean_text(html):
    """Extract and clean meaningful text from HTML."""
    soup = BeautifulSoup(html, "lxml")

    # Remove <script> and <style> tags (but keep footer since user wants it)
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    # Extract text from selected tags
    parts = []
    for tag in soup.find_all(["p", "h1", "h2", "h3", "li", "span"]):
        txt = tag.get_text(" ", strip=True)
        if txt:
            parts.append(txt)

    # Join everything into a single string
    text = " ".join(parts)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def scrape_page(url):
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        print(f"Response from request : {response}")
    except Exception as e:
        print(f"\n\nScraping Failed at {url}: {e}")
        return None, []
    
    text = clean_text(response.text)

    # Extract links to follow
    soup = BeautifulSoup(response.text, "lxml")
    links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]

    return text, links


def text_scraper():
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()

        if url in visited or domain not in url:
            continue

        print(f"Scraping URL : {url}")
        text, links = scrape_page(url)

        visited.add(url)

        if text:
            cleaned_data.append({"url":url, "content": text})

        for link in links:
            if link not in visited and domain in link:
                to_visit.append(link)
        
        time.sleep(delay) # for polite crawling


    return cleaned_data


# df = pd.DataFrame(cleaned_data)
# df.to_csv("cleaned_scraped_data.csv", index=False, encoding="utf-8")
# print(f"Saved {len(df)} pages to cleaned_scraped_data.csv")

