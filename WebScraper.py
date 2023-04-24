import requests
from bs4 import BeautifulSoup
import re

def get_soup(url):
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup
    else:
        print(f"Error {response.status_code}: Unable to access {url}")
        return None

def get_subpages_urls(soup, base_url):
    subpages = set()
    for link in soup.find_all('a', href=True):
        href = link['href']
        if not href.startswith('http'):
            href = base_url + href
        subpages.add(href)
    return subpages

def scrape_text(soup):
    text = ''
    for paragraph in soup.find_all('p'):
        text += paragraph.get_text() + '\n'
    return text

def main():
    base_url = 'https://example.com/'  # Replace with the target website URL
    main_page_soup = get_soup(base_url)

    if main_page_soup is not None:
        subpages = get_subpages_urls(main_page_soup, base_url)

        for subpage_url in subpages:
            subpage_soup = get_soup(subpage_url)
            if subpage_soup is not None:
                text = scrape_text(subpage_soup)
                print(f"Text from {subpage_url}:\n{text}")