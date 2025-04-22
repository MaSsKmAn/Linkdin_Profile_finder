import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time
import itertools

# Search LinkedIn URLs via DuckDuckGo
def get_linkedin_profiles(name, max_pages=2):
    query = f'"{name}" site:linkedin.com/in'
    base_url = 'https://html.duckduckgo.com/html/'
    headers = {"User-Agent": "Mozilla/5.0"}
    linkedin_links = set()
    current_page = 0
    params = {'q': query}

    while current_page < max_pages:
        response = requests.post(base_url, headers=headers, data=params)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('a', href=True)

        for result in results:
            href = result['href']
            if 'linkedin.com/in/' in href and name.lower() in result.get_text().lower():
                linkedin_links.add(href)

        next_form = soup.find('form', class_='results_links_more')
        if next_form:
            params = {inp['name']: inp.get('value', '') for inp in next_form.find_all('input')}
            current_page += 1
            time.sleep(1)
        else:
            break

    return list(linkedin_links)

# Rotate user-agent headers
user_agents = [
    "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)",
    "LinkedInBot/1.0",
    "Twitterbot/1.0",
    "facebookexternalhit/1.1",
    "WhatsApp/2.0",
    "Googlebot/2.1 (+http://www.google.com/bot.html)"
]
user_agent_cycle = itertools.cycle(user_agents)
def mimic_bot_headers():
    return next(user_agent_cycle)

# Fetch LinkedIn raw HTML
class LinkedInRawFetcher:
    def __init__(self):
        self.proxies = None  # Disable proxy for Streamlit simplicity

    def fetch(self, url: str) -> str:
        for _ in range(3):
            headers = {"User-Agent": mimic_bot_headers()}
            try:
                response = requests.get(url, headers=headers, proxies=self.proxies)
                if response.status_code == 200:
                    return response.text
            except Exception as e:
                print(f"Request failed for {url}: {e}")
        return ""

def get_interaction_count(statistics, target_type):
    for x in statistics:
        interaction_type = x.get("interactionType")
        interaction_id = interaction_type.get("@id") if isinstance(interaction_type, dict) else interaction_type
        if interaction_id == target_type:
            return x.get("userInteractionCount")
    return None

# Extract structured info from LinkedIn HTML
def process_linkedin_url(url, fetcher):
    raw_html = fetcher.fetch(url)
    if not raw_html:
        return pd.DataFrame()

    soup = BeautifulSoup(raw_html, 'html.parser')
    ld_json_script = soup.find("script", {"type": "application/ld+json"})
    ld_data = json.loads(ld_json_script.string) if ld_json_script and ld_json_script.string else {}

    main_entity = ld_data.get("mainEntity", {})
    statistics = main_entity.get("interactionStatistic", [])

    cleaned_data = {
        "profile_url": url,
        "full_name": main_entity.get("name"),
        "profile_image": main_entity.get("image", {}).get("contentUrl"),
        "headline": main_entity.get("description"),
        "location_city": main_entity.get("address", {}).get("addressLocality"),
        "location_country": main_entity.get("address", {}).get("addressCountry"),
        "connections": get_interaction_count(statistics, "https://schema.org/BefriendAction"),
        "followers": get_interaction_count(statistics, "https://schema.org/FollowAction"),
    }

    return pd.DataFrame([cleaned_data])

# Streamlit UI
st.set_page_config(page_title="🔍 LinkedIn Profile Finder", layout="wide")

st.title("🔍 LinkedIn Profile Finder")
name_input = st.text_input("Enter a person's name to search LinkedIn profiles:", "")

if st.button("Search"):
    if name_input.strip() == "":
        st.warning("Please enter a name.")
    else:
        with st.spinner("Searching LinkedIn profiles..."):
            profile_urls = get_linkedin_profiles(name_input)
            fetcher = LinkedInRawFetcher()
            all_data = []

            for url in profile_urls:
                df = process_linkedin_url(url, fetcher)
                if not df.empty:
                    all_data.append(df)

            if all_data:
                final_df = pd.concat(all_data, ignore_index=True)
                final_df.drop_duplicates(subset=["profile_url", "full_name"], inplace=True)

                st.success(f"Found {len(final_df)} profiles!")
                st.dataframe(final_df)

                csv = final_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download as CSV", data=csv, file_name="linkedin_profiles.csv", mime="text/csv")
            else:
                st.warning("No profiles could be parsed.")

