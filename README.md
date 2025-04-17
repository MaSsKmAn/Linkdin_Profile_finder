# LinkedIn Profile Finder

This project is designed to analyze a JSON-based dataset containing a person's LinkedIn-related parameters and output the most relevant LinkedIn profile for them by calculating similarity metrics.

## Features

- **Preprocessing the Original Dataset**: 
  - The raw dataset is cleaned and transformed into an acceptable format for further analysis, ensuring compatibility with the models and algorithms used for profile matching.

- **Extracting LinkedIn Profiles**:
  - The system retrieves LinkedIn profile links from the dataset and pulls relevant details such as name, profile picture, and other professional attributes, ensuring precision in the data extracted.

- **Similarity Calculation**:
  - The project calculates the similarity between various parameters of the original dataset and LinkedIn profiles. This includes not only textual data (e.g., name, skills, experience) but also visual data (e.g., profile pictures) using image similarity techniques.

- **Profile Matching**:
  - Based on the computed similarity scores, the system ranks profiles and returns the most similar LinkedIn profile, helping in identifying candidates with the highest relevance to the input dataset.


## 🔄 Data Preprocessing for the original dataset

### 🔠 Name Cleaning
The name field undergoes two main preprocessing steps:

```python
name_cleaned = re.sub(r"\(.*?\)", "", entry.get("name", "")).strip()
formatted_name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name_cleaned)
```
- Remove Parentheses: Any information in brackets (like job titles or extra notes) is removed from the name.

  - Example: "John Doe (Manager)" → "John Doe"

- Add Space Between CamelCase: Automatically inserts a space between a lowercase and an adjacent uppercase letter.

  - Example: "johnDoe" → "john Doe"

### 🌍 Timezone to Country/State Extraction
The timezone field is parsed to extract the country and state:

```python

if entry.get("timezone") and "/" in entry["timezone"]:
    country, state = entry["timezone"].split("/")
else:
    country, state = "", ""
```

- Example: "US/California" becomes:
  - country = "US"
  - state = "California"

If the timezone is missing or malformed, both values are set as empty strings.

## 🚀 LinkedIn Profile Scraper – Main Highlights

This script extracts and cleans data from publicly visible LinkedIn profiles using DuckDuckGo search results and parses structured metadata from each profile page.

---

### 🔍 1. LinkedIn Profile Search
```python
get_linkedin_profiles(name)
```
* Performs a DuckDuckGo search for "{name}" site:linkedin.com/in.
* Collects up to max_pages of links containing 'linkedin.com/in/'.
* Filters links based on name match in search result text.
🔧 LinkedInRawFetcher Class
This class is responsible for fetching raw HTML content from a LinkedIn profile URL using proxy routing and rotating bot-mimicking headers to minimize blocking and detection.

⚙️ LinkedInRawFetcher

📍 __init__ Method
```python
def __init__(self):
    self.proxies = {
        "https": "http://<proxy_credentials>@brd.superproxy.io:33335",
        "http": "http://<proxy_credentials>@brd.superproxy.io:33335"
    }
```
- Sets up HTTP and HTTPS proxy credentials.

- Uses Bright Data (formerly Luminati) datacenter proxies for anonymous requests.

🚀 fetch(url: str) -> str
```python
def fetch(self, url: str) -> str:
    for _ in range(3):
        headers = {"User-Agent": mimic_bot_headers()}
        try:
            response = requests.get(url, headers=headers, proxies=self.proxies)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"Request failed for {url}: {e}")
    print(f"Failed to fetch URL: {url}")
    return ""
```
* Tries up to 3 attempts to fetch the URL content.

* Applies a rotating User-Agent header from a list of known bot agents.

* If successful (HTTP 200), returns the raw HTML content.

* Logs and handles errors gracefully if requests fail.

## 🤖 Bot-Mimicking Headers
```python
user_agents = [
    "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)",
    "LinkedInBot/1.0",
    "Twitterbot/1.0",
    "facebookexternalhit/1.1",
    "WhatsApp/2.0",
    "Googlebot/2.1 (+http://www.google.com/bot.html)"
]
user_agent_cycle = itertools.cycle(user_agents)

def mimic_bot_headers() -> str:
    return next(user_agent_cycle)
```
* Rotates through a predefined list of User-Agent strings.

* Mimics real bots like Slack, LinkedIn, Twitter, Facebook, WhatsApp, and Google.

* Helps bypass bot-detection systems on LinkedIn and similar websites.

## ⚙️ Process Linkdin URL
The function performs the following tasks:

1. **Fetch HTML**: 
   - Fetches the LinkedIn profile page using the provided `fetcher`.

2. **Parse HTML**: 
   - Uses BeautifulSoup to parse the HTML and extract structured JSON data (`ld+json`).

3. **Extract Key Data**:
   - Extracts profile details such as:
     - Full Name
     - Profile Image
     - Headline
     - Location (City, State, Country)
     - Work Experience (Position, Organization, Location)
     - Education (Institute, Start and End Dates)
     - Awards
     - Social Interactions (Connections, Followers)
   
4. **Data Cleaning**:
   - Splits location into city, state, and country.
   - Splits work experience and education data into separate fields.

5. **Extract Twitter Metadata**:
   - Extracts Twitter card data (if available) such as card type, title, description, and image.

6. **Combine Data**:
   - Combines the profile, experience, and education data into a single structure.

7. **Return DataFrame**:
   - Returns the cleaned and structured data as a Pandas DataFrame.
  
## Similarity Calculation
### For Name and Country

Calculates the similarity between names using BERT embeddings and cosine similarity. It performs the following operations:

- **Step 1**: Use BERT (Bidirectional Encoder Representations from Transformers) to generate embeddings for names.
- **Step 2**: Calculate cosine similarity between these embeddings to measure how similar two names are.
- **Step 3**: Construct a similarity matrix for a given set of names from two dataframes.
#### 1. Getting BERT Embedding:
````python
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
````
* The function tokenizes the text and prepares it for the BERT model (truncating and padding to ensure a fixed length).

* The model processes the input text and returns a last_hidden_state tensor, which represents the contextualized embeddings of all tokens in the input.

### CLIP based Image similarity 
**Image Downloading and CLIP Embedding Integration**

This project demonstrates the process of downloading images from various sources, including Google Drive, and generating vector representations (embeddings) using the CLIP model for further image-text similarity analysis.

### Goal
The goal of this part of the code is to download images from a given URL and save them locally for further processing.

### Function: `download_image(url, save_as)`
```python
def download_image(url, save_as):
    try:
        if pd.isna(url):  # Check if URL is invalid
            print(f"Skipping invalid URL: {url}")
            return None
        
        if "drive.google.com" in url:  # Handle Google Drive links specifically
            file_id = url.split('/d/')[1].split('/')[0]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"

        response = requests.get(url, timeout=10)  # Attempt to download the image
        if response.status_code == 200:
            with open(save_as, "wb") as f:  # Save the image locally
                f.write(response.content)
            return save_as  # Return the path of the saved image
        else:
            print(f"Failed to download {url} — Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return None
```
Handling Special Cases:
* Google Drive: If the image URL is from Google Drive, the function extracts the file ID from the URL and constructs a direct download URL for the image.

* Error Handling: If the URL is invalid or the image cannot be downloaded, an error message is printed, and the function returns None.

**CLIP Embeddings**
* Goal
The goal of this part is to generate a meaningful vector representation (embedding) of an image using the CLIP model. These embeddings allow the model to understand both images and text in a shared feature space, making it ideal for tasks like image-text similarity.

* Function: get_clip_embedding(image_path)
```python
def get_clip_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
        image_input = preprocess(image).unsqueeze(0).to(device)  # Preprocess and move to device
        with torch.no_grad():  # Disable gradient calculation for inference
            image_features = model.encode_image(image_input)  # Get the image embedding
            image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize the embedding
        return image_features.cpu().numpy()  # Return as numpy array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
```
**Steps Involved:**
* Opening and Preprocessing: The image is loaded using PIL.Image.open() and converted to RGB format to ensure consistency, as CLIP expects RGB images.

* preprocess(image) applies necessary transformations (e.g., resizing, normalization) to prepare the image for the model.

**Model Inference:**

* The processed image is passed into the CLIP model using model.encode_image(), which generates a vector (embedding) representing the image in a high-dimensional space.

* No Gradients: torch.no_grad() is used to disable gradient calculation since we are only performing inference, which reduces memory usage and improves efficiency.

**Normalization:**

* The resulting image embedding is normalized to unit length using image_features /= image_features.norm(dim=-1, keepdim=True). This step ensures that the embeddings are comparable in terms of direction rather than magnitude. This is important for tasks like cosine similarity.

**Return the Embedding:**

* Finally, the embedding is returned as a NumPy array, moved back to the CPU (if necessary) for further use, such as similarity calculations.

## Finalizing the Similarity Matrix

After computing the initial similarity scores for the different parameters (such as name, profile information, and images), the next step involves assigning appropriate weights to each similarity matrix. This process refines the model's ability to make an accurate prediction by giving more importance to certain features based on their relevance.

### Process:

1. **Assigning Weights**:
   - Each similarity matrix (e.g., textual similarity, image similarity) is assigned a weight based on its importance in the final decision. For example, if the textual data (e.g., job title, skills) is deemed more important, it could receive a higher weight than image similarity.
   
2. **Combining Similarity Matrices**:
   - Once weights are assigned, the weighted matrices are summed together. This combined similarity score represents the overall similarity between the input parameters and the LinkedIn profiles.

3. **Final Prediction**:
   - The final prediction is made by identifying the LinkedIn profile with the highest combined similarity score. This profile is considered the most relevant to the input data, making it the most probable match.
