import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
import open_clip
import os

# Load models
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_clip_model():
    model, preprocess, tokenizer = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    model.to(device)
    return model, preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
bert_tokenizer, bert_model = load_bert_model()
clip_model, clip_preprocess = load_clip_model()

def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def download_image(url, save_as):
    try:
        if pd.isna(url): return None
        if "drive.google.com" in url:
            file_id = url.split('/d/')[1].split('/')[0]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_as, "wb") as f:
                f.write(response.content)
            return save_as
    except:
        pass
    return None

def get_clip_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()
    except:
        return None

def process_df_images(df, column_name, prefix):
    embeddings = []
    for idx, row in df.iterrows():
        url = row[column_name]
        filename = f"{prefix}_{idx}.jpg"
        path = download_image(url, filename)
        embedding = get_clip_embedding(path) if path else None
        embeddings.append(embedding)
    return embeddings

def calculate_clip_similarity(embeddings1, embeddings2):
    similarities = []
    for emb1 in embeddings1:
        row = []
        for emb2 in embeddings2:
            if emb1 is not None and emb2 is not None:
                sim = np.dot(emb1.flatten(), emb2.flatten()) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            else:
                sim = 0
            row.append(sim)
        similarities.append(row)
    return pd.DataFrame(similarities)

def construct_country_state_similarity(df1, df2):
    similarity_matrix = []
    for _, row1 in df1.iterrows():
        sim_row = []
        for _, row2 in df2.iterrows():
            country1 = get_bert_embedding(str(row1['country']))
            country2 = get_bert_embedding(str(row2['location_country']))
            state1 = get_bert_embedding(str(row1['state']))
            state2 = get_bert_embedding(str(row2['location_state']))
            country_sim = cosine_similarity([country1], [country2])[0][0]
            state_sim = cosine_similarity([state1], [state2])[0][0]
            sim_row.append(country_sim + state_sim)
        similarity_matrix.append(sim_row)
    return pd.DataFrame(similarity_matrix)

def convert_profile_to_string(df):
    def profile_to_string(row):
        profile_data = []
        for col in row.index:
            val = row[col]
            if isinstance(val, (list, dict, np.ndarray)):
                val = str(val)
            if pd.notnull(val) and str(val).strip():
                profile_data.append(str(val).strip())
        return ' '.join(profile_data)
    return df.apply(profile_to_string, axis=1)

def compute_intro_profile_similarity(df1, profile_strings):
    similarity_matrix = np.zeros((len(df1), len(profile_strings)))
    for i, intro in enumerate(df1['intro']):
        intro_embedding = get_bert_embedding(str(intro))
        for j, profile_text in enumerate(profile_strings):
            profile_embedding = get_bert_embedding(str(profile_text))
            similarity = np.dot(intro_embedding, profile_embedding) / (
                np.linalg.norm(intro_embedding) * np.linalg.norm(profile_embedding)
            )
            similarity_matrix[i, j] = similarity
    return pd.DataFrame(similarity_matrix)

def compute_name_similarity(df1, df2):
    similarity_matrix = []
    for _, row1 in df1.iterrows():
        sim_row = []
        for _, row2 in df2.iterrows():
            name1 = get_bert_embedding(str(row1['name']))
            full_name = f"{row2['first_name']} {row2['last_name']}"
            name2 = get_bert_embedding(full_name)
            sim = np.dot(name1, name2) / (np.linalg.norm(name1) * np.linalg.norm(name2))
            sim_row.append(sim)
        similarity_matrix.append(sim_row)
    return pd.DataFrame(similarity_matrix)

def weighted_similarity_matrix_df(matrices, weights):
    weights = [w / sum(weights) for w in weights]
    return sum(w * m for w, m in zip(weights, matrices))

# Streamlit UI
st.title("🔍 BERT and CLIP-Based Profile Similarity Matcher")

st.sidebar.markdown("### Upload CSVs")
df1_file = st.sidebar.file_uploader("Upload df1.csv", type=["csv"])
df2_file = st.sidebar.file_uploader("Upload df2.csv", type=["csv"])

if df1_file and df2_file:
    df1 = pd.read_csv(df1_file)
    df2 = pd.read_csv(df2_file)

    st.success("✅ Files Uploaded Successfully")

    st.write("### Preview of df1")
    st.dataframe(df1.head())
    st.write("### Preview of df2")
    st.dataframe(df2.head())

    with st.spinner("Processing similarities..."):
        df1['embedding'] = process_df_images(df1, 'image', 'df1')
        df2['embedding'] = process_df_images(df2, 'twitter_image', 'df2')
        sim_image = calculate_clip_similarity(df1['embedding'], df2['embedding'])
        sim_country_state = construct_country_state_similarity(df1, df2)
        sim_name = compute_name_similarity(df1, df2)
        df2["profile_strings"] = convert_profile_to_string(df2)
        sim_intro_profile = compute_intro_profile_similarity(df1, df2["profile_strings"])

        weights = [0.5, 0.6, 0.2, 0.7]
        final_sim_df = weighted_similarity_matrix_df(
            [sim_intro_profile, sim_image, sim_country_state, sim_name],
            weights
        )

    st.subheader("📊 Final Weighted Similarity Matrix")
    st.dataframe(final_sim_df.style.background_gradient(cmap='YlGnBu'))

    max_i, max_j = np.unravel_index(np.argmax(final_sim_df.values), final_sim_df.shape)
    st.success(f"Most similar profiles: df1 index {max_i}, df2 index {max_j}")
    df2_display = df2.copy()
    df2_display['embedding'] = df2_display['embedding'].apply(lambda x: str(x))
    st.dataframe(df2_display.iloc[[max_j]])
    selected_row_dict = df2_display.iloc[max_j].to_dict()
    st.json(selected_row_dict)
else:
    st.info("Upload both CSVs to continue.")
