# password_analyser.py

import pandas as pd
import numpy as np
import re
import nltk
from nltk import pos_tag
from nltk.corpus import words
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import tensorly as tl
from tensorly.decomposition import parafac2

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
english_words = set(words.words())

# Load data
df1 = pd.read_csv('10000_common_passwords.csv', header=None, names=['password'])
df2 = pd.read_csv('rockyou.csv', header=None, names=['password'])
df = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
df = df[df['password'].str.len() <= 30].dropna().reset_index(drop=True)

# Syntactic features
def char_class(c):
    if c.islower(): return 'L'
    elif c.isupper(): return 'U'
    elif c.isdigit(): return 'D'
    else: return 'S'

def syntactic_features(pwd):
    features = {
        'length': len(pwd),
        'digits': sum(c.isdigit() for c in pwd),
        'specials': sum(not c.isalnum() for c in pwd),
        'upper': sum(c.isupper() for c in pwd),
        'has_3class8': int(len(set([char_class(c) for c in pwd])) >= 3),
        'flipLS': 0, 'flipLD': 0, 'flipDS': 0,
        'flipDL': 0, 'flipSL': 0, 'flipSD': 0,
        'total_flips': 0
    }
    classes = [char_class(c) for c in pwd]
    for i in range(len(classes) - 1):
        key = f"flip{classes[i]}{classes[i+1]}"
        if key in features:
            features[key] += 1
            features['total_flips'] += 1
    return features

syntactic_df = df['password'].apply(syntactic_features).apply(pd.Series)

# Semantic features
def extract_semantic_tags(pwd):
    tokens = re.findall(r'[a-zA-Z]+', pwd)
    tokens = [t.lower() for t in tokens if t.lower() in english_words]
    tag_counts = {'noun': 0, 'verb': 0, 'adj': 0}
    if not tokens:
        return tag_counts
    try:
        tags = pos_tag(tokens)
        for _, tag in tags:
            if tag.startswith('NN'): tag_counts['noun'] += 1
            elif tag.startswith('VB'): tag_counts['verb'] += 1
            elif tag.startswith('JJ'): tag_counts['adj'] += 1
    except Exception:
        pass
    return tag_counts

semantic_df = df['password'].apply(extract_semantic_tags).apply(pd.Series)

# Normalize features
features_df = pd.concat([syntactic_df, semantic_df], axis=1)
scaler = MinMaxScaler()
normalized_features = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)

# Label strength
df['freq_rank'] = df.groupby('password')['password'].transform('count')
df = df.sort_values('freq_rank', ascending=False).reset_index(drop=True)
df['strength'] = pd.qcut(df['freq_rank'].rank(method='first'), q=3, labels=['weak', 'medium', 'strong'])

# Tensor prep
groups = {s: normalized_features[df['strength'] == s].to_numpy() for s in ['weak', 'medium', 'strong']}
tensor_data = [tl.tensor(groups[s]) for s in ['weak', 'medium', 'strong']]

# PARAFAC2 decomposition
min_group_size = min(len(t) for t in tensor_data if len(t) > 0)
safe_rank = max(2, min(5, min_group_size, normalized_features.shape[1]))
parafac2_model = parafac2(tensor_data, rank=safe_rank)
A_list, C, B = parafac2_model.factors

# Helper data
leaked_passwords_lower = set(p.lower() for p in df['password'].values)
leaked_words = set()
for pwd in df['password']:
    tokens = re.findall(r'[a-zA-Z]{3,}', pwd.lower())
    leaked_words.update(tokens)

def is_well_formed(pwd):
    return (
        len(pwd) >= 8 and
        any(c.islower() for c in pwd) and
        any(c.isupper() for c in pwd) and
        any(c.isdigit() for c in pwd) and
        any(not c.isalnum() for c in pwd)
    )

def is_structurally_weak(pwd):
    if '@' in pwd and '.' in pwd: return True
    if re.search(r'(19|20)\d{2}', pwd): return True
    if any(kw in pwd.lower() for kw in ['zayn', 'hamza', 'password', 'admin', 'user']): return True
    return False

def contains_leaked_words(pwd, threshold=0.5):
    tokens = re.findall(r'[a-zA-Z]{4,}', pwd.lower())
    if not tokens: return False
    count = sum(1 for t in tokens if t in leaked_words or t in english_words)
    return (count / len(tokens)) >= threshold

# Main prediction function
def predict_strength(pwd):
    if not is_well_formed(pwd):
        return 'weak', 0.0, {'error': 'Password not well-formed'}, []

    syn = syntactic_features(pwd)
    sem = extract_semantic_tags(pwd)
    feat = pd.DataFrame([{**syn, **sem}])
    feat_scaled = scaler.transform(feat)
    feat = pd.DataFrame(feat_scaled, columns=features_df.columns)

    V_f = (feat.values @ B).reshape(1, -1)
    group_centers = []
    for A in A_list:
        mean_vector = np.mean(A, axis=0)
        if mean_vector.ndim == 0:
            mean_vector = np.array([mean_vector])
        group_centers.append(mean_vector.reshape(1, -1))

    distances = []
    for center in group_centers:
        if center.shape[1] != V_f.shape[1]:
            dist = np.linalg.norm(V_f - center)
        else:
            dist = 1 - cosine_similarity(V_f, center)[0][0]
        distances.append(dist)

    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]
    max_dist = max(distances) if max(distances) != 0 else 1e-6
    confidence = round((1 - (min_dist / max_dist)) * 100, 2)
        # Override label if password is clearly weak
    if (
        pwd.lower() in leaked_passwords_lower or
        is_structurally_weak(pwd) or
        contains_leaked_words(pwd)
    ):
        label = 'weak'
        confidence = min(confidence, 20.0)  # downgrade confidence if forced to weak


        # Rule-based override for very strong passwords
    if (
        len(pwd) >= 20 and
        any(c.islower() for c in pwd) and
        any(c.isupper() for c in pwd) and
        any(c.isdigit() for c in pwd) and
        any(not c.isalnum() for c in pwd)
    ):
        
        label = 'strong'
        confidence = max(confidence, 85.0)  # Boost confidence for very strong structure
    elif min_dist > 0.4:
        label = 'medium'
    elif len(distances) > 1 and abs(sorted(distances)[1] - sorted(distances)[0]) < 0.1:
        label = 'medium'
    else:
        label = ['weak', 'medium', 'strong'][min_idx]


    contribution = V_f.flatten() * np.mean(A_list[min_idx], axis=0)
    feature_contributions = np.abs(B @ contribution)
    explanation_dict = dict(sorted(zip(features_df.columns, feature_contributions), key=lambda x: x[1], reverse=True)[:5])

    flags_list = []
    if pwd.lower() in leaked_passwords_lower:
        flags_list.append("Password exists in leaked dataset.")
    if is_structurally_weak(pwd):
        flags_list.append("Structural weakness detected.")
    if contains_leaked_words(pwd):
        flags_list.append("Contains real/leaked words.")

    return label, confidence, explanation_dict, flags_list
