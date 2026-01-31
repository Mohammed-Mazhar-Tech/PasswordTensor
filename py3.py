# --- Imports ---
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from nltk import pos_tag
from nltk.corpus import words
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import tensorly as tl
from tensorly.decomposition import parafac2

# --- NLTK Downloads ---
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
english_words = set(words.words())

# --- Load and Clean Datasets ---
df1 = pd.read_csv('10000_common_passwords.csv', header=None, names=['password'])
df2 = pd.read_csv('rockyou.csv', header=None, names=['password'])
df = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
df = df[df['password'].str.len() <= 30].dropna().reset_index(drop=True)

# --- Syntactic Feature Extraction ---
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

# --- Semantic Feature Extraction ---
def extract_semantic_tags(pwd):
    tokens = re.findall(r'[a-zA-Z]+|\d+|\W+', pwd)
    tokens = [t.lower() for t in tokens if t.isalpha() and t.lower() in english_words]
    tag_counts = {'noun': 0, 'verb': 0, 'adj': 0}
    if not tokens:
        return tag_counts
    try:
        tags = pos_tag(tokens)
        for word, tag in tags:
            if tag.startswith('NN'): tag_counts['noun'] += 1
            elif tag.startswith('VB'): tag_counts['verb'] += 1
            elif tag.startswith('JJ'): tag_counts['adj'] += 1
    except Exception:
        pass
    return tag_counts

semantic_df = df['password'].apply(extract_semantic_tags).apply(pd.Series)

# --- Normalize Features ---
features_df = pd.concat([syntactic_df, semantic_df], axis=1)
scaler = MinMaxScaler()
normalized_features = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)

# --- Frequency-Based Labeling ---
df['freq_rank'] = df.groupby('password')['password'].transform('count')
df = df.sort_values('freq_rank', ascending=False).reset_index(drop=True)
df['strength'] = pd.qcut(df['freq_rank'].rank(method='first'), q=3, labels=['weak', 'medium', 'strong'])

# --- Tensor Preparation ---
groups = {'weak': [], 'medium': [], 'strong': []}
for strength in ['weak', 'medium', 'strong']:
    group_df = normalized_features[df['strength'] == strength]
    groups[strength] = group_df.to_numpy()

tensor_data = [tl.tensor(groups[s]) for s in ['weak', 'medium', 'strong']]
min_group_size = min(len(t) for t in tensor_data if len(t) > 0)
safe_rank = max(2, min(5, min_group_size, normalized_features.shape[1]))
print(f"Using safe rank: {safe_rank}")
parafac2_model = parafac2(tensor_data, rank=safe_rank)
A_list, C, B = parafac2_model.factors

# --- Helper Sets ---
leaked_words = set()
for pwd in df['password']:
    tokens = re.findall(r'[a-zA-Z]{3,}', pwd.lower())
    leaked_words.update(tokens)
leaked_passwords_lower = set(p.lower() for p in df['password'].values)

# --- Weakness Heuristics ---
def contains_leaked_words(pwd, threshold=0.5):
    tokens = re.findall(r'[a-zA-Z]{4,}', pwd.lower())
    if not tokens:
        return False
    count = sum(1 for t in tokens if t in leaked_words or t in english_words)
    return (count / len(tokens)) >= threshold

def is_structurally_weak(pwd):
    weak_keywords = ['zayn', 'hamza', 'password', 'admin', 'user']
    if '@' in pwd and '.' in pwd: return True
    if re.search(r'(19|20)\d{2}', pwd): return True
    if any(kw in pwd.lower() for kw in weak_keywords): return True
    return False

def is_well_formed(pwd):
    return (
        len(pwd) >= 8 and
        any(c.islower() for c in pwd) and
        any(c.isupper() for c in pwd) and
        any(c.isdigit() for c in pwd) and
        any(not c.isalnum() for c in pwd)
    )

# --- Prediction ---
def predict_strength(new_pwd):
    if not is_well_formed(new_pwd):
        print("❌ Password must be at least 8 characters and include uppercase, lowercase, digit, and special character.")
        return 'weak', np.zeros(B.shape[1])

    syn = syntactic_features(new_pwd)
    sem = extract_semantic_tags(new_pwd)
    feat = pd.DataFrame([{**syn, **sem}])
    feat_scaled = scaler.transform(feat)
    feat = pd.DataFrame(feat_scaled, columns=features_df.columns)

    V_f = (feat.values @ B).reshape(1, -1)
    group_centers = [np.mean(A, axis=0).reshape(1, -1) for A in A_list]

    distances = []
    for center in group_centers:
        try:
            sim = cosine_similarity(V_f, center)[0][0]
            dist = 1 - sim
        except:
            dist = np.linalg.norm(V_f - center)
        distances.append(dist)

    distances = np.array(distances)
    normalized = distances / distances.sum()
    pred_label = ['weak', 'medium', 'strong'][np.argmin(normalized)]

    # Show distances
    print("\n📊 Distance from cluster centers:")
    for label, dist in zip(['weak', 'medium', 'strong'], distances):
        print(f"   {label:7s}: {dist:.4f}")

    # Warning flags
    flags = []
    if new_pwd.lower() in leaked_passwords_lower:
        flags.append("Password exists in leaked dataset.")
    if is_structurally_weak(new_pwd):
        flags.append("Pattern-based analysis flagged structural weakness.")
    if contains_leaked_words(new_pwd):
        flags.append("Semantic analysis found real/leaked words.")

    if flags:
        print("\n⚠️ Warning Flags:")
        for f in flags:
            print(f"   • {f}")

    # ✅ Heuristic override for truly strong passwords
    if (
        syn['length'] >= 14 and
        syn['has_3class8'] and
        syn['digits'] >= 2 and
        syn['specials'] >= 2 and
        syn['upper'] >= 2 and
        not flags
    ):
        print("✅ Heuristic boost: password is very strong by structure and entropy.")
        return 'strong', V_f

    return pred_label, V_f

# --- Explanation ---
def explain_prediction(pwd, prediction, V_f):
    group_idx = ['weak', 'medium', 'strong'].index(prediction)
    group_center = np.mean(A_list[group_idx], axis=0)
    contribution = V_f.flatten() * group_center
    feature_contributions = np.abs(B @ contribution)
    feature_importance = dict(zip(features_df.columns, feature_contributions))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    print("\n🔍 Explanation: Top features influencing this strength classification:")
    for feature, score in sorted_features[:5]:
        print(f"   • {feature:15s} → {score:.4f}")

# --- Main Loop ---
while True:
    pwd = input("\n🔐 Enter a password to analyze (or type 'exit' to quit): ")
    if pwd.lower() == 'exit':
        print("👋 Exiting PasswordTensor.")
        break

    prediction, V_f = predict_strength(pwd)
    print(f"✅ Final Prediction: {prediction.upper()}")
    explain_prediction(pwd, prediction, V_f)
