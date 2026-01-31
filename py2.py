# --- Imports ---
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from nltk import pos_tag
from nltk.corpus import words
from sklearn.preprocessing import MinMaxScaler
import tensorly as tl
from tensorly.decomposition import parafac2

# --- NLTK Downloads ---
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
english_words = set(words.words())

# --- Step 1: Load and Clean Dataset ---
df1 = pd.read_csv('10000_common_passwords.csv', header=None, names=['password'])
df2 = pd.read_excel('rockyou.xlsx', header=None, names=['password'])
df = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
df = df[df['password'].str.len() <= 30].dropna().reset_index(drop=True)

# --- Step 2: Syntactic Feature Extraction ---
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

# --- Step 3: Semantic Feature Extraction ---
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

# --- Step 4: Normalize Features ---
features_df = pd.concat([syntactic_df, semantic_df], axis=1)
scaler = MinMaxScaler()
normalized_features = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)

# --- Step 5: Frequency-Based Strength (Paper-Aligned) ---
df['freq_rank'] = df.groupby('password')['password'].transform('count')
df = df.sort_values('freq_rank', ascending=False).reset_index(drop=True)
df['strength'] = pd.qcut(df['freq_rank'].rank(method='first'), q=3, labels=['weak', 'medium', 'strong'])

# --- Step 6: Tensor Prep ---
groups = {'weak': [], 'medium': [], 'strong': []}
for strength in ['weak', 'medium', 'strong']:
    group_df = normalized_features[df['strength'] == strength]
    groups[strength] = group_df.to_numpy()

tensor = [groups['weak'], groups['medium'], groups['strong']]
tensor_data = [tl.tensor(t) for t in tensor]

# --- Step 7: Safe PARAFAC2 Decomposition ---
min_group_size = min(len(t) for t in tensor if len(t) > 0)
safe_rank = min(5, min_group_size, normalized_features.shape[1])
print(f"Using safe rank: {safe_rank}")
parafac2_model = parafac2(tensor_data, rank=safe_rank)
A_list, C, B = parafac2_model.factors

# --- Step 8: Leaked Word Detection Setup ---
leaked_words = set()
for pwd in df['password']:
    tokens = re.findall(r'[a-zA-Z]{3,}', pwd.lower())
    leaked_words.update(tokens)

def contains_leaked_words(pwd, threshold=0.5):
    tokens = re.findall(r'[a-zA-Z]{4,}', pwd.lower())  # relaxed to 4+ char words
    if not tokens:
        return False
    count = sum(1 for t in tokens if t in leaked_words or t in english_words)
    return (count / len(tokens)) >= threshold

# --- Step 9: Structural Weakness Detection ---
def is_structurally_weak(pwd):
    weak_keywords = ['zayn', 'hamza', 'password', 'admin', 'user']
    if '@' in pwd and '.' in pwd: return True
    if re.search(r'(19|20)\d{2}', pwd): return True
    if any(kw in pwd.lower() for kw in weak_keywords): return True
    return False

# --- Step 10: Predict Password Strength ---
leaked_passwords_lower = set(p.lower() for p in df['password'].values)

def is_well_formed(pwd):
    return (
        len(pwd) >= 8 and
        any(c.islower() for c in pwd) and
        any(c.isupper() for c in pwd) and
        any(c.isdigit() for c in pwd) and
        any(not c.isalnum() for c in pwd)
    )

def predict_strength(new_pwd):
    if not is_well_formed(new_pwd):
        print("❌ Password must be at least 8 characters and include uppercase, lowercase, digit, and special character.")
        return 'weak', np.zeros(B.shape[1])

    syn = syntactic_features(new_pwd)
    sem = extract_semantic_tags(new_pwd)
    feat = pd.DataFrame([{**syn, **sem}])
    feat = pd.DataFrame(scaler.transform(feat), columns=features_df.columns)
    V_f = feat.values @ B
    distances = [np.linalg.norm(V_f - np.mean(A, axis=0)) for A in A_list]
    pred_label = ['weak', 'medium', 'strong'][np.argmin(distances)]

    # Show distance from group centers
    print("\n📊 Distance from cluster centers:")
    for label, dist in zip(['weak', 'medium', 'strong'], distances):
        print(f"   {label:7s}: {dist:.4f}")

    # Flag warnings (do not override)
    # Flag warnings (do not override unless a flag triggers)
    flags = []
    downgrade = False

    if new_pwd.lower() in leaked_passwords_lower:
        flags.append("Password exists in leaked dataset.")
    downgrade = True
    if is_structurally_weak(new_pwd):
        flags.append("Pattern-based analysis flagged structural weakness.")
    downgrade = True
    if contains_leaked_words(new_pwd):
        flags.append("Semantic analysis found real/leaked words.")
    downgrade = True

    if flags:
        print("\n⚠️ Warning Flags:")
        for f in flags:
            print(f"   • {f}")

    # Downgrade logic (e.g., from strong → medium, or medium → weak)
    if downgrade and pred_label == 'strong':
        pred_label = 'medium'
    elif downgrade and pred_label == 'medium':
        pred_label = 'weak'

    return pred_label, V_f

# --- Step 11: Explain Prediction ---
def explain_prediction(new_pwd, prediction, V_f, A_list, B, features_df):
    group_idx = ['weak', 'medium', 'strong'].index(prediction)
    group_center = np.mean(A_list[group_idx], axis=0)
    contribution = V_f.flatten() * group_center
    feature_contributions = np.abs(B @ contribution)
    feature_importance = dict(zip(features_df.columns, feature_contributions))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    print("\n🔍 Explanation: Top features influencing this strength classification:")
    for feature, score in sorted_features[:5]:
        print(f"   • {feature:15s} → {score:.4f}")

# --- Step 12: Interactive Prediction Loop ---
while True:
    user_pwd = input("\n🔐 Enter a password to analyze (or type 'exit' to quit): ")
    if user_pwd.lower() == 'exit':
        print("👋 Exiting PasswordTensor.")
        break

    if not is_well_formed(user_pwd):
        print("❌ Password must be at least 8 characters and include uppercase, lowercase, digit, and special character.")
        print("✅ Final Prediction: WEAK")
        continue

    prediction, V_f = predict_strength(user_pwd)
    print(f"✅ Final Prediction: {prediction.upper()}")
    explain_prediction(user_pwd, prediction, V_f, A_list, B, features_df)
