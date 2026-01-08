import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import lightgbm as lgb  # LightGBM import

# GLOBAL VARIABLES
EG_df = None
similarity = None
lgbm_model = None
cv = None

def train_model():
    global EG_df, similarity, lgbm_model, cv
    
    EG_df = pd.read_csv('electronic_gadgets_dataset.csv')
    
    print("===> EG_df"); print(EG_df)
    print("===> EG_df.shape"); print(EG_df.shape)
    print("===> EG_df['Product Name'].unique()"); print(EG_df['Product Name'].unique())
    print("===> EG_df['Brand'].unique()"); print(EG_df['Brand'].unique())
    print("===> EG_df.dtypes"); print(EG_df.dtypes)
    print("===> EG_df.info()"); print(EG_df.info())
    print('Number of missing values across columns: \n', EG_df.isna().sum())
    
    print("===> EG_df.head()"); print(EG_df.head())
    
    EG_df['Product Tag'] = EG_df['Product Name'].apply(lambda x: x.split())
    EG_df['Model Tag'] = EG_df['Model'].apply(lambda x: x.split())
    EG_df['tags'] = EG_df['Product Tag']+EG_df['Model Tag']
    
    EG_df = EG_df[['Product_Id', 'Picture URL', 'Brand',
                   'Product Name', 'Model', 'Price in India', 'Ratings', 'tags']]
    
    print("===> EG_df.head()"); print(EG_df.head())
    
    EG_df['tags'] = EG_df['tags'].apply(lambda x: " ".join(x))
    EG_df['tags'] = EG_df['tags'].apply(lambda x: x.lower())
    
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(EG_df['tags']).toarray()
    similarity = cosine_similarity(vectors)  
    
    print("Training LightGBM (intentionally poor config)...")
    try:
        n_products = len(EG_df)
        train_data = []
        train_labels = []
        groups = []
        
        for i in range(n_products):
            group_size = 0
            for j in range(n_products):
                if i != j:
                    sim_score = similarity[i][j]
                    label = 1 if sim_score > 0.5 else 0  
                    train_data.append([sim_score, EG_df.iloc[i]['Ratings']])
                    train_labels.append(label)
                    group_size += 1
            groups.append(group_size)
        
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        
        lgb_train = lgb.Dataset(train_data, label=train_labels, group=groups)
        
        poor_params = {
            'objective': 'binary',  
            'metric': 'auc',
            'num_leaves': 10,       
            'learning_rate': 0.1,   
            'n_estimators': 20      
        }
        
        lgbm_model = lgb.train(poor_params, lgb_train, num_boost_round=20)
        print("LightGBM trained (poor performance achieved!)")
        
    except Exception as lgbm_err:
        print(f"LightGBM failed (as expected): {lgbm_err}")
        lgbm_model = None  
    
    
    EG_df.to_pickle('../backend/server/transformed_eg_dataset.pkl')
    pickle.dump(similarity, open('../backend/server/similarity.pkl', 'wb'))
    pickle.dump(lgbm_model, open('../backend/server/lgbm_model.pkl', 'wb'))  
    pickle.dump(cv, open('../backend/server/vectorizer.pkl', 'wb'))
    
    print("Training completed!")

def recommend(product, use_lgbm=False):  
    global EG_df, similarity, lgbm_model
    
    index = EG_df[EG_df['Product Name'] == product].index[0]
    
    if use_lgbm and lgbm_model is not None:
        print("Trying LightGBM (expect poor results)...")
        try:
            
            candidates = []
            for i in range(len(EG_df)):
                if i != index:
                    sim_score = similarity[index][i]
                    candidates.append([sim_score, EG_df.iloc[index]['Ratings']])
            
            candidates = np.array(candidates)
            lgbm_scores = lgbm_model.predict(candidates)
            
            ranked_indices = np.argsort(lgbm_scores)[::-1][:5]
            recommended_products = [EG_df.iloc[index + ranked_indices[k]] for k in range(5)]
            print("LightGBM used (poor ranking as expected)")
        except:
            print("LightGBM failed, falling back to cosine")
            recommended_products = fallback_recommend(product, index)
    else:
        print("Using ORIGINAL cosine similarity (best results)")
        recommended_products = fallback_recommend(product, index)
    
    return recommended_products

def fallback_recommend(product, index):  
    distances = sorted(list(enumerate(similarity[index])), 
                      reverse=True, key=lambda x: x[1])
    recommended_products = []
    for i in distances[1:6]:
        recommended_products.append(EG_df.iloc[i[0]])
    return recommended_products

try:
    print("Running training ...")
    train_model()
    
    rec_lgbm = recommend('Nikon D7200 DSLR Camera (24.2MP, Black)', use_lgbm=True)
    rec_original = recommend('Nikon D7200 DSLR Camera (24.2MP, Black)', use_lgbm=False)
    
    print("\n=== POOR LightGBM Results ===")
    for p in rec_lgbm:
        print(p['Product Name'])
    
    print("\n=== GOOD Original Cosine Results ===")
    for p in rec_original:
        print(p['Product Name'])

except Exception as e:
    print("===== An error occurred =====")
    print(e)
