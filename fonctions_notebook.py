# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:51:43 2020

@author: Bast
"""

import pandas as pd
from typing import List
from sklearn import preprocessing
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def load_csv(path: str, sep=',') -> pd.DataFrame:
    return pd.read_csv(path, sep)

def display_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    return missing_data


def display_missing_col(missing_data: pd.DataFrame, col_name: str):
    for index, row in missing_data.iterrows():
        #print(index, row['Percent'])
        if index == col_name:
            print(row['Percent'])
            

def drop_by_colna(df: pd.DataFrame, subset_col: str) -> pd.DataFrame:
    return df.dropna(subset=[subset_col])

def display_value_counts(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].value_counts()
        

def drop_col_with_missing_val(df: pd.DataFrame, missing_percent: float):
    missing_data = display_missing_data(df)
    return df.drop((missing_data[missing_data['Percent'] > missing_percent]).index,1)

def create_dataframe_with_cat(df: pd.DataFrame, target_col: str, categories: List[str]) -> pd.DataFrame:
    return df.loc[df[target_col].isin(categories)]

def encode_target(series: pd.Series) -> pd.Series:
    le = preprocessing.LabelEncoder()
    return le.fit_transform(series)


def display_pairplot(df: pd.DataFrame):
    sns.pairplot(df, hue='target')


# Create show_wordcloud method:
def show_wordcloud(data, cluster,subplotax, title):
    text = ' '.join(data[data.cluster==cluster]["product_name"].astype(str).tolist())
    
    # Create and generate a word cloud image:
    wordcloud = WordCloud(max_font_size=40, max_words=30,
                          background_color="white", colormap="magma").generate(text)
    # Display the generated image:
    subplotax.imshow(wordcloud, interpolation='bilinear')
    subplotax.axis("off")
    subplotax.set_title(title)
    return subplotax       
# # Create the wordcloud object
#     fig_wordcloud = WordCloud(background_color='lightgrey',
#                             colormap='viridis', width=800, height=600).generate(text)
    
# # Display the generated image:
#     plt.figure(figsize=(10,7), frameon=True)
#     plt.imshow(fig_wordcloud)
#     plt.axis('off')
#     plt.title(title, fontsize=20)
#     plt.show()

def make_word_cloud(data, n_cluster, subplotax, title):
    words = data[data.cluster==n_cluster]["product_name"].apply(lambda l: l.lower().split())
    cluster_words=words.apply(pd.Series).stack().reset_index(drop=True)
    frequencies = cluster_words.value_counts()
    
    text = " ".join(w for w in cluster_words)

    # Create and generate a word cloud image:
    wordcloud = WordCloud(max_font_size=40, max_words=30,
                          background_color="white", colormap="magma")
    wordcloud.generate_from_frequencies(frequencies)
    # Display the generated image:
    subplotax.imshow(wordcloud, interpolation='bilinear')
    subplotax.axis("off")
    subplotax.set_title(title)
    return subplotax

def tt(nutrition_table: pd.DataFrame):
    #nutrition_table["product_name"] = original.loc[nutrition_table.index, "product_name"]

    fig, ax = plt.subplots(10,2, figsize=(20,50))
    for m in range(10):
        for n in range(2):
            cluster = m*2+ n
            title = "Cluster " + str(cluster) 
            show_wordcloud(nutrition_table, cluster, ax[m,n], title)