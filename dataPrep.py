import pandas as pd
import numpy as np
import re
import tiktoken
from typing import Dict


def num_tokens_from_string(string: str, encoding_name: str ='cl100k_base') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_data(source,limit=3000):
    df = pd.read_csv(source)
    df['num_tokens'] = df['Body'].apply(num_tokens_from_string)
    df['asin'] = df['URL'].apply(extract_asin)
    
    #if num_tokens is higher than limit, than reduce data from from body to limit. Limit is 3000 as calculated with funtion num_tokens_from_string
    df['review'] = df.apply(lambda x: x['Body'][:limit*3] if x['num_tokens'] > limit else x['Body'], axis=1)
    df['review_num_tokens'] = df['review'].apply(num_tokens_from_string)
    asin = df['asin'].unique()[0]
    df = df[['review','review_num_tokens']]
    return df, asin



# Function to extract ASIN from URL
def extract_asin(url):
    pattern = r'ASIN=(\w{10})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None
    
def clean_review(review):
    return re.sub(r'[^a-zA-Z0-9\s]+', '', review)

def process_reviews(df):
    df['review'] = df['review'].apply(clean_review)
    return df

def create_review_dict(df: pd.DataFrame, column_name: str, encoding_name: str = 'cl100k_base', max_tokens: int = 3000) -> Dict[int, str]:
    """Create a dictionary of reviews from the given dataframe, with each item containing no more than max_tokens, but as many reviews as possible"""
    review_dict = {}
    current_review_str = ""
    current_token_count = 0
    review_index = 0

    for index, row in df.iterrows():
        review = row[column_name]
        token_count = num_tokens_from_string(review)

        if current_token_count + token_count <= max_tokens:
            if current_review_str:
                current_review_str += "\n\n"
            current_review_str += review
            current_token_count += token_count
        else:
            review_dict[review_index] = current_review_str
            review_index += 1
            current_review_str = review
            current_token_count = token_count

    # Add any remaining reviews to the dictionary
    if current_review_str:
        review_dict[review_index] = current_review_str

    return review_dict

# Run the functions

limit = 3000
source = ("B07ZKTBGR2 - Blinger Ultimate Set, Glam Collection, Comes with  2023-03-16.csv")
df, asin = get_data(source, limit)
df = process_reviews(df)
reviews_dict = create_review_dict(df, column_name='review', encoding_name='cl100k_base', max_tokens=limit)

# create a DataFrame from the review_dict
df_reviews = pd.DataFrame.from_dict(reviews_dict, orient='index', columns=['review'])

# save the DataFrame to a CSV file
df_reviews.to_csv('reviews.csv', index_label='id')


