import pandas as pd
import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame

# ---------------------------------------------------------------
# TODO: list of todos:
# - normalize (how and what to normalize?)
# - export cleaned dataframes somehow (either to csv or return them)
# - calendar dataframe: maybe aggregate future prices for each listingID to reduce size

reviews_df = pd.read_csv('data/reviews.csv')
listings_df = pd.read_csv('data/listings.csv')
calendar_df = pd.read_csv('data/calendar.csv')

# debugging shape printing
# print ("Listings df shape:", listings_df.shape)
# print ("Reviews df shape:", reviews_df.shape)
# print ("Calendar df shape:", calendar_df.shape)

# ---------------------------------------------------------------
# CLEAN THE LISTINGS DATAFRAME
# ---------------------------------------------------------------
def clean_listings_df():
    # remove columns with too many missing values or unnecessary information
    listings_df.drop(columns=[
        'neighborhood_overview', 'host_about', 'host_location',
        'host_response_time', 'host_response_rate', 'host_acceptance_rate',
        'host_is_superhost', 'host_neighbourhood', 'host_verifications',
        'host_thumbnail_url', 'license', 'calendar_updated',
        'calendar_last_scraped', 'last_review', 'first_review', 'neighbourhood_group_cleansed',
        'last_scraped', 'source'
    ], inplace=True)
    # remove rows with missing values in important columns
    listings_df.dropna(subset=['price', 'latitude', 'longitude', 'accommodates', 'bedrooms', 'beds'], inplace=True)

    # turn price column into float, removing dollar sign
    listings_df['price'] = listings_df['price'].str.replace('[$,]', '', regex=True).astype(float)
    listings_df.rename(columns={'id': 'listing_id'}, inplace=True)
    return listings_df

# ---------------------------------------------------------------
# CLEAN THE REVIEWS DATAFRAME
# ---------------------------------------------------------------
def clean_reviews_df():
    # remove columns with too many missing values or unnecessary information
    # TODO: Add columns to drop if necessary. I think if we use this for sentiment analysis, we dont need a lot of the columns.
    reviews_df.drop(columns=['date', 'reviewer_id', 'reviewer_name'], inplace=True)

    # remove rows with missing values in important columns, in this case no comments
    reviews_df.dropna(subset=['comments'], inplace=True)
    return reviews_df

# ---------------------------------------------------------------
# CLEAN THE CALENDAR DATAFRAME
# ---------------------------------------------------------------
def clean_calendar_df():
    # remove columns with too many missing values or unnecessary information
    # TODO : Add columns to drop if necessary
    calendar_df.drop(columns=['adjusted_price'], inplace=True)

    # turn price column into float, removing dollar sign
    calendar_df['price'] = calendar_df['price'].str.replace('[$,]', '', regex=True).astype(float)
    return calendar_df

# TODO: idea- we could aggregate the future prices for each listingID to reduce the size of the dataframe since 
# the prices tend to not fluctuate too much

# ---------------------------------------------------------------
# JOINS OF DATAFRAMES
# Improvement suggestions: Takes a lot of time and needs to be faster configured.
# ---------------------------------------------------------------
def join_dfs(*dfs, join_key='listing_id', how='left'):
    if not dfs:
        raise ValueError("At least one DataFrame must be provided.")

    # Ensure join_key is set as index for optimization (optional)
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=join_key, how=how, copy=False, sort=False)
    return merged_df

def export_processed_data(arg):
    if len(arg) == 0:
        raise ValueError("At least one DataFrame must be provided.")
    arg.to_csv('./processed-data/data.csv', index=False)

nDf = join_dfs(clean_listings_df(), clean_reviews_df(), clean_calendar_df())
print(nDf.head())

# Improve: return?, or perhaps save the cleaned dataframes to csv files
# like this:
# listings_df.to_csv('data/cleaned_listings.csv', index=False)
# export_processed_data(nDf) # inperformant