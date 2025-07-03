import pandas as pd
import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# TODO: list of todos:
# - normalize (how and what to normalize?)
# - export cleaned dataframes somehow (either to csv or return them)
# - calendar dataframe: maybe aggregate future prices for each listingID to reduce size

# init dataframes
reviews_df = pd.read_csv('data/reviews.csv')
full_listings_df = pd.read_csv('data/listings.csv')

# method to fill NaN values with the mean of the column
def fill_na_with_mean(df, column):
    if column in df.columns:
        mean_value = df[column].mean()
        df.fillna({column: mean_value}, inplace=True)
    else:
        raise ValueError(f"Column {column} does not exist in the DataFrame.")

# ---------------------------------------------------------------
# CLEAN THE LISTINGS DATAFRAME
# TODO: boolean columns from t/f to 1/0?
# TODO: make types correct (dates as datetime etc)
# ---------------------------------------------------------------

def clean_listings_df(keep_text_columns=True):

    listings_df = full_listings_df.copy()

    # turn price column into float, removing dollar sign
    listings_df['price'] = listings_df['price'].str.replace('[$,]', '', regex=True).astype(float)
    # change id column name to listing_id for consistency with reviews_df
    listings_df.rename(columns={'id': 'listing_id'}, inplace=True)

    # remove rows with missing values in important columns
    listings_df.dropna(subset=['price', 'latitude', 'longitude', 'accommodates', 'bedrooms', 'beds', 'has_availability'], inplace=True)
    listings_df.fillna({'description':'no description'}, inplace=True)

    # remove columns with too many missing values or unnecessary information
    listings_df.drop(columns=[
        'neighborhood_overview', 'host_about', 'host_location',
        'host_response_time', 'host_response_rate', 'host_acceptance_rate',
        'host_is_superhost', 'host_neighbourhood', 'host_verifications',
        'host_thumbnail_url', 'license', 'calendar_updated',
        'calendar_last_scraped', 'last_review', 'first_review', 'neighbourhood_group_cleansed',
        'last_scraped', 'source', 'neighbourhood', 'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes',
        'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'scrape_id', 'host_picture_url',
        'host_listings_count', 'host_total_listings_count', 'host_has_profile_pic', 'host_identity_verified',
        'host_name', 'host_id', 'host_url', 'availability_eoy', 
    ], inplace=True)
    # if keep_text_columns is False, remove object columns
    if not keep_text_columns:
        listings_df.drop(columns=listings_df.select_dtypes(include=['object']).columns, inplace=True)


    # fill NaN values in review score columns with the mean of the column
    for col in ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                'review_scores_checkin', 'review_scores_communication',
                'review_scores_location', 'review_scores_value', 'reviews_per_month']:
        # fill NaN values with the mean of the column
        fill_na_with_mean(listings_df, col)

    
    # prepare all numeric columns for normalization
    numeric_cols = listings_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('listing_id')
    normalize_df(listings_df, numeric_cols)

    listings_df.info()

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
# NORMALIZE DATAFRAMES (Z-SCORE NORMALIZATION)
# ---------------------------------------------------------------

def normalize_df(df, columns_to_normalize):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    if not columns_to_normalize:
        raise ValueError("No columns provided for normalization.")

    for col in columns_to_normalize:
        if col not in df.columns:
            raise ValueError(f"Column {col} does not exist in the DataFrame.")

    # use the StandardScaler to normalize the specified columns
    # this will standardize the columns to have a mean of 0 and a standard deviation of 1
    scaler = StandardScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    
    return df



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
    arg.to_csv('./processed-data/clean-data.csv', index=False)

nDf = join_dfs(clean_listings_df(), clean_reviews_df())
print(nDf.head())
print("finished dataframe shape ", nDf.shape)

clean_listings_df(keep_text_columns=False).to_csv('./processed-data/clean-listings.csv', index=False)  # Clean listings without text columns
export_processed_data(nDf)
# TODO: the finished csv file looks kinda weird. get rid of the <br> and such?