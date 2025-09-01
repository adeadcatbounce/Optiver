# import libraries and packages

import pandas as pd
import os
import numpy as np
import lightgbm as lgbm
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set print options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Get current directory
current_dir = os.getcwd()

# Get training and test file names
train_file = "train.csv"
test_file = "test.csv"

# Combine current directory and filename
full_path_train = os.path.join(current_dir, train_file)
full_path_test = os.path.join(current_dir, test_file)

# Read CSV training and test dataset by creating a dataframe
df_train = pd.read_csv(full_path_train)
test = pd.read_csv(full_path_test)

# drop NAN values from the dataframe
df_train= df_train.dropna()

# Developing features
def engineered_features(df):

    # prices and sizes
    prices = ['reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    sizes = ['matched_size', 'ask_size', 'bid_size', 'imbalance_size']

    # total volume of stock
    df['volume'] = df.eval('ask_size + bid_size')

    # mid price of stock
    df['mid_price'] = df.eval('(ask_price + bid_price)/2')

    # size and volume features
    df['bid_ask_size_ratio'] = df['bid_size'] / df['ask_size']
    df['imbalance_bid_size_ratio'] = df['imbalance_size'] / df['bid_size']
    df['imbalance_ask_size_ratio'] = df['imbalance_size'] / df['ask_size']
    df['matched_size_ratio'] = df['matched_size'] / (df['bid_size'] + df['ask_size'])

    df['ref_wap_diff'] = df['reference_price'] - df['wap']
    df['bid_ask_spread'] = df['ask_price'] - df['bid_price']
    df['near_far_price_difference'] = df['far_price'] - df['near_price']

    # estimating price move relative to stock WAP at the beginning of the auction period
    df['wap_rate_of_change'] = df.groupby('stock_id')['wap'].pct_change()
    df['wap_momentum'] = df.groupby('stock_id')['wap'].diff()

    df['auction_start'] = (df['seconds_in_bucket'] == 0).astype(int)
    df['auction_end'] = (df['seconds_in_bucket'] == 550).astype(int)
    df['time_since_last_change'] = df.groupby('stock_id')['imbalance_buy_sell_flag'].diff(periods=1).ne(0).cumsum()
    df['time_since_auction_close'] = 600 - df['seconds_in_bucket']

    # liquidity imbalance is the difference between buy and sell orders
    # 1 denotes more buying than selling orders, so more buying pressure
    # -1 denotes more selling than buying orders, so more selling pressure
    # 0 denotes a balance between buy and sell orders

    df['liquidity_imbalance'] = df.eval('(bid_size - ask_size)/(bid_size + ask_size)')

    # matched imbalance is the difference between buy/sell orders after an amount of orders is executed
    # 1 indicates buying pressure, -1 indicates selling pressure, 0 indicates a balance
    df['matched_imbalance'] = df.eval('(imbalance_size - matched_size) / (matched_size + imbalance_size)')

    df['size_imbalance'] = df.eval('bid_size / ask_size')
    df['imbalance_momentum'] = df.groupby('stock_id')['imbalance_size'].diff(periods=1) / df['matched_size']
    df['price_spread'] = df['ask_price'] - df['bid_price']
    df['spread_intensity'] = df.groupby('stock_id')['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size']*(df['ask_price'] - df['bid_price'])
    df['depth_pressure'] = (df['ask_price'] - df['bid_price']) * (df['far_price'] - df['near_price'])

    # compute statistical metrics and add them as new columns to the dataframe, for prices and sizes
    for func in ['mean', 'std', 'skew', 'kurt']:
        df[f'all_prices_{func}'] = df[prices].agg(func, axis=1)
        df[f'all_sizes_{func}'] = df[sizes].agg(func, axis=1)

    # compute lagged value and percentage changes for indicators
    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1,2, 3, 10]:
            df[f'{col}_shift_{window}'] = df.groupby('stock_id')[col].shift(window)
            df[f'{col}_ret_{window}'] = df.groupby('stock_id')[col].pct_change(window)

    # developing estimations of metrics
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['imbalance_ratio'] = df['imbalance_size']/df['matched_size']
    df['wap_askprice_diff'] = df['ask_price'] - df['wap']
    df['wap_bidprice_diff'] = df['bid_price'] - df['wap']

    df['wap_askprice_diff_urg'] = df['wap_askprice_diff'] - df['liquidity_imbalance']
    df['wap_bidprice_diff_urg'] = df['wap_bidprice_diff'] - df['liquidity_imbalance']
    df['bid_size_ask_size_diff'] = df.eval('(bid_size - ask_size) / (bid_size + ask_size)')
    df['imbalance_size_matched_size_diff'] = df.eval('(imbalance_size - matched_size) / (matched_size + imbalance_size)')

    # target is the 60-second future move in the wap of the stock minus the 60-second future move
    # in the synthetic index
    df['day_of_week'] = df['date_id'] % 5
    df['seconds'] = df['seconds_in_bucket'] % 60
    df['minute'] = df['seconds_in_bucket'] // 60

    global_stock_id_feats = {
    "median_size" : df.groupby('stock_id')['bid_size'].median() + df.groupby('stock_id')['ask_size'].median()
    ,"std_size": df.groupby('stock_id')['bid_size'].std() + df.groupby('stock_id')['ask_size'].std()
    ,"ptp_size": df.groupby('stock_id')['bid_size'].max() - df.groupby('stock_id')['ask_size'].min()
    ,"median_price": df.groupby('stock_id')['bid_price'].median() + df.groupby('stock_id')['ask_price'].median()
    ,"std_size" : df.groupby('stock_id')['bid_price'].std() + df.groupby('stock_id')['ask_price'].std()
    ,"ptp_size" : df.groupby('stock_id')['bid_price'].max() - df.groupby('stock_id')['ask_price'].min()}

    for key, value in global_stock_id_feats.items():
        df[f'global_{key}'] = df['stock_id'].map(value.to_dict())

    return df.replace([np.inf, -np.inf], 0)

# Use the function written above to get features and results
df_train = engineered_features(df_train)

# Specify lightgbm model
# drop dependent variable (target) from the dataframe
X = df_train.drop(['target', 'row_id'], axis=1)
y = df_train['target']

lgb_model = lgbm.LGBMRegressor()
lgb_model.fit(X, y)
pred = lgb_model.predict(X)

# Evaluate quality of predictions using statistical metrics
mse = mean_absolute_error(y, pred)
rmse = mean_squared_error(y, pred, squared=False)
mae = mean_absolute_error(y, pred)
r2 = r2_score(y, pred)

print(f"MSE : {mse}")
print(f"RMSE : {rmse}")
print(f"MAE : {mae}")
print(f"R_Squared : {r2}")

# Consolidate actual (y) and predicted values in a dataframe
results = pd.DataFrame({"Actual" : y, "Predicted" : pred})
results[['Actual', 'Predicted']] = results[['Actual', 'Predicted']].round(2)
results = results[(results['Actual'] >= -75.00) & (results['Actual'] <= 75.00)]
# Use a fraction of the results dataframe
sampled_results = results.sample(frac=2/10, random_state=42)


# Visualize actual values and predictions
sns.set_style('darkgrid')
plt.title('Actual and Predicted Values of Target Variable'
          '\n Target Variable = 60 second future move, WAP Stock - 60 second future move, WAP Index'
          '\n Model Used : LightGBM, Gradient Boost', fontsize=10)
plt.scatter('Actual', 'Predicted', data=sampled_results, s=10, facecolor = 'none', edgecolor = 'blue', alpha=1/2)
plt.xlabel('Actual Values of Target Variable', fontsize=12)
plt.ylabel('Predicted Values of Target Variable', fontsize=12)
plt.xticks(np.arange(-75,76,25))
plt.yticks(np.arange(-75,76,25))
plt.tight_layout()
#plt.savefig('actual_versus_predicted.png', dpi=1000)
plt.show()

# Plot feature importance
sns.set_style('darkgrid')
lgbm.plot_importance(lgb_model, importance_type = "split", max_num_features=10)
plt.title("Top Ten Feature Importance, by Split \n Using Light Gradient Boosting Machine LightGBM")
plt.tight_layout()
#plt.savefig('feature_importance.png', dpi=500)
plt.show()

# Predict using test dataset
test_df = engineered_features(test)
test_df = test_df.drop(['currently_scored', 'row_id'], axis=1)
test_predictions = np.array(lgb_model.predict(test_df))


















