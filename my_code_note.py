
# jupyter notebook:
# ![](https://imgur.com/RBJ5YSd.png)      

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''PANDA:'''


# Explore .`.`.`.`.`.`.`.`.`.`.`.`..`.`.`.`.``.`.`.`.`.`.`.`.`.`.`.`.`..`.`.`.`.`.``.`.`.`.`.`.`.`.`.`.`.`.`..`.`.`.`.
cols = ['User ID','Movie ID','Rating','Timestamp']
df = pd.read_csv('ml-100k/u.data', 
                    delimiter='\t', 
                    header=None, 
                    names=cols)
df.sample(5)

df.describe(include='all')

    #checking nan values + nice format/look
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)
'''isnull() and notnull()'''

    # Change value in specific column with specific conditions:
df.loc[df['action_detail'] == '-unknown-', ['action_type', 'action_detail']] = np.nan
    # Change value type while ignoring NaN - overwrite
df['age'] = df['age'].astype('Int64')

    # Countings:
'''series.count --> total sample count'''
'''series.value_counts() --> count of each value type'''

    # Compute dates differences:
df['date_account_created'] = pd.to_datetime(df['date_account_created'])
df['date_first_booking'] = pd.to_datetime(df['date_first_booking'])
df['days_book_account'] = (df['date_first_booking'] - df['date_account_created']).dt.days

    # Note:
'''dont DROP the column, after "engineered", select wanted column and copy()'''

# Plotting .`.`.`.`.`.`.`.`.`.`.`.`..`.`.`.`.``.`.`.`.`.`.`.`.`.`.`.`.`..`.`.`.`.`.``.`.`.`.`.`.`.`.`.`.`.`.`..`.`.`.`.

    # Alternative to Histogram:
df.groupby('StoreType')['Sales'].describe()
df.groupby('StoreType')['Customers', 'Sales'].sum()

    # Factorplot --> show trend, distribution and other statistics
sns.factorplot(data = df, x = 'Month', y = "Sales", 
                col = 'StoreType', # per store type in cols
                palette = 'plasma', hue = 'StoreType', row = 'Promo') # per promo in the store in rows

# Technique .`.`.`.`.`.`.`.`.`.`.`.`..`.`.`.`.``.`.`.`.`.`.`.`.`.`.`.`.`..`.`.`.`.`.``.`.`.`.`.`.`.`.`.`.`.`.`..`.`.`.`.
df[df['action_detail'].str.contains('book')]

df.apply(func_name)
    # Fillna in multiple columns
df.fillna({'action_detail':'-unknown-', 'action_type': '-unknown-'}, inplace=True)
    # Merge certain columns
df = df.merge(df2[['Key_Column','Target_Column']],on='Key_Column', how='left')

    # DateTime process + Feature engineering example
def split_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.isocalendar().week
    # df.drop(columns='Date', inplace=True)
def comp_months(df):
    df['CompetitionOpen'] = 12 * (df.Year - df.CompetitionOpenSinceYear) + (df.Month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df['CompetitionOpen'].map(lambda x: 0 if x < 0 else x).fillna(0) # map to eliminate those negative month

    # Pivot top "categ" value and create new dataframe
top_val = list(df['val'].value_counts().sort_values(ascending=False).index)[:70]
# Here, groupby sum the "secs" base on "by", then create new dataframe with "pivot_table()"
tmp_df = df[df['val'].isin(top_val)].groupby(by=['user_id','val'])["secs_elapsed"].sum().reset_index().pivot_table('secs_elapsed',index=['user_id'],columns='action')

# Set options .`.`.`.`.`.`.`.`.`.`.`.`..`.`.`.`.``.`.`.`.`.`.`.`.`.`.`.`.`..`.`.`.`.`.``.`.`.`.`.`.`.`.`.`.`.`.`..`.`.`.`.
pd.set_option('display.float_format', lambda x: f"{x:.2f}")
pd.set_option('display.max_rows', 500)

'''EDA'''

    # Plot according to date:
fig, axs = plt.subplots(2, 2, figsize=(25,5), gridspec_kw={'height_ratios': [1, 3]})
fig.suptitle('Bookings in 2012 and 2013')
mask = (df['date_'] > dt.datetime(2012, 1, 1)) & (df['date_'] <= dt.datetime(2012, 12, 31))
data = df['date_'].loc[(mask)&(...)].value_counts().resample('M').sum() # M mean months!
axs[0, 0].plot(data)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_and_evaluate(X_train, train_targets, X_val, val_targets, **params):
    model = XGBRegressor(random_state=42, n_jobs=-1, **params)
    model.fit(X_train, train_targets)
    train_rmse = rmse(model.predict(X_train), train_targets)
    val_rmse = rmse(model.predict(X_val), val_targets)
    return model, train_rmse, val_rmsea

kfold = KFold(n_splits=5)
models = []
for train_idxs, val_idxs in kfold.split(X):
    X_train, train_targets = X.iloc[train_idxs], targets.iloc[train_idxs]
    X_val, val_targets = X.iloc[val_idxs], targets.iloc[val_idxs]
    model, train_rmse, val_rmse = train_and_evaluate(X_train, 
                                                    train_targets, 
                                                    X_val, 
                                                    val_targets, 
                                                    max_depth=4, 
                                                    n_estimators=20)
    models.append(model)
    print('Train RMSE: {}, Validation RMSE: {}'.format(train_rmse, val_rmse))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
