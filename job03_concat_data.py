import pandas as pd

df = pd.read_csv('data/naver_news_Culture_20260605.csv')
print(df.head())

df_temp = pd.read_csv('data/naver_news_Economic_20260605.csv')
print(df_temp.head())
df = pd.concat([df,df_temp], ignore_index=True)
df_temp = pd.read_csv('data/naver_news_IT_20260605.csv')
print(df_temp.head())
df = pd.concat([df,df_temp], ignore_index=True)
df_temp = pd.read_csv('data/naver_news_Politics_20260605.csv')
print(df_temp.head())
df = pd.concat([df,df_temp], ignore_index=True)
df_temp = pd.read_csv('data/naver_news_Society_20260605.csv')
print(df_temp.head())
df = pd.concat([df,df_temp], ignore_index=True)
df_temp = pd.read_csv('data/naver_news_World_20260605.csv')
print(df_temp.head())
df = pd.concat([df,df_temp], ignore_index=True)

df['category'] = df['category'].replace({
    'Economics': 'Economic',
    'IT': 'IT_Science'
})
df = df.drop_duplicates()
print(df.category.value_counts())
print(df.isnull().sum())
df.info()
df.to_csv('./data/news_titles.csv', index=False)
