from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import datetime
# re, datetime은 내장 라이브러리



category = ['Politics','Economic','Social','Culture','World','IT']
df_titles = pd.DataFrame()

for i in range(0,len(category)):
    url = 'https://news.naver.com/section/10{}'.format(i)


    resp = requests.get(url)
    # print(list(resp))

    soop = BeautifulSoup(resp.text, 'html.parser')
    # print(soop) # html파일 그대로 쭉 나옴

    soop.select('.sa_text_strong')
    title_tag = soop.select('.sa_text_strong')
    # print(title_tag)
    # print(title_tag[0].text) # 뉴스의 헤드기사 가져왔음

    titles = []
    for title in title_tag:
        titles.append(title.text)
    print(titles)

    df_section_titles = pd.DataFrame(titles, columns=['titles'])
    df_section_titles['category'] = category[i] # i번째 카테고리 이름 가져오기
    df_titles = pd.concat([df_titles,df_section_titles], ignore_index=True) # IT, Economy등이 들어간다.

print(df_titles)
df_titles.info()
# 저장
df_titles.to_csv('./data/naver_headline_news_{}'.format(datetime.datetime.now().strftime('%Y%m%d')), index=False)


