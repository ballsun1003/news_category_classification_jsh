# 파일명은 naver_news_section.csv로 해주세요.
# 컬럼명은 titles, category로 해주세요.
# 00님이 정치, 경제
# 01님이 사회, 문화
# 02님이 세계, IT
# 다 되면 PR부탁합니다.(Pull request)


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import datetime


options = ChromeOptions()
options.add_argument('lang=ko_KR')
options.add_argument('headless')          #'headless' 작업 중인 브라우저를 보이지 않게 해주는 기능

service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

url = 'https://news.naver.com/section/102'
driver.get(url)
button_xpath = '//*[@id="newsct"]/div[4]/div/div[2]/a'
for i in range(31):
    driver.find_element(By.XPATH, button_xpath).click()
    time.sleep(0.5)
time.sleep(5)

#/html/body/div/div[2]/div[2]/div[2]/div[4]/div/div[2]/a            #기사더보기 XPath
#/html/body/div/div[2]/div[2]/div[2]/div[4]/div/div[1]/div[7]/ul/li[2]/div/div/div[2]/a/strong          #기사제목 XPath

titles = []
for i in range(1, 220):
    for j in range(1, 7):
        try:
            title_xpath = '//*[@id="newsct"]/div[4]/div/div[1]/div[{}]/ul/li[{}]/div/div/div[2]/a/strong'.format(i, j)
            title = driver.find_element(By.XPATH, title_xpath).text
            print(title)
            titles.append(title)
        except:
             print('error', i, j)

df_titles = pd.DataFrame(titles, columns=['titles'])
df_titles['category'] = 'Social'

df_titles.to_csv('./data/naver_news_section_social.csv', index=False)
