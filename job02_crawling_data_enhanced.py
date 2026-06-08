import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

max_iteration = 30 #더보기 누를 최대횟수(이 수와 관계없이 끝에 도달하면 자동 정지)
category_sel = 3 #카테고리 선택
li_per_div = 6 #div당 li수
div_begin_num = 7 #처음 나오는 div수
div_per_click = 6 #더보기 누를때마다 나오는 div수
category = ['Politics', 'Economic', 'Society', 'Culture', 'World', 'IT']
if category_sel == 1:
    first_div = 5 #경제일때는 첫번째 div다름
else:
    first_div = 4

options = ChromeOptions()
options.add_argument("lang=ko_KR")
options.add_argument("headless")

service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

url = 'https://news.naver.com/section/10{}'.format(category_sel)
driver.get(url)
button_xpath = '//*[@id="newsct"]/div[{}]/div/div[2]/a'.format(first_div)
click_num = 0
for i in range(max_iteration):
    try:
        button = driver.find_element(By.XPATH, button_xpath)
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
        button.click()

        print(i + 1, "번 클릭 성공")
        click_num += 1
        time.sleep(0.5)

    except:
        print(i + 1, "번 클릭 실패. 3초 기다렸다 재시도")
        time.sleep(3)

        try:
            button = driver.find_element(By.XPATH, button_xpath)
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
            time.sleep(0.3)
            button.click()

            print(i + 1, "번 재시도 성공")
            click_num += 1
            time.sleep(0.5)

        except:
            print(i + 1, "번 재시도도 실패. 중단")
            break
time.sleep(3)

df_titles = pd.DataFrame()
titles = []
for i in range(1, (click_num*div_per_click)+div_begin_num+1):
    for j in range(1, li_per_div+1):
        try:
            title_xpath = '//*[@id="newsct"]/div[{}]/div/div[1]/div[{}]/ul/li[{}]/div/div/div[2]/a/strong'.format(first_div, i, j)
            title = driver.find_element(By.XPATH, title_xpath).text
            print(title)
            titles.append(title)
        except:
            print('error in',i,j)
df_section_titles = pd.DataFrame(titles, columns=['title'])
df_section_titles['category'] = category[category_sel]
df_titles = pd.concat([df_titles, df_section_titles], ignore_index=True)
print(df_titles.head())
df_titles.info()
df_titles.to_csv('./data/naver_news_{}_{}.csv'.format(category[category_sel],datetime.datetime.now().strftime('%Y%m%d')), index=False)