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

options = ChromeOptions()
options.add_argument('lang=ko_KR')
options.add_argument('headless') # 새창이 띄워지는 것을 안보고 싶으면 이 코드를 살리면 된다.

service = ChromeService(executable_path=ChromeDriverManager().install()) # 크롬 드라이버 설치
driver = webdriver.Chrome(service=service, options=options) # 서비스로 만들고

# url = 'https://news.naver.com/section/100'
# driver.get(url)
# # 해당 url로 브라우저가 뜬다.
# # 그냥 두면 창이 닫힌다. 그래서 딜레이가 필요하다.
# time.sleep(5)

# 더보기 누르고 읽기하면 된다.

# 더보기 버튼을 어떻게 누를까?
# 버튼의 xpath를 가져온다. html상의 위치정보이다.
url = 'https://news.naver.com/section/100'
driver.get(url)
button_path = '//*[@id="newsct"]/div[4]/div/div[2]/a'

for i in range(6):
    driver.find_element(By.XPATH, button_path).click() # click으로 누름
    # 버튼이 새로 생길 때까지 딜레이가 필요함
    time.sleep(0.5)

# time.sleep(5) # 아래 코드 까지 실행시 주석

# 다른 위치를 볼 것
# '//*[@id="newsct"]/div[4]/div/div[1]/div[27]/ul/li[4]/div/div/div[2]/a/strong'
# '//*[@id="newsct"]/div[4]/div/div[1]/div[27]/ul/li[6]/div/div/div[2]/a/strong'
# '//*[@id="newsct"]/div[4]/div/div[1]/div[31]/ul/li[3]/div/div/div[2]/a/strong'
# div[]/ul/li[] 규칙인듯

# 5 37
# 6 43.
# 대충 위에 *7 + 5정도

for i in range(1,51): # 40까지
    for j in range(1,7):
        try:
            title_xpath = '//*[@id="newsct"]/div[4]/div/div[1]/div[{}]/ul/li[{}]/div/div/div[2]/a/strong'.format(i,j)
            title = driver.find_element(By.XPATH, title_xpath).text
            print(title)
        except:
            print('error',i,j) # 3이 없다.

