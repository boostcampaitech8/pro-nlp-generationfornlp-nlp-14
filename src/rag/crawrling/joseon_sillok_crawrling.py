import csv
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# CSV 파일 초기 설정 (헤더 작성)
file_name = "./data/chosun_sillok.csv"
with open(file_name, mode="a", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["한글명칭", "시기", "제목", "내용"])

options = webdriver.ChromeOptions()
# options.add_argument('--headless') # 창을 띄우지 않고 실행하려면 주석 해제
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

main_url = "https://sillok.history.go.kr/main.do"
driver.get(main_url)
wait = WebDriverWait(driver, 10)

# [Step 1] 왕 목록
main_items = wait.until(
    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ul.view-list li.item a.anchor"))
)
total_main = len(main_items)

for i in range(2, total_main):
    main_items = wait.until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ul.view-list li.item a.anchor"))
    )
    target_king = main_items[i]
    king_name = target_king.text
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_king)
    target_king.click()
    print(f"\n>>> [{i + 1}/{total_main}] {king_name} 페이지 진입")
    time.sleep(3)

    # [Step 2] 월 목록 (month-box)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "month-box")))
    month_boxes_elements = driver.find_elements(By.CSS_SELECTOR, ".month-box:not(.no-list)")
    total_boxes = len(month_boxes_elements)

    for j in range(total_boxes):
        current_boxes = driver.find_elements(By.CSS_SELECTOR, ".month-box:not(.no-list)")
        month_links = current_boxes[j].find_elements(By.CSS_SELECTOR, ".list-wrap a")
        total_months = len(month_links)

        for k in range(total_months):
            current_boxes = driver.find_elements(By.CSS_SELECTOR, ".month-box:not(.no-list)")
            target_month = current_boxes[j].find_elements(By.CSS_SELECTOR, ".list-wrap a")[k]
            month_name = target_month.text
            print(f"  - {month_name} 기사 목록 진입")

            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_month)
            target_month.click()
            time.sleep(2)
            # [Step 3] 개별 기사(item) 순회
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.list li.item")))
                articles_elements = driver.find_elements(By.CSS_SELECTOR, "ul.list li.item a")
                total_articles = len(articles_elements)
                print(f"    * {total_articles}개의 기사 순회 시작")

                for idx in range(4, total_articles):
                    articles = driver.find_elements(By.CSS_SELECTOR, "ul.list li.item a")
                    target_article = articles[idx]
                    article_title = target_article.text

                    driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'});", target_article
                    )
                    target_article.click()
                    time.sleep(1)

                    # --- [수집 구간] ---
                    # 예: 기사 제목과 본문 출력
                    try:
                        print(f"{article_title}에 들어왔다!! 제발 들어가라 젭ㅈㅂ...")
                        date = driver.find_element(By.CLASS_NAME, "date").text
                        parent_content_xpath = "/html/body/div[2]/div/div[1]/main/div[2]/div/div/div[2]/div[1]/div[3]/div[1]/div"
                        paragraph_content = driver.find_elements(
                            By.XPATH, f"{parent_content_xpath}//p[contains(@class, 'paragraph')]"
                        )
                        contents = ""
                        for i in paragraph_content:
                            contents += i.text

                        # --- 데이터 저장 (추가된 부분) ---
                        with open(file_name, mode="a", encoding="utf-8-sig", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([king_name, date, article_title, contents])

                        print(f"내용: {contents}")
                        print(f"날짜: {date}")
                        print(
                            f"    ({idx + 1}/{total_articles}) 수집 완료: {article_title[:20]}..."
                        )
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"except: {e}")
                        print(f"{article_title}에서 except 발생")

                    driver.back()  # 기사 상세 -> 기사 목록
                    time.sleep(1.5)
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.list li.item")))

            except Exception as e:
                print(f"    ! 기사 목록 처리 중 오류: {e}")

            driver.back()  # 기사 목록 -> 왕 상세(월 선택)
            time.sleep(1)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "month-box")))

    driver.back()  # 왕 상세 -> 메인 목록
    time.sleep(1)
    wait.until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ul.view-list li.item a.anchor"))
    )
    print(f"<<< {king_name} 완료 및 메인 복귀")

print("\n전체 순회가 완료되었습니다.")
