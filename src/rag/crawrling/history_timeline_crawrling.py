import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# 1. 브라우저 설정
chrome_options = Options()
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# 2. 메인 페이지 접속
url = "https://db.history.go.kr/diachronic/level.do?itemId=ch"
driver.get(url)
wait = WebDriverWait(driver, 10)
main_window = driver.current_window_handle

results = []

try:
    # '연표해설' 버튼 목록 가져오기
    buttons = wait.until(
        EC.presence_of_all_elements_located((By.XPATH, "//a[contains(text(), '연표해설')]"))
    )
    total_buttons = len(buttons)
    print(f"총 {total_buttons}개의 연표해설 버튼")

    for i in range(total_buttons):
        # 페이지 갱신을 대비해 버튼 요소를 매번 다시 찾음
        current_buttons = driver.find_elements(By.XPATH, "//a[contains(text(), '연표해설')]")
        target_btn = current_buttons[i]

        # 자바스크립트로 클릭 (ElementNotInteractableException 방지)
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_btn)
        time.sleep(1)
        driver.execute_script("arguments[0].click();", target_btn)

        # 3. 새 창으로 전환
        time.sleep(2)
        new_window = None
        for handle in driver.window_handles:
            if handle != main_window:
                new_window = handle
                break

        if new_window:
            driver.switch_to.window(new_window)

            # 4. 이미지 구조 기반 데이터 추출 (td.txtVeiw 내의 b와 본문)
            try:
                items = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "txtVeiw")))

                for item in items:
                    try:
                        # 1) 제목(b 태그) 추출
                        title_elem = item.find_element(By.TAG_NAME, "b")
                        title = title_elem.text.strip()

                        # 2) 본문 추출
                        context = driver.execute_script(
                            """
                            var parent = arguments[0];
                            var child = parent.querySelector('b');
                            var text = parent.innerText;
                            if (child) {
                                return text.replace(child.innerText, '').trim();
                            }
                            return text.trim();
                        """,
                            item,
                        )

                        if title:
                            results.append({"title": title, "context": context})
                    except Exception as e:
                        print(f"데이터 추출 중 오류 발생: {e}")

                print(f"[{i + 1}/{total_buttons}] 팝업 데이터 수집 완료")

            except Exception as e:
                print(f"[{i + 1}] 팝업 내용 로딩 실패 또는 데이터 없음: {e}")

            # 5. 창 닫고 메인 창으로 복귀
            driver.close()
            driver.switch_to.window(main_window)
            time.sleep(1)

finally:
    # 6. 브라우저 종료 및 파일 저장
    driver.quit()

    if results:
        df = pd.DataFrame(results)

        # 엑셀 저장
        df.to_excel("./data/history_data_final.xlsx", index=False)

        # JSON 저장 (한글 깨짐 방지 설정)
        df.to_json("./data/history_data_final.json", orient="records", force_ascii=False, indent=4)

        print("\n" + "=" * 30)
        print(f"총 {len(df)}개의 데이터 행 저장.")
        print("파일명: history_data_final.xlsx, history_data_final.json")
        print("=" * 30)
    else:
        print("수집된 데이터가 없습니다.")
