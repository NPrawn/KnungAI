from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, os, re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pdfkit
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError
import time
import base64
import shutil
import pandas as pd
# wkhtmltopdf 설치 경로
config = pdfkit.configuration(wkhtmltopdf=r"/usr/local/bin/wkhtmltopdf")

# PDF 저장 디렉토리
SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
os.makedirs(SAVE_DIR, exist_ok=True)
download_dir = SAVE_DIR
#os.makedirs(download_dir, exist_ok=True)
# 웹드라이버 설정
options = Options()
#options.add_argument("--headless=new")  # 창 없이 실행
options.add_argument("--no-sandbox")
#options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")

options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

def sanitize_filename(title):
    return re.sub(r'[\\/*?:"<>|]', "_", title)



def merge_pdfs(output_path, file_count):
    # PDF 병합을 위한 PdfWriter 객체 생성
    merger = PdfWriter()

    # 각 PDF 파일을 열어서 페이지를 병합
    pdf_folder = os.path.join(os.getcwd(), '../data')
    pdf_list = [os.path.join(pdf_folder, f'output{i}.pdf') for i in range(1,file_count)]

    
    for pdf in pdf_list:
        try:
            with open(pdf, "rb") as f:
                reader = PdfReader(f)
                tmp = open(pdf,"rb")
                merger.append(tmp)
        except (PdfReadError, Exception) as e:
            print(f"[SKIPPED] Invalid PDF: {pdf} ({e})")
            continue
    # 병합된 PDF를 새로운 파일로 저장
    with open(output_path, 'wb') as output_pdf:
        merger.write(output_pdf)



def attach_download(link, file_count):
    global driver
    driver.execute_script("window.open(arguments[0]);", link)
    driver.switch_to.window(driver.window_handles[-1])
    #print("첨부파일 다운받을게요 "+link)
    time.sleep(2)
    # 4. 스크롤 다운으로 모든 콘텐츠 로딩
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_pause_time = 1
    while True:
        # 스크롤 맨 아래로
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)
        
        # 새 높이 계산
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break  # 더 이상 스크롤할 부분 없음
        last_height = new_height
    # 5. PDF 저장 (printToPDF 사용)
    pdf = driver.execute_cdp_cmd("Page.printToPDF", {
        "landscape": False,
        "printBackground": True,
        "preferCSSPageSize": False,
        "paperWidth": 8.27,
        "paperHeight": 11.69,
        "marginTop": 0,
        "marginBottom": 0,
        "marginLeft": 0,
        "marginRight": 0,
        "scale": 1.0
    })

    # 6. 저장
    filename = os.path.join(SAVE_DIR, f"output{file_count}.pdf")
    with open(filename, "wb") as f:
        f.write(base64.b64decode(pdf['data']))
    #print("첨부파일 저장완료")
    driver.close()
    driver.switch_to.window(driver.window_handles[-1])
    time.sleep(2)



def crawl_notice_board(notice_url):
    driver.get(notice_url)
    time.sleep(2)  # 페이지 로딩 대기

    page_num = 1
    # 게시글 목록 가져오기 (예시 CSS 셀렉터 - 실 페이지에 맞게 수정해야 함)
    while True:
        if page_num == 5:
            print("지정한 페이지까지 글을 내려받았습니다.")
            break
        post_links = driver.find_elements(By.CSS_SELECTOR, "td.td-subject a")
        if len(post_links) == 0:
            print("더이상 현재 게시판에 글이 없습니다.")
            break
        for idx, post in enumerate(post_links):
            print(f"----------------현재 페이지 {page_num}, {idx}번째 게시글")
            link_title = post.text.strip()
            link = post.get_attribute("href")
            driver.execute_script("window.open(arguments[0]);", link)
            driver.switch_to.window(driver.window_handles[1])
            time.sleep(2)

            if os.path.exists(os.path.join(SAVE_DIR, sanitize_filename(link_title[:50]) + ".pdf")):
                print(f"[skipped]파일이 이미 존재합니다: {link_title[:10]}")
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                continue

            soup = BeautifulSoup(driver.page_source, "html.parser")

            # 게시글 본문과 첨부파일 추출
            title = soup.select_one("body > div > div.board-view-info > div.view-info > h2")
            title = title.text.strip()
            content = soup.select_one("div.view-con")
            date = soup.select_one("div.board-view-info div.view-detail div.view-util dl.write dd")
            date = date.text.strip()
            #attachment_links = soup.select("body > div > div.view-file > dl > dd > ul > li > a:nth-child(1)")  # 첨부파일 셀렉터
            synap_links_raw = soup.select("body > div > div.view-file > dl > dd > ul > li > a.prev")  # 첨부파일 셀렉터
            attachment_links_raw = driver.find_elements(By.CSS_SELECTOR, 'a[title="파일 다운로드"]')

            file_names = [attachment.text.strip() for attachment in attachment_links_raw]
            synap_links = [
                urljoin(driver.current_url, a.get("href")) for a in synap_links_raw
            ]
            
            #print("첨부파일 링크들: ", synap_links)
            print("attachments: ", file_names)
            #print("raw파일들: ", attachment_links_raw)
            """
            attachments = driver.find_elements(By.CSS_SELECTOR, 'a[title="파일 다운로드"]')
            for file_link in attachments:
                print(f"hello: {file_link}")
                file_link.click()
                time.sleep(5)  # 다운로드 대기 시간
            """
            # HTML로 PDF 만들기
            html = f"""
            <head>
            <meta charset="UTF-8">
            <style>
                img {{
                    max-width: 100%;
                    height: auto;
                    page-break-inside: avoid;
                }}
            </style>
            </head>
            <h1>{title}</h1>
            <p><strong>URL:</strong> <a href="{link}">{link}</a> 작성일자:{date}</p>
            {str(content)}
            <p><strong>Attachments:</strong></p>
            <ul>
            {''.join(f'<li><a href="{a}">{a}</a></li>' for a in synap_links)}
            </ul>
            """
            pdf_origin = os.path.join(SAVE_DIR, "output1.pdf")

            pdfkit.from_string(html, pdf_origin, configuration=config)

            file_count=2

            for link, name, file_url in zip(synap_links, file_names, attachment_links_raw):
                ext = name.split('.')[-1].lower()
                if ext == 'pdf':
                    #print(f"{name}은 PDF 파일입니다. 링크: {link}")
                    file_url.click()
                    time.sleep(10)

                    str_file_count = str(file_count)
                    new_file_name = max([download_dir + "/" + f for f in os.listdir(download_dir)],key=os.path.getctime)
                    shutil.move(new_file_name,os.path.join(download_dir,f"output{file_count}.pdf"))
                    new_file_name = download_dir + "/" + f"output{file_count}.pdf"
                    #print(f'{new_file_name}')
                    #attach_download(link, file_count)
                    file_count+=1

                elif ext == 'hwp':
                    #print(f"{name}은 HWP 파일입니다. 링크: {link}")
                    driver.execute_script("window.open(arguments[0]);", link)
                    driver.switch_to.window(driver.window_handles[2])
                    try:
                        iframe = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "iframe.wrap__innerWrap")))
                    except TimeoutException:
                        print("[skipped]열 수 없는 한글파일입니다.")
                        driver.close()
                        driver.switch_to.window(driver.window_handles[-1])
                        continue
                    # iframe src 추출
                    iframe_src = iframe.get_attribute("src")
                    # 절대 URL 만들기
                    full_url = urljoin(driver.current_url, iframe_src)
                    driver.close()
                    driver.switch_to.window(driver.window_handles[-1])
                    attach_download(full_url, file_count)
                    file_count+=1
                elif ext == 'png' or ext == 'jpg':
                    #print(f"{name}은 이미지 파일입니다. 링크: {link}")
                    attach_download(link, file_count)
                    file_count+=1
                elif ext == 'xlsx':
                    #print(f"{name}은 엑셀 파일입니다  링크: {link}")
                    file_url.click()
                    time.sleep(5)
                    new_file_name = max([download_dir + "/" + f for f in os.listdir(download_dir)],key=os.path.getctime)
                    shutil.move(new_file_name,os.path.join(download_dir,r"tmpExcel.xlsx"))
                    new_file_name = download_dir + "/" + r"tmpExcel.xlsx"
                    
                    dfs = pd.read_excel(os.path.join(download_dir,r"tmpExcel.xlsx"), sheet_name=None)
                    for sheet_name, df in dfs.items():
                        filename = os.path.join(download_dir,f"{sanitize_filename(link_title[:50])}_{name}_{sheet_name}.csv")
                        df.to_csv(filename, index=False, encoding="utf-8-sig")
                    os.remove(new_file_name)
                    
                    
                else:
                    print(f"{name}은 처리하지 않는 파일입니다 ex) zip .... 링크: {link}")
                    # 기타 파일 처리
            ##merge
            merge_filename = os.path.join(SAVE_DIR, sanitize_filename(link_title[:50]) + ".pdf")
            merge_pdfs(merge_filename, file_count)
            print(f"Saved: {merge_filename}")
            for i in range(1, file_count):
                os.remove(os.path.join(download_dir,f"output{i}.pdf"))
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
        next_button = driver.find_element(By.CSS_SELECTOR, "a[href*='page_link']")
        if next_button:
            page_script = next_button.get_attribute('href')
            page_num += 1
            driver.execute_script(f"page_link({page_num});")
            time.sleep(2)  # 페이지 전환 대기
        else:
            print("더 이상 다음 페이지가 없습니다.")
            break



crawl_notice_board("https://www.kongju.ac.kr/KNU/16909/subview.do") # 정보광장 - 학생소식
crawl_notice_board("https://www.kongju.ac.kr/KNU/16828/subview.do") # 대학생활 - 서식
crawl_notice_board("https://www.kongju.ac.kr/KNU/16910/subview.do") # 정보광장 - 행정소식
crawl_notice_board("https://computer.kongju.ac.kr/ZD1140/11607/subview.do") # 컴공 - 학과공지
crawl_notice_board("https://ai.kongju.ac.kr/ZK0120/6358/subview.do") # 인공지능학부 - 학부공지

driver.quit()
