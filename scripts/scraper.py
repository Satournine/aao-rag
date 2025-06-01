import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

DOWNLOAD_DIR = "data/raw"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Setup headless Chrome
options = Options()
options.add_argument("--headless")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 15)

print("[*] Navigating to AAO Decisions page...")
driver.get("https://www.uscis.gov/administrative-appeals/aao-decisions/aao-non-precedent-decisions")

def select_dropdown(label_text, value_text):
    print(f"[*] Selecting: {value_text}")
    dropdown = wait.until(EC.presence_of_element_located((
        By.XPATH, f"//span[@aria-label[contains(., '{label_text}')]]"
    )))
    dropdown.click()
    time.sleep(1)
    option = wait.until(EC.presence_of_element_located((
        By.XPATH, f"//li[contains(text(), '{value_text}')]"
    )))
    option.click()
    time.sleep(2)

# Step-by-step dropdown selections
select_dropdown("Decision Category", "I-140 - Immigrant Petition for Alien Worker (Extraordinary Ability)")
select_dropdown("Month", "February")
select_dropdown("Year", "2025")
select_dropdown("Rows per page", "100")

print("[*] Waiting for decision results to load...")
wait.until(EC.presence_of_element_located((By.CLASS_NAME, "views-row")))

# Grab all PDF links from rendered decision blocks
print("[*] Extracting PDF links...")
decision_rows = driver.find_elements(By.CLASS_NAME, "views-row")
pdf_links = []

for row in decision_rows:
    try:
        link_tag = row.find_element(By.TAG_NAME, "a")
        href = link_tag.get_attribute("href")
        title = link_tag.text.strip()
        if href and href.endswith(".pdf"):
            pdf_links.append((title, href))
    except Exception as e:
        print(f"[!] Failed to extract from a row: {e}")

print(f"[+] Found {len(pdf_links)} PDF decisions. Starting download...")

# Download each PDF
for title, url in pdf_links:
    name = title.replace(" ", "_").replace("/", "_") + ".pdf"
    filepath = os.path.join(DOWNLOAD_DIR, name)

    try:
        print(f"[*] Downloading: {name}")
        r = requests.get(url)
        with open(filepath, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"[!] Failed to download {url}: {e}")

driver.quit()
print("[✓] Done.")