import json
import os
import time
import glob

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, \
    ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager

F1 = open('stage_to_EFR_products.json',)
F2 = open('stage_to_WFR_products.json',)

URL = "https://finder.creodias.eu"

DOWNLOADS_PER_SESSION_LIMIT = 20

IDENTIFIER_INPUT_ID = "product-identifier"
SEARCH_BUTTON_ID = "search-button"
PRODUCT_LINK_CLASS = "main-table-link"
DOWNLOAD_BUTTON_ID = "product-download-link"

LOGIN_BUTTON_ID = "login-button"
USERNAME_INPUT_ID = "username"
PASSWORD_INPUT_ID = "password"
KC_LOGIN_ID = "kc-login"

DOWNLOAD_PATH = "G:\\MGSTR\\source_data"

ALERT_MODAL_ID = "alert-modal"
LOGOUT_ALERT_MESSAGE = "You must log in to download/order products"


def new_browser(headless=True, download_path=None):
    """ Helper function that creates a new Selenium browser """
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument('headless')
    if download_path is not None:
        prefs = dict()
        prefs["profile.default_content_settings.popups"] = 0
        prefs["download.default_directory"]=download_path
        options.add_experimental_option("prefs", prefs)
    return webdriver.Chrome(options=options, executable_path=ChromeDriverManager().install())


def find_element(method, identifier):
    count = 0
    threshold = 1000
    element = None
    while element is None:
        try:
            element = method(identifier)
        except (NoSuchElementException, StaleElementReferenceException) as e:
            if count > threshold:
                raise e
            count += 1
    return element


def interact(lookup_method, identifier, method, value=None):
    element = find_element(lookup_method, identifier)
    interacted = False
    click_count = 0
    click_threshold = 1000
    stale_count = 0
    stale_threshold = 5
    while not interacted:
        try:
            if value is None:
                getattr(element, method)()
                interacted = True
            else:
                getattr(element, method)(value)
                interacted = True
        except ElementClickInterceptedException as e:
            if click_count > click_threshold:
                raise e
            click_count += 1
        except StaleElementReferenceException as e:
            element = find_element(lookup_method, identifier)
            if stale_count > stale_threshold:
                raise e
            stale_count += 1


def log_in():
    interact(BROWSER.find_element_by_id, LOGIN_BUTTON_ID, "click")
    interact(BROWSER.find_element_by_id, USERNAME_INPUT_ID, "send_keys", "228097@student.pwr.edu.pl")
    interact(BROWSER.find_element_by_id, PASSWORD_INPUT_ID, "send_keys", "Kurw@Mac#")
    interact(BROWSER.find_element_by_id, KC_LOGIN_ID, "click")


def download_product(product_identifier):
    interact(BROWSER.find_element_by_id, IDENTIFIER_INPUT_ID, "clear")
    interact(BROWSER.find_element_by_id, IDENTIFIER_INPUT_ID, "send_keys", product_identifier)
    interact(BROWSER.find_element_by_id, SEARCH_BUTTON_ID, "click")
    interact(BROWSER.find_element_by_class_name, PRODUCT_LINK_CLASS, "click")
    interact(BROWSER.find_element_by_id, DOWNLOAD_BUTTON_ID, "click")


def check_for_sign_out():
    time.sleep(1)
    alert_modal = BROWSER.find_element_by_id(ALERT_MODAL_ID)
    if alert_modal.is_displayed():
        return LOGOUT_ALERT_MESSAGE in alert_modal.text
    return False


def move_products_to_download_folder(batch, sub_directory):
    files = [product["name"].replace("SEN3", "zip") for product in batch]
    threshold = 720
    count = 0
    while len(files) > 0:
        errors = 0
        files_copy = list(files)
        for file_name in files_copy:
            try:
                file = glob.glob(os.path.join(DOWNLOAD_PATH, file_name))
                file_destination = os.path.join(DOWNLOAD_PATH, sub_directory, file_name)
                os.rename(file[0], file_destination)
                files.remove(file_name)
            except:
                errors += 1
        if errors > 0:
            if count > threshold:
                print("Download time out!")
                break
            count += 1

            BROWSER.refresh()
            print(f"Page refreshed {count}/{threshold}")

            time.sleep(60)

    # file = None
    # threshold = 100
    # count = 0
    # while file is None:
    #     try:
    #         file = glob.glob(os.path.join(DOWNLOAD_PATH, file_name))
    #     except Exception as e:
    #         if count > threshold:
    #             raise e
    #         count += 1
    #         time.sleep(60)
    #
    # file_destination = os.path.join(DOWNLOAD_PATH, sub_directory, file_name)
    # os.rename(file[0], file_destination)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def filter_downloads(products):
    existing_products = dict()
    missing_products = dict()
    for stage in products.keys():
        existing_products[stage] = set([prod.replace(".zip", ".SEN3") if ".zip" in prod else prod for prod in os.listdir(os.path.join(DOWNLOAD_PATH, stage))])
        missing_products[stage] = list(
            filter(lambda product: product["name"] not in existing_products[stage], products[stage])
        )
    return missing_products


def run_download(file):
    stages = json.load(file)
    products_to_download = filter_downloads(stages)
    print(products_to_download.values())
    print(f"{len([item for sublist in list(products_to_download.values()) for item in sublist])} items")
    for (stage, products) in products_to_download.items():
        products_batches = list(chunks(products, DOWNLOADS_PER_SESSION_LIMIT))
        for batch in products_batches:
            for product in batch:
                print(f"Downloading {product['name']}...")
                download_product(product["name"])
                if check_for_sign_out():
                    print(f"Signed out! Retrying download...")
                    log_in()
                    download_product(product["name"])
            move_products_to_download_folder(batch, stage)


# Initiate the browser
BROWSER = new_browser(download_path=DOWNLOAD_PATH, headless=False)
BROWSER.get(URL)
log_in()
run_download(F1)
run_download(F2)


# download_product("S3A_OL_1_EFR____20160529T153001_20160529T153301_20180213T174834_0179_004_339_2340_LR2_R_NT_002.SEN3")
# move_to_download_folder(file_name="test.txt", sub_directory="stage_1")

# move_products_to_download_folder([list(STAGES.values())[0][0],list(STAGES.values())[0][1]], "stage_1")
