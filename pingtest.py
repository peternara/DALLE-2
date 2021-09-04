


from selenium import webdriver
import requests
import time
import argparse


def relogin(driver: webdriver.Chrome, username_str: str, password_str: str) -> None:
    # Go to page
    driver.get("https://www.btwifi.com:8443/")
    time.sleep(3)

    # Accept cookies if needed
    try:
        cookie_window = driver.find_element_by_class_name("cookieUI__section")
        (_, accept_cookies) = cookie_window.find_elements_by_class_name("cookieUI__btn")
        accept_cookies.click()
    except:
        pass

    # Click on login button
    login = driver.find_element_by_id("bthub-loginForm-toggle")
    login.click()

    # Fill in username password
    username = driver.find_element_by_id("wifi-inputUsername")
    password = driver.find_element_by_id("wifi-password")
    username.send_keys(username_str)
    password.send_keys(password_str)

    # Submit
    submit = driver.find_element_by_id("wifi-submit")
    submit.click()


def am_I_online() -> bool:
    try:
        requests.get("https://google.com/")
        return True
    except:
        return False


def main(args: argparse.Namespace):
    driver = webdriver.Chrome(executable_path=args.driver_path)
    while True:
        online = am_I_online()
        if not online:
            relogin(driver, args.username, args.password)
        time.sleep(10)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver_path", default=r"C:\Users\kovid\OneDrive\Documents\GitHub\DALLE\chromedriver.exe")
    parser.add_argument("--username", default="koviduppal")
    parser.add_argument("--password", default="wc5zxme3")
    (args, _) = parser.parse_known_args()
    main(args)
