# instagram_reporter.py
from reporter_interface import SocialMediaReporter
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
import time

class InstagramReporter(SocialMediaReporter):
    def login(self):
        try:
            self.driver.get('https://www.instagram.com/accounts/login/')
            self.wait.until(EC.presence_of_element_located((By.NAME, "username")))

            # Enter username and password
            username_field = self.driver.find_element(By.NAME, 'username')
            username_field.send_keys(self.username)
            password_field = self.driver.find_element(By.NAME, 'password')
            password_field.send_keys(self.password)
            password_field.send_keys(Keys.RETURN)
            print("Logged into Instagram successfully.")
        except TimeoutException:
            print("Error: Timeout while trying to log in to Instagram.")

    def report_post(self, post_url, fakeness_score):
        if fakeness_score >= self.threshold:
            try:
                self.driver.get(post_url)
                self.wait.until(EC.presence_of_element_located((By.XPATH, '//button[@aria-label="Options"]')))
                more_button = self.driver.find_element(By.XPATH, '//button[@aria-label="Options"]')
                more_button.click()

                self.wait.until(EC.presence_of_element_located((By.XPATH, '//button[text()="Report"]')))
                report_option = self.driver.find_element(By.XPATH, '//button[text()="Report"]')
                report_option.click()

                self.wait.until(EC.presence_of_element_located((By.XPATH, '//button[text()="It\'s inappropriate"]')))
                inappropriate_option = self.driver.find_element(By.XPATH, '//button[text()="It\'s inappropriate"]')
                inappropriate_option.click()

                self.wait.until(EC.presence_of_element_located((By.XPATH, '//button[text()="False information"]')))
                false_info_option = self.driver.find_element(By.XPATH, '//button[text()="False information"]')
                false_info_option.click()

                print(f"Post reported on Instagram: {post_url}")
            except TimeoutException:
                print(f"Error: Timeout while attempting to report the post on Instagram: {post_url}")