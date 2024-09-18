# facebook_reporter.py
from reporter_interface import SocialMediaReporter
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
import time

class FacebookReporter(SocialMediaReporter):
    def login(self):
        try:
            self.driver.get('https://www.facebook.com/login')
            self.wait.until(EC.presence_of_element_located((By.ID, "email")))

            # Enter username and password
            email_field = self.driver.find_element(By.ID, 'email')
            email_field.send_keys(self.username)
            password_field = self.driver.find_element(By.ID, 'pass')
            password_field.send_keys(self.password)
            password_field.send_keys(Keys.RETURN)
            print("Logged into Facebook successfully.")
        except TimeoutException:
            print("Error: Timeout while trying to log in to Facebook.")

    def report_post(self, post_url, fakeness_score):
        if fakeness_score >= self.threshold:
            try:
                self.driver.get(post_url)
                self.wait.until(EC.presence_of_element_located((By.XPATH, '//div[@aria-label="Actions for this post"]')))
                more_button = self.driver.find_element(By.XPATH, '//div[@aria-label="Actions for this post"]')
                more_button.click()

                self.wait.until(EC.presence_of_element_located((By.XPATH, '//span[text()="Find support or report post"]')))
                report_option = self.driver.find_element(By.XPATH, '//span[text()="Find support or report post"]')
                report_option.click()

                self.wait.until(EC.presence_of_element_located((By.XPATH, '//span[text()="False news"]')))
                false_news_option = self.driver.find_element(By.XPATH, '//span[text()="False news"]')
                false_news_option.click()

                self.wait.until(EC.presence_of_element_located((By.XPATH, '//button[text()="Submit"]')))
                submit_button = self.driver.find_element(By.XPATH, '//button[text()="Submit"]')
                submit_button.click()

                print(f"Post reported on Facebook: {post_url}")
            except TimeoutException:
                print(f"Error: Timeout while attempting to report the post on Facebook: {post_url}")