# twitter_reporter.py
from reporter_interface import SocialMediaReporter
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
import time

class TwitterReporter(SocialMediaReporter):
    def login(self):
        try:
            self.driver.get('https://twitter.com/login')
            self.wait.until(EC.presence_of_element_located((By.NAME, "text")))

            # Enter username
            username_field = self.driver.find_element(By.NAME, 'text')
            username_field.send_keys(self.username)
            username_field.send_keys(Keys.RETURN)
            time.sleep(2)

            # Enter password
            self.wait.until(EC.presence_of_element_located((By.NAME, 'password')))
            password_field = self.driver.find_element(By.NAME, 'password')
            password_field.send_keys(self.password)
            password_field.send_keys(Keys.RETURN)
            print("Logged into Twitter successfully.")
        except TimeoutException:
            print("Error: Timeout while trying to log in to Twitter.")

    def report_post(self, post_url, fakeness_score):
        if fakeness_score >= self.threshold:
            try:
                self.driver.get(post_url)
                self.wait.until(EC.presence_of_element_located((By.XPATH, '//div[@aria-label="More"]')))
                more_button = self.driver.find_element(By.XPATH, '//div[@aria-label="More"]')
                more_button.click()

                self.wait.until(EC.presence_of_element_located((By.XPATH, '//span[text()="Report Tweet"]')))
                report_option = self.driver.find_element(By.XPATH, '//span[text()="Report Tweet"]')
                report_option.click()

                self.wait.until(EC.presence_of_element_located((By.XPATH, '//span[contains(text(), "It\'s misleading or harmful")]')))
                misleading_option = self.driver.find_element(By.XPATH, '//span[contains(text(), "It\'s misleading or harmful")]')
                misleading_option.click()

                self.wait.until(EC.presence_of_element_located((By.XPATH, '//span[text()="Submit"]')))
                submit_button = self.driver.find_element(By.XPATH, '//span[text()="Submit"]')
                submit_button.click()

                print(f"Post reported on Twitter: {post_url}")
            except TimeoutException:
                print(f"Error: Timeout while attempting to report the post on Twitter: {post_url}")