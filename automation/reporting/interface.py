# reporter_interface.py
from abc import ABC, abstractmethod
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

class SocialMediaReporter(ABC):
    def _init_(self, username, password, threshold, webdriver_path):
        self.username = username
        self.password = password
        self.threshold = threshold
        self.driver = webdriver.Chrome(service=Service(webdriver_path))
        self.wait = WebDriverWait(self.driver, 10)

    @abstractmethod
    def login(self):
        pass

    @abstractmethod
    def report_post(self, post_url, fakeness_score):
        pass

    def close(self):
        self.driver.quit()