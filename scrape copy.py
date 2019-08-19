# Import Dependencies
from bs4 import BeautifulSoup
import pymongo
from splinter import Browser
import requests
import time
import pandas as pd

# Splinter browser
executable_path = {'executable_path': '/usr/local/bin/chromedriver'}
browser = Browser('chrome', **executable_path, headless = False)

# URL
url = "https://mars.nasa.gov/news/"
browser.visit(url)

# Get HTML and parse
html_code = browser.html
soup = BeautifulSoup(html_code, "html.parser")

# Grab Info from HTML
news_title = soup.find('div', class_="bottom_gradient").text
news_p = soup.find('div', class_="rollover_description_inner").text

# Get Featured Image URL
jpl_url = "https://www.jpl.nasa.gov/spaceimages/?search=&category=Mars"
browser.visit(jpl_url)

# Go to link
browser.click_link_by_partial_text('FULL IMAGE')
browser.click_link_by_partial_text('more info')

# Get HTML
image_html = browser.html

# Parse
soup = BeautifulSoup(image_html, "html.parser")

# Find thr path
image_path = soup.find('figure', class_='lede').a['href']
featured_image_url = "https://www.jpl.nasa.gov/" + image_path

# Get Mars Weather
marsweather_url = "https://twitter.com/marswxreport?lang=en"
browser.visit(marsweather_url)

weather_html = browser.html

soup = BeautifulSoup(weather_html, 'html.parser')

mars_weather = soup.find('p', class_="TweetTextSize TweetTextSize--normal js-tweet-text tweet-text").text

# Get Mars Facts
facts_url = "https://space-facts.com/mars/"
browser.visit(facts_url)

facts_html = browser.html

soup = BeautifulSoup(facts_html, 'html.parser')

# Get the table
table_data = soup.find('table', class_="tablepress tablepress-id-mars")
In [95]:
# Find rows
table_all = table_data.find_all('tr')

# Create lists for labels and values
labels = []
values = []

# Append the lists
for tr in table_all:
    td_elements = tr.find_all('td')
    labels.append(td_elements[0].text)
    values.append(td_elements[1].text)

# Create Data Frame
mars_facts_df = pd.DataFrame({
    "Label": labels,
    "Values": values
})

# Get the HTML code for the Data Frame
fact_table = mars_facts_df.to_html(header = False, index = False)
fact_table

# URL
usgs_url = "https://astrogeology.usgs.gov/search/results?q=hemisphere+enhanced&k1=target&v1=Mars"

browser.visit(usgs_url)

usgs_html = browser.html

soup = BeautifulSoup(usgs_html, "html.parser")

# Gets the class holding hemisphere picture
returns = soup.find('div', class_="collapsible results")
hemispheres = returns.find_all('a')

# Create lists
hemisphere_image_urls =[]

# Get URLS
for a in hemispheres:
    title = a.h3.text
    link = "https://astrogeology.usgs.gov" + a['href']
    
    browser.visit(link)
    time.sleep(5)
    
    image_page = browser.html
    results = BeautifulSoup(image_page, 'html.parser')
    img_link = results.find('div', class_='downloads').find('li').a['href']
    
    image_dict = {}
    image_dict['title'] = title
    image_dict['img_url'] = img_link
    
    hemisphere_image_urls.append(image_dict)
    
print(hemisphere_image_urls)