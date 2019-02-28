import pandas as pd
import os
import time
from selenium import webdriver
from . import dbScrapeVars

class GC:
    def __init__(self, webdriver_path=None, download_folder=None ):
        print('Thank you for using DBscrape for google correlate! please use our \
              \n read me to see a simple tutorial on how to use it. GCpy opens a\
              \n web browser using both Selenium and chrome webdriver libraries. \n\n')

        #Browser wait times
        #soptions = webdriver.ChromeOptions()
        #options.add_argument('headless')
        self.wait_time_get = 2
        self.wait_time_csv = 10
        self.download_folder= download_folder
        try:
            if webdriver_path is None:
                self.driver = webdriver.Chrome(chrome_options=options)
            else:
                self.driver = webdriver.Chrome(executable_path=webdriver_path) #chrome_options=options
                print('Succesfully initialized  web browser.')
        except Exception as e:
            print('Oops! We were unable to initialize chrome webdriver. Please make sure\
                    \n selenium is installed and phantomJS is in your PATH or that you provide a\
                    \n path using the var exec_path in GCpy() \n\n')
            print(e)

    def login(self,user=None, password=None, verbose=False):

        if verbose:
            print('Logging into Gmail. WARNING! GCpy original source is just supposed \
                  \n to use your credentials with the purpose to accessing and managing Google Correlate.\
                  \n Any other activity is not part os the scope in GCpy and should be inmediately.\n\n')

        try:
            self.driver.get(dbScrapeVars.GMAIL_LOGIN_URL)
            time.sleep(self.wait_time_get)
            self.driver.find_element_by_name("identifier").send_keys(user)
            self.driver.find_element_by_id("identifierNext").click()
            time.sleep(self.wait_time_get)
            self.driver.find_element_by_name("password").send_keys(password)
            self.driver.find_element_by_id("passwordNext").click()
            time.sleep(self.wait_time_get)
            print('Logged onto Gmail account.')

            if verbose:
                print('Successfully logged into Gmail.\n\n')
        except Exception as e:
            print('GCpy was unable to log in. Please make sure credentials are correct.\n \
                  Other possible reasons include: \n Gmail not recognizing your login location \n \
                  browser wait time per actions are too brief. \n you lost connection?\n\n')
            print(e)

    def correlate_term(self, search_term=None, country=None, verbose=False, sdate=None, edate=None):

        if country in dbScrapeVars.GC_AVAILABLE:
            if verbose: print('Searching for search terms correlated to the word \n \
                              {0} in {1}\n\n'.format(search_term, dbScrapeVars.CODE_DICT[country]))
            country_index = dbScrapeVars.GC_AVAILABLE.index(country)
            try:
                #Directly download and rename a file from google correlate
                if sdate and edate:
                    self.driver.get("https://www.google.com/trends/correlate/csv?e={0}&t=weekly&p={1}&tr={2}_{3}\n\n".format(search_term, country.lower(), sdate, edate))
                else:
                    self.driver.get("https://www.google.com/trends/correlate/csv?e={0}&t=weekly&p={1}\n\n".format(search_term, country.lower()))
                time.sleep(self.wait_time_csv)
                if self.download_folder is not None:
                    os.rename(self.download_folder+'/correlate-{0}.csv'.format(search_term),\
                              self.download_folder+'/{0}_{1}.csv'.format(search_term, country))
                else:
                    print("Warning! No default download folder specified. gcpy can't rename correlate file. \n \
                          If you're downloading more than one term it might become lost")
                print('Successfully downloaded data for {0} in {1}'.format(search_term, dbScrapeVars.CODE_DICT[country]))
            except Exception as e:
                print('An error ocurred while downloading data for {0} in {1}'.format(search_term,dbScrapeVars.CODE_DICT[country]))
                if verbose: print(e)
        else:
            print('{0} is not available in Google Correlate. Please try another country'.format(dbScrapeVars.CODE_DICT[country]))

    def correlate_from_csv(self, path_to_csv=None, timeseries_name=None, country=None, verbose=False):
        if country in dbScrapeVars.GC_AVAILABLE:

            if verbose: print('Searching for terms correlated to {0} \n \
                               in {1}\n\n'.format(timeseries_name,dbScrapeVars.CODE_DICT[country]))
            country_index = dbScrapeVars.GC_AVAILABLE.index(country)
            try:
                # Load csv file
                with open(path_to_csv) as f:
                    csv_text = f.read() + '\n' # add trailing new line character

                #Scraping
                self.driver.get('https://www.google.com/trends/correlate/search?e=influenza&t=weekly&p={0}'.format(country.lower()))
                time.sleep(self.wait_time_get)
                self.driver.find_element_by_link_text("Enter your own data").click()
                time.sleep(self.wait_time_get)
                self.driver.get('https://www.google.com/trends/correlate/edit?e=&t=weekly')
                time.sleep(self.wait_time_get)
                self.driver.find_element_by_id('csv-weekly').send_keys(path_to_csv)
                self.driver.find_element_by_id('name-weekly').send_keys(timeseries_name)
                time.sleep(self.wait_time_get)
                self.driver.execute_script("document.getElementById('csv').value={0}".format(repr(csv_text)))
                self.driver.execute_script("document.getElementById('place-weekly').options[{0}].setAttribute(\"Selected\",\"\")".format(country_index))
                #driver.find_element_by_id("document.getElementById('csv').value={0}".format(csv_text))
                time.sleep(self.wait_time_get)
                self.driver.find_element_by_id('submit-weekly').click()
                time.sleep(self.wait_time_get)
                self.driver.find_element_by_link_text("CSV").click()
                time.sleep(self.wait_time_csv)

                if self.download_folder is not None:
                    os.rename(self.download_folder+'/correlate-{0}.csv'.format(timeseries_name),\
                              self.download_folder+'/{0}_{1}.csv'.format(country, timeseries_name))

                else:
                    print("Warning! No default download folder specified. gcpy can't rename correlate file. \n \
                          If you're downloading more than one term it might become lost")
                print('Successfully downloaded data for {0} in {1}'.format(timeseries_name, dbScrapeVars.CODE_DICT[country]))
            except Exception as e:
                print('An error ocurred while downloading data for {0} in {1}'.format(timeseries_name,dbScrapeVars.CODE_DICT[country]))
                if verbose: print(e)
        else:
            print('{0} is not available in Google Correlate. Please try another country'.format(dbScrapeVars.CODE_DICT[country]))


    def close_browser(self):
        self.driver.quit()
    def init_browser(self):
        try:
            if exec_path is None:
                self.driver = webdriver.Chrome(chrome_options=options)
            else:
                self.driver = webdriver.Chrome(executable_path=exec_path, chrome_options=options)
                print('Succesfully initialized hidden web browser.')
        except Exception as e:
            print('Oops! We were unable to initialize chrome webdriver. Please make sure\
                    \n selenium is installed and chromedriver is in your PATH or that you provide a\
                    \n path using the var webdriver_path()')


class INMET:

    def __init__(self, webdriver_path=None, download_folder=None ):
            print('Thank you for using DBscrape for INMET! please use our \
                  \n read me to see a simple tutorial on how to use it. DBscrapepy opens a\
                  \n web browser using both Selenium and chrome webdriver libraries. \n\n')

class SINAVE:

    def __init__(self, webdriver_path=None, download_folder=None ):
        print('Thank you for using DBscrape for SINAVE! please use our \
              \n read me to see a simple tutorial on how to use it. DBscrape opens a\
              \n web browser using both Selenium and chrome webdriver libraries. \n\n')

        #Browser wait times
        #soptions = webdriver.ChromeOptions()
        #options.add_argument('headless')
        self.wait_time_get = 2
        self.wait_time_csv = 10
        self.download_folder= download_folder
        try:
            if webdriver_path is None:
                self.driver = webdriver.Chrome(chrome_options=options)
            else:
                self.driver = webdriver.Chrome(executable_path=webdriver_path) #chrome_options=options
                print('Succesfully initialized  web browser.')
        except Exception as e:
            print('Oops! We were unable to initialize chrome webdriver. Please make sure\
                    \n selenium is installed and phantomJS is in your PATH or that you provide a\
                    \n path using the var exec_path in GCpy() \n\n')
            print(e)

    def login(self,user=None, password=None, verbose=False):

        if verbose:
            print("Logging into SINAVE website. WARNING!  original source is just supposed \
                  \n to use your credentials with the purpose to accessing and managing SINAVE's flu activity\
                  \n Any other activity is not part of the scope and should be inmediately.\n\n")

        try:
            # Initialize
            self.driver.get(dbScrapeVars.SINAVE_LOGIN_URL)
            self.driver.find_element_by_id("myPopup").click() # Remove legend
            #log in
            self.driver.find_element_by_id("usuario").send_keys(user)
            self.driver.find_element_by_id("clave").send_keys(password)
            self.driver.find_element_by_id("entrar").click()

            if verbose:
                print('Successfully logged into SINAVE.\n\n')
        except Exception as e:
            print('Unable to log in. Please make sure credentials are correct.\n \
                  Other possible reasons include:  \n \
                  browser wait time per actions are too brief. \n you lost connection?\n\n')
            print(e)

    def query_time_window(self, start_date=None, end_date=None,to_csv=True, verbose=False):
        '''
            -Dates should be in YYYY-MM-DD format
        '''

        if verbose: print('Extracing influenza activity from {0} to {1}')
        print(start_dates[i], end_dates[i])

        try:
            self.driver.get(dbScrapeVars.SINAVE_REPORT_URL)

            #Input initial date
            driver.find_element_by_id("fInicial").send_keys(start_date[8:10])
            time.sleep(.1)
            driver.find_element_by_id("fInicial").send_keys(start_date[5:7])
            time.sleep(.1)
            driver.find_element_by_id("fInicial").send_keys(start_date[0:4])

            #Input final date
            driver.find_element_by_id("fFinal").send_keys(end_date[8:10])
            time.sleep(.1)
            driver.find_element_by_id("fFinal").send_keys(end_date[5:7])
            time.sleep(.1)
            driver.find_element_by_id("fFinal").send_keys(end_date[0:4])

            driver.find_element_by_link_text("ATENCIONES POR ENTIDAD FEDERATIVA").click()
            time.sleep(2)

            s=driver.find_element_by_id("reporte")
            s.get_attribute('innerHTML')
            df= pd.read_html(s.get_attribute('innerHTML'), index_col=0, skiprows=2)
            df=df[0]
            df=df.rename(index=str, columns={1:'ALTA', 2:'DEFUNCIONES', 3:'GRAVE', 4:'NO_GRAVE', 5:'HOSP', 6:'TOTAL'})
            df=df.transpose()
            ind = pd.MultiIndex(levels=[['ALTA', 'DEFUNCIONES', 'GRAVE', 'NO_GRAVE', 'HOSP', 'TOTAL'], [pd.Timestamp(STANDARD_DATES[i])]],
                       labels=[[0,1,2,3,4,5], [0,0,0,0,0,0]],
                       names=['TYPE', 'DATE'])
            df = df.set_index(ind)
            if to_csv:
                df.to_csv('SINAVE_influenza_query_{0}_{1}.csv'.format(start_date, end_date))
                return
            else:
                return df
        except Exception as t:
            return ('Could not retrieve SINAVE report')

class GT:
    def __init__(self, webdriver_path=None, download_folder=None ):
        print('Thank you for using DBscrape for Google Trends! please use our \
              \n read me to see a simple tutorial on how to use it.')

        #Browser wait times
        #soptions = webdriver.ChromeOptions()
        #options.add_argument('headless')
        self.wait_time_get = 2
        self.wait_time_csv = 10
        self.download_folder= download_folder
        try:
            if webdriver_path is None:
                self.driver = webdriver.Chrome(chrome_options=options)
            else:
                self.driver = webdriver.Chrome(executable_path=webdriver_path) #chrome_options=options
                print('Succesfully initialized  web browser.')
        except Exception as e:
            print('Oops! We were unable to initialize chrome webdriver. Please make sure\
                    \n selenium is installed and phantomJS is in your PATH or that you provide a\
                    \n path using the var exec_path in GCpy() \n\n')
            print(e)

    def login(self,user=None, password=None, verbose=False):

        if verbose:
            print('Logging into Gmail. WARNING! GCpy original source is just supposed \
                  \n to use your credentials with the purpose to accessing and managing Google Correlate.\
                  \n Any other activity is not part os the scope in GCpy and should be inmediately.\n\n')

        try:
            self.driver.get(dbScrapeVars.GMAIL_LOGIN_URL)
            time.sleep(self.wait_time_get)
            self.driver.find_element_by_name("identifier").send_keys(user)
            self.driver.find_element_by_id("identifierNext").click()
            time.sleep(self.wait_time_get)
            self.driver.find_element_by_name("password").send_keys(password)
            self.driver.find_element_by_id("passwordNext").click()
            time.sleep(self.wait_time_get)
            print('Logged onto Gmail account.')

            if verbose:
                print('Successfully logged into Gmail.\n\n')
        except Exception as e:
            print('GCpy was unable to log in. Please make sure credentials are correct.\n \
                  Other possible reasons include: \n Gmail not recognizing your login location \n \
                  browser wait time per actions are too brief. \n you lost connection?\n\n')
            print(e)

    def find_relatedqueries(self, search_term=None, country=None, verbose=False, sdate=None, edate=None):


        if verbose: print('Searching for related queries correlated to the word \n \
                          {0} in {1}\n\n'.format(search_term, dbScrapeVars.CODE_DICT[country]))
        try:
            #Directly download and rename a file from google correlate
            if sdate and edate:
                self.driver.get("https://trends.google.com/trends/explore?date={2}%20{3}&geo={1}&q={0}\n\n".format(search_term, country.upper(), sdate, edate))
                time.sleep(10)
            else:
                self.driver.get("https://trends.google.com/trends/explore?geo={1}&q={0}\n\n".format(search_term, country.upper(), sdate, edate))
            self.driver.find_element_by_xpath("/html[1]/body[1]/div[2]/div[2]/div[1]/md-content[1]/div[1]/div[1]/div[4]/trends-widget[1]/ng-include[1]/widget[1]/div[1]/div[1]/div[1]/widget-actions[1]/div[1]/button[1]/i[1]").click()
            time.sleep(self.wait_time_csv)
            if self.download_folder is not None:
                os.rename(self.download_folder+'/relatedQueries.csv'.format(search_term),\
                          self.download_folder+'/{0}_{1}_relatedQueries.csv'.format(search_term, country))
            else:
                print("Warning! No default download folder specified. gtpy can't rename correlate file. \n \
                      If you're downloading more than one term it might become lost")
            print('Successfully downloaded data for {0} in {1}'.format(search_term, dbScrapeVars.CODE_DICT[country]))
        except Exception as e:
            print('An error ocurred while downloading data for {0} in {1}'.format(search_term,dbScrapeVars.CODE_DICT[country]))
            if verbose: print(e)


    def close_browser(self):
        self.driver.quit()
    def init_browser(self):
        try:
            if exec_path is None:
                self.driver = webdriver.Chrome(chrome_options=options)
            else:
                self.driver = webdriver.Chrome(executable_path=exec_path, chrome_options=options)
                print('Succesfully initialized hidden web browser.')
        except Exception as e:
            print('Oops! We were unable to initialize chrome webdriver. Please make sure\
                    \n selenium is installed and chromedriver is in your PATH or that you provide a\
                    \n path using the var webdriver_path()')
