import requests
import time
from logging import getLogger

logger = getLogger(__name__)

class FileNetworkProcessor:
    def __init__(self):
        pass

    @staticmethod
    def download_from_url(url, output_path):
        """
        Downloads an audio file from a URL and saves it to the specified output path.

        :param url: URL of the audio file to download.
        :param output_path: File path to save the downloaded audio file.
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code.

            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logger.info(f"Audio file downloaded successfully: {output_path}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")


class TimeTrack(object):
    def __init__(self, tag):
        self.tag = tag
        self.start_time = time.time()
     
    def __enter__(self):
        return self.tag
 
    def __exit__(self, *args):
        end_time = time.time()
        t = round((end_time - self.start_time) * 1000, 2) # time in MS rounded with 2 digits after decimal
        logger.info(f"**** TIME **** {self.tag}: {t} ms")
