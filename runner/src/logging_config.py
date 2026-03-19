import logging
from logging.handlers import TimedRotatingFileHandler
import gzip
import os
import shutil
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("runner/.env", verbose=True)
LOG_PATH = os.environ['LOG_PATH']

class CustomTimedRotatingFileHandler(TimedRotatingFileHandler):
    def doRollover(self):
        """
        Override the doRollover method to check if the file is empty before rotating and compressing.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        if os.path.getsize(self.baseFilename) > 0:
            # Rotate the file if it is not empty
            now = datetime.now().strftime("%Y%m%d")
            new_log_name = f"{self.baseFilename}.{now}"
            os.rename(self.baseFilename, new_log_name)
            self.compress_old_log(new_log_name)

        if not self.delay:
            self.stream = self._open()

    def compress_old_log(self, source):
        """
        Compress the old log file.
        """
        with open(source, 'rb') as f_in:
            with gzip.open(source + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(source)


# Shut up kafka
logger = logging.getLogger('kafka')
logger.setLevel(logging.WARNING)

# Create a custom logger
logger = logging.getLogger('main_logger')
logger.setLevel(logging.DEBUG)  # Set the base level to debug for the console output

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Log all debug messages to the console

file_handler = CustomTimedRotatingFileHandler(f'{LOG_PATH}/app.error.log',
                                              when='midnight', interval=1, backupCount=30)
file_handler.setLevel(logging.ERROR)  # Log only error messages to the file

# Create formatters and add them to the handlers
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
