import logging
import os
from datetime import datetime

# Get the current date and time
current_datetime = datetime.now()

# Create directory names based on the current year, month, and day
year = current_datetime.strftime('%Y')
month = current_datetime.strftime('%m')
day = current_datetime.strftime('%d')

# Define the directory path for storing log files (logs/year/month/day)
logs_directory = os.path.join(os.getcwd(), 'logs', year, month, day)
os.makedirs(logs_directory, exist_ok=True)  # Create the directories if they don't exist

# Create a log file name with a timestamp
LOG_FILE = f"{current_datetime.strftime('%Y_%m_%d_%H_%M_%S')}.log"

# Define the full file path for the log file
LOG_FILE_PATH = os.path.join(logs_directory, LOG_FILE)

# Configure logging to write to the file, with specific format and level
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

# if __name__ == '__main__':
#     # Example log message to indicate the start of logging
#     logging.info('Started Logging...')
