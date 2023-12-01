import sys
from src.logger import logging

def error_message_details(error, error_details: sys):
    """
    Constructs a detailed error message including the file name, line number, and error message.

    Args:
    error: The exception object.
    error_details: The sys module to access exception traceback.

    Returns:
    A formatted string with error details.
    """
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in the Python script [{file_name}], line number: [{line_number}], error message: [{str(error)}]"
    return error_message
    
class CustomException(Exception):
    """
    A custom exception class that formats and logs error details such as file name and line number.
    """
    def __init__(self, error, error_detail: sys):
        """
        Constructor for CustomException.

        Args:
        error: The exception object.
        error_detail: The sys module to access exception traceback.
        """
        self.error_message = error_message_details(error=error, error_details=error_detail)
        super().__init__(self.error_message)
        
    def __str__(self):
        """
        String representation of the custom exception.
        """
        return self.error_message
    
# if __name__ == '__main__':
#     try:
#         a = 1/0
#     except Exception as e:
#         logging.error('Divide by Zero')
#         raise CustomException(e, sys)