import sys

class RAGException(Exception):
    def __init__(self, error_message, error_details:sys):
        self.error_message = error_message
        _,_,traceback = error_details.exc_info()
        
        self.lineno = traceback.tb_lineno
        self.file_name = traceback.tb_frame.f_code.co_filename
        
    def __str__(self):
        return f"Error occured in the following script: {self.file_name} at line: {self.lineno}. Error message: {str(self.error_message)}"