import sys

def err_message_details(error, err_detail:sys):
  _,_, exc_tb= err_detail.exc_info()
  file_name = exc_tb.tb_frame.f_code.co_filename
  error_message = "Error occured in python script: [{0}]\nError occurred at line: [{1}]\nError message: [{2}]".format(
    file_name,exc_tb.tb_lineno,str(error)
  )
  return error_message

class CustomException(Exception):
  def __init__(self, error_message, error_details:sys):
    super().__init__(error_message)
    self.error_message=err_message_details(error_message,err_detail=error_details)
  
  def __str__(self):
    return self.error_message