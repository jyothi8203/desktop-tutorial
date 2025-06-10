# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from FaceRDlib import FaceRecg
from datetime import datetime
from app import app_run


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def currTm():
    return datetime.now().strftime("%dd-%mon-%YY:%H:%M:%S")


if __name__ == '__main__':
    # print_hi('PyCharm')
    objFcR = FaceRecg()
    ret_val = "No Face"
    if objFcR.logger is None:
        objFcR.set_log('attendance.log')
    while True:
        print("starting from main",currTm())
        # if objFcR.logger is None:
        #     ret_val = objFcR.identify_face('',None)
        # else:
        #     ret_val = objFcR.identify_face('',objFcR.logger)
        ret_val = objFcR.identify_face('',objFcR.logger)
        if ret_val == "Unknown":
            print("breaking in main")
            break
        print("ending at main",currTm())
    if ret_val == "Unknown":
        #launching chat window for visitor help
        app_run()