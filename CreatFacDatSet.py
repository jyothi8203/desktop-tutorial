import cv2 as cv
import os
# import pyautogui
import logging
from FaceRDlib import FaceRecg


class CreateFace:

    def __init__(self):
        self.data_fldr = None
        self.emp_id='100'
        self.emp_name='AiFA'
        self.wdth , self.hght = self.set_scrn_sz()
        self.bbW=200
        self.bbH=200
        self.data_fldr = 'datasetpy'
        self.logger=None
        # self.set_log('attendance.log')

    def set_log(self,log_file_nm):
        # logging.basicConfig(filename=log_file_nm, format='%(asctime)s : %(levelname)s : %(name)s : %(message)s',level=logging.WARNING)
        # define file handler and set formatter
        file_handler = logging.FileHandler(log_file_nm)
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(funcName)s :  %(message)s')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.WARNING)

        i_handler = logging.StreamHandler()
        wi_formatter = logging.Formatter('%(levelname)s : %(name)s : %(funcName)s : %(message)s')
        i_handler.setFormatter(wi_formatter)
        i_handler.setLevel(logging.INFO)

        # w_handler = logging.FileHandler(log_file_nm)
        # w_handler.setLevel(logging.ERROR)
        # w_handler.setFormatter(wi_formatter)
        # self.logger.addHandler(w_handler)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(file_handler)#warning
        self.logger.addHandler(i_handler)#error

    def create_faces(self,fDetect,fldrNm):
        file_name=''
        count=0
        camera = cv.VideoCapture(0)
        camera.set(3, self.wdth)  # set video width
        camera.set(4, self.hght)  # set video height
        while(True):
            ret, frame = camera.read()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = fDetect.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                if w >= self.bbW and h >= self.bbH:
                    cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                    count += 1    # Save the captured image into the datasets folder
                    file_name = fldrNm+"/User." + str(self.emp_id) + '.' +str(count) + ".jpg"
                    # print(os.path.abspath(file_name))
                    img_data = gray[y:y + h, x:x + w]
                    reszd_img = cv.resize(img_data,(self.bbW,self.bbH))
                    val = cv.imwrite(file_name,reszd_img)
                    # if val:
                    #     print("Successfully wrote",file_name)
                    cv.imshow('frame',frame)
            #if esc key pressed or no more video input
            if cv.waitKey(1) & 0xFF == 27 or count == 100:#need to change to 100 or 500
                break
        # When everything done, release the capture
        camera.release()
        cv.destroyAllWindows()
        img = cv.imread(file_name, cv.IMREAD_UNCHANGED)
        # cv.imshow('Image',img)

    def set_scrn_sz(self):
        wd = 1280
        ht = 780
        # screen_height, screen_width = pyautogui.size()
        screen_height, screen_width = 1280,780

        # print(screen_width, screen_height)
        if screen_width != 0 and screen_height != 0:
            wd = screen_width
            ht = screen_height
        return wd,ht

    def csv_not_duplicate(self):
        fp = open('emp_details.csv', 'a+')
        # check if already exists in emp_details.csv
        fp.seek(0)
        str_emp_data = fp.readlines()
        # print(str_emp_data)
        self.logger.warning(str_emp_data)
        strEmpid = str(self.emp_id)
        found = False
        for echLn in str_emp_data:
            if echLn.find(strEmpid) != -1 and echLn.find(self.emp_name) != -1:
                found = True
        if found is False:
            strTxt = f'{self.emp_id}, {self.emp_name}\n'
            # print(strTxt)
            self.logger.warning(strTxt)
            fp.writelines(strTxt)
            # print("Successfully created {0} :{1} dataset".format(self.emp_name, self.emp_id))
            self.logger.warning("Successfully created {0} :{1} dataset".format(self.emp_name, self.emp_id))
        fp.close()

    def create_face_dataset(self,eid,ename):
        try:
            if self.logger is None:
                self.set_log('attendance.log')
            self.bbW = 200
            self.bbH = 200
            self.data_fldr='datasetpy'
            # self.wdth,self.hght = self.set_scrn_sz()
            self.emp_id = eid
            self.emp_name = ename
            fdetector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
            self.logger.warning('Newly creating face dataset')
            # print(self.data_fldr,self.emp_id,type(self.data_fldr),type(self.emp_id))
            self.logger.warning('{0}-{1}-{2}-{3}'.format(self.data_fldr,self.emp_id,type(self.data_fldr),type(self.emp_id)))
            fldrNm = os.path.join(self.data_fldr, self.emp_id)
                    # print(fldrNm)
            if not os.path.exists(fldrNm):
                # logging.info(fldrNm)
                self.logger.warning('{0}'.format(fldrNm))
                os.mkdir(fldrNm)
            self.create_faces(fdetector,fldrNm)
            self.csv_not_duplicate()
            # print("closing file")
            # print("closed file ptr")
            self.logger.warning('{0}'.format('closing file'))
            self.logger.warning('{0}'.format('closed file ptr'))
            objFcR = FaceRecg()
            objFcR.identify_face(self.emp_id,self.logger)
        except Exception as e:
            self.logger.error(e)
        # import Button