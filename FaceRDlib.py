import tkinter.messagebox
import traceback
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import pandas as pd
from datetime import datetime
# import pyautogui
from tkinter import messagebox
import logging
import win32com.client

class FaceRecg:
    def __init__(self):
        self.emp_names_dict={}
        self.noFcFlg = 0
        self.fcFlg = 0
        self.img_fldr = 'datasetpy'
        self.name_list=[]
        self.dir_lst_img_fldr = []
        self.wdth, self.hght = 640,480#self.set_scrn_sz()
        self.known_encodings = []
        self.known_names = []
        self.min_wdth = 200
        self.min_hght = 200
        self.logger = None
        self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
        self.match_dict={}
        # self.set_log('attendance.log')

    def set_log(self, log_file_nm):
        # logging.basicConfig(filename=log_file_nm, format='%(asctime)s : %(levelname)s : %(name)s : %(message)s',level=logging.WARNING)
        # define file handler and set formatter
        file_handler = logging.FileHandler(log_file_nm)
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(funcName)s %(message)s')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.WARNING)

        i_handler = logging.StreamHandler()
        wi_formatter = logging.Formatter('%(levelname)s : %(name)s : %(funcName)s %(message)s')
        i_handler.setFormatter(wi_formatter)
        i_handler.setLevel(logging.INFO)

        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(file_handler)  # warning

    def load_names_csv(self):
        try:
            csv_empids = pd.read_csv('emp_details.csv',skiprows=0)
            # print(csv_empids,csv_empids.values)
            self.emp_names_dict=dict((csv_empids.values))
            # self.emp_names_dict={}
            self.name_list = self.emp_names_dict.values()
            # print(emp_names_dict)
        except Exception as e:
            self.logger.error(e)

# loop over the image paths
    def train_faces(self,dir_lst_I_fldr):
        try:
            train_count=0
            for emp_id in dir_lst_I_fldr:
                emp_id_path = os.path.join(self.img_fldr, emp_id)
                # print(emp_id_path)
                self.logger.warning('{0}'.format(emp_id_path))
                emp_id_imgs = [os.path.join(emp_id_path, fl_nm) for fl_nm in os.listdir(emp_id_path)]  # (img_fldr,emp_id)]
                name = self.emp_names_dict[int(emp_id)]
                for imagePath in emp_id_imgs:
                    # print(imagePath)
                    if imagePath.endswith('.jpg'):
                        image = cv2.imread(imagePath)
                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # Use Face_recognition to locate faces # compute the facial embedding for the face
                        boxes = face_recognition.face_locations(rgb, model='hog')
                        encodings = face_recognition.face_encodings(rgb, boxes)
                        # loop over the encodings
                        # print("training image",train_count,imagePath)
                        # print(encodings,len(encodings))
                        for encoding in encodings:
                            self.known_encodings.append(encoding)
                            self.known_names.append(name)
                            # print(name)
                        train_count+=1
            # save emcodings along with their names in dictionary data
            data = {"encodings": self.known_encodings, "names": self.known_names}
            f = open("face_enc", "wb")
            f.write(pickle.dumps(data))
            f.close()
            # print("Finished training ", train_count, "faces")
            self.logger.warning('{0}{1}{2}'.format('Finished training',train_count,'faces'))
        except Exception as e:
            self.logger.error(e)

    def readAttendance(self,nm):
        try:
            strCurrDy = str(self.currDay())
            self.logger.warning('object type{0}{1}'.format(self.currDay(),strCurrDy))
            att_fldr = 'Attendance'
            att_signin = 'Attendance/signAttend'+strCurrDy+'.csv'
            att_log = 'Attendance/logAttendance.csv'
            ret_val = False
            if os.path.exists(att_fldr) is False:
                os.mkdir('Attendance')
                self.logger.warning('creating Attendance folder')
                ret_val = False
            else:
                if os.path.exists(att_signin) is False:
                    self.logger.warning('signinAttendance file not exists')
                    ret_val = False
                else:
                    fp = open(att_signin,'r')
                    lst_sign = fp.readlines()
                    fp.close()
                    self.logger.warning('{}'.format(lst_sign))
                    ret_lst = [nm for ech_log in lst_sign if ech_log.find(nm)!=-1 and ech_log.find(strCurrDy)!=-1]
                    self.logger.warning('{}'.format(ret_lst))
                    if len(ret_lst) >= 1:
                        ret_val = True
        except Exception as e:
            self.logger.error(e)
        return ret_val

    def writeAttend(self,fp,name):
        try:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            strTxt = f'{name}, {time}, {date}\n'
            # print(strTxt)
            fp.writelines(strTxt)
        except Exception as e:
            self.logger.error(e)

    def markLog(self,name):
        try:
            f = open('logAttendance.csv', 'a+')
            if name in self.name_list:
                self.writeAttend(f,name)
                self.logger.warning('For {} Marking log attendance for the day {}'.format(name,self.currTm()))
            f.close()
        except Exception as e:
            self.logger.error(e)

    def markSign(self,name):
        try:
            strCurrDy = self.currDay()
            att_fldr = 'Attendance'
            att_signin = 'Attendance/signAttend'+strCurrDy+'.csv'
            f = open(att_signin, 'a+')
            if name in self.name_list:
                self.writeAttend(f,name)
                self.logger.warning('For {} Marking signin attendance for the day {}'.format(name,self.currTm()))
            f.close()
        except Exception as e:
            self.logger.error(e)

    def markAttendance(self,name,msg):
        try:
            strTxt = msg
            val = self.readAttendance(name)
            if val is  True:
                strLogTxt = '{0}{1}'.format(msg,"Logging")
                self.markLog(name)
                self.speak_voice(strLogTxt)
            else:
                strSignTxt = '{0}{1}'.format(msg,"Marking signin")
                self.markSign(name)
                self.speak_voice(strSignTxt)

        except Exception as e:
            self.logger.error(e)

    def find_match_nms(self,mtchs,dstns,trn_dta):
        try:
            names=[]
            fnl_nm=''
            # Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(mtchs) if b]
            dsts_lst = [dstns[idx] for (idx) in matchedIdxs]
            names_lst = []
            for (idx) in matchedIdxs:
                if dstns[idx] < 0.4:  # for 0.6 giving 3 names
                # if dstns[idx] < 0.32:
                #     print(dstns[idx], trn_dta['names'][idx])
                    self.logger.warning('{0},{1}'.format(dstns[idx],trn_dta['names'][idx]))
                    names_lst.append(trn_dta['names'][idx])
            # print("matched distncs", dsts_lst)
            self.logger.warning("names lst in match names{}".format(names_lst))
            # loop over the matched indexes and maintain a count for
            count_dict = {}
            max_count = -1
            if len(names_lst) > 0:
                for nm in set(names_lst):
                    nm_freq = names_lst.count(nm)
                    if max_count < nm_freq:
                        max_count = nm_freq
                        name_dst = nm
                    count_dict[nm] = names_lst.count(nm)
                # print(count_dict)
                # print(name_dst)
                name = name_dst
        # update the list of names
                names.append(name)
                self.logger.warning("name found{}".format(names))
            self.logger.warning(self.match_dict)
            if len(names_lst) > 0:
                for nm in set(names_lst):
                    nm_freq = names_lst.count(nm)
                    if max_count < nm_freq:
                        max_count = nm_freq
                    if nm in self.match_dict.keys():
                        self.match_dict[nm] = self.match_dict[nm] + names_lst.count(nm)
                    else:
                        self.match_dict[nm] = names_lst.count(nm)
            # if len(self.match_dict) >=1:
            #     names=[]
            #     self.logger.warning("After cumulative {}".format(self.match_dict))
            #     sorted_dict = sorted(self.match_dict.items(), key=lambda x: x[1], reverse=True)
            #     self.logger.warning("After sorting {}".format(sorted_dict))
            #     names_dict = dict(sorted_dict)
            #     self.logger.warning("Dict of sorted Dict {}".format(names_dict))
            #     print(names_dict)
            #     final_name = list(names_dict.keys())
            #     self.logger.warning("Final name list {}".format(final_name))
            #     names.append(final_name[0])
            #     self.logger.warning("matched name  {}".format(names))
        except Exception as e:
            self.logger.error(e)
        return names #set(names_lst)

    def currTm(self):
        return datetime.now().strftime("%d-%m-%Y:%H:%M:%S")

    def currDay(self):
        return datetime.now().strftime("%d-%B-%Y")

    def final_name(self):
        names = []
        if len(self.match_dict) >=1:
            self.logger.warning("After cumulative {}".format(self.match_dict))
            sorted_dict = sorted(self.match_dict.items(), key=lambda x: x[1], reverse=True)
            self.logger.warning("After sorting {}".format(sorted_dict))
            names_dict = dict(sorted_dict)
            self.logger.warning("Dict of sorted Dict {}".format(names_dict))
            print(names_dict)
            final_name = list(names_dict.keys())
            self.logger.warning("Final name list {}".format(final_name))
            names.append(final_name[0])
            self.logger.warning("matched name  {}".format(names))
        else:
            print(self.noFcFlg,self.fcFlg)
            if self.fcFlg>=1:
                names.append("Unknown")
            elif self.noFcFlg>=1:
                names.append("No Face")
        return names

    def auto_face_rec_from_camera(self,fDet,data):
        try:
            # print("Streaming started","auto_face_rec_from_camera")
            # print("Starting video capture",self.currTm())
            self.logger.warning("{0}{1}".format("Starting video capture",self.currTm()))
            video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            self.logger.warning("{0}{1}".format("Started video capture", self.currTm()))
            # video_capture.set(3, self.wdth)  # set video width
            # video_capture.set(4, self.hght)  # set video height
            video_capture.set(3, 1280)
            video_capture.set(4, 720)
            # loop over frames from the video file stream
            recg_nm = "No Face"
            strt_time=now = datetime.now()
            print("Started video capture", self.currTm())
            while True:
                # grab the frame from the threaded video stream
                # print(date,curr_time)
                # print("Before video capture read", self.currTm())
                ret, frame = video_capture.read()
                # print("After video capture read", self.currTm())
                # frame = cv2.resize(frame,(1920,1080))
                print(frame.shape)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = fDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_wdth, self.min_hght),flags=cv2.CASCADE_SCALE_IMAGE)
                # print(faces)
                # convert the input frame from BGR to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # the facial embeddings for face in input
                encodings = face_recognition.face_encodings(rgb)
                print("No of encodings",len(encodings))
                if len(encodings) == 0:
                    self.noFcFlg = self.noFcFlg + 1
                mtch_nms = []
                # loop over the facial embeddings incase
                # we have multiple embeddings for multiple fcaes
                for encoding in encodings:
                    # Compare encodings with encodings in data["encodings"]
                    # Matches contain array with boolean values and True for the embeddings it matches closely
                    # and False for rest
                    matches = face_recognition.compare_faces(data["encodings"],encoding)
                    distncs = face_recognition.face_distance(data['encodings'],encoding)
                    # check to see if we have found a match
                    if True in matches:
                        print(("True in matches"))
                        self.fcFlg = self.fcFlg + 1
                        mtch_nms = self.find_match_nms(matches,distncs,data)
                        print("inside face recog from camera",mtch_nms)
                        # if len(mtch_nms)==0:
                        #     recg_nm = 'Unknown'
                        for ((x, y, w, h), name) in zip(faces, mtch_nms):
                            print("inside zip faces for loop",name)
                            # rescale the face coordinates
                            # draw the predicted face name on the image
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.75, (0, 255, 0), 2)
                            if name != "Unknown":
                                recg_nm = name
                            # print(name)
                            self.logger.warning("{0}".format(recg_nm))
                    else:
                        print("false in matches")
                        recg_nm="No Face"
                        print(recg_nm)
                        self.noFcFlg = self.noFcFlg + 1
                cv2.imshow("FaceRecognition", frame)
                curr_time = datetime.now()
                c = curr_time-strt_time
                diff_secs=int(c.total_seconds())
                cv2.waitKey(1)
                if diff_secs >= 8:
                    # print("breaking",)
                    recg_nm = self.final_name()
                    print("Before breaking",recg_nm)
                    break
            # print("Releasing video capturing", self.currTm())
            self.logger.warning("{0}".format("Releasing video capture"))
            video_capture.release()
            # print("Released video capturing",self.currTm())
            self.logger.warning("{0}".format("Released video capture"))
            cv2.destroyAllWindows()
            # print("Streaming ended",self.currTm())
            self.logger.warning("{0}".format("Streaming ended"))
        except Exception as e:
            self.logger.error(traceback.format_exception())
        return recg_nm

    def set_scrn_sz(self):
        wdth = 1280
        hght = 720
        screen_height, screen_width = 2500,2500
        wdth = screen_width
        hght = screen_height
        print(os.name)
        try:
            if os.name!="":
                # screen_height, screen_width = pyautogui.size()
                print("screen WH",screen_width,screen_height)
            if screen_width != 0 and screen_height != 0:
                wdth = screen_width
                hght = screen_height
        except KeyError:
            self.logger.error("")

        return wdth,hght

    def speak_voice(self,strMsg):
        # speaker = win32com.client.Dispatch("SAPI.SpVoice")
        self.speaker.Speak(strMsg)
        self.logger.warning("{0}{1}".format("speak voice",strMsg))

    def face_detector_mrk_auto_attnd(self):
        try:
            fdetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            # load the known faces and embeddings saved in last file
            face_data = pickle.loads(open('face_enc', "rb").read())
            fc_idnt = self.auto_face_rec_from_camera(fdetector,face_data)
            if type(fc_idnt) is list and len(fc_idnt)==1:
                fc_idnt = fc_idnt[0]
            print(fc_idnt)
            if fc_idnt == "Unknown":
                strMsg = 'Hi Are you visitor to AiFA. Please visit our Chat Window to check visiting purpose !!!\n'
                self.logger.warning('{0}'.format('Unknown Visitor @ AiFA'))
                self.speak_voice(strMsg)
            elif fc_idnt == "No Face":
                self.logger.warning('{0}'.format('No face detected'))
                print("NO face detected")
            else:
                # print("'Hi"+fc_idnt+"!!!\nMarked attendance")
                strMsg = '{0}{1}'.format("Hi",fc_idnt,'!!!\n')
                self.logger.warning('{0}{1}'.format("Hi",fc_idnt,'!!!\n'))
                self.markAttendance(fc_idnt,strMsg)
                self.logger.warning('{0}'.format('closed file'))
        except Exception as e:
            self.logger.error(e)
        return fc_idnt


    # get paths of each file in folder named Images
# Images here contains my data(folders of various persons)
    def identify_face(self,id_to_train,log_obj=None):
        idnt_face = "Unknown"
        try:
            # print(self.wdth,self.hght)
            self.logger = log_obj
            self.match_dict = {}
            self.noFcFlg = self.fcFlg = 0
            # if self.logger is None:
            #     self.set_log('attendance.log')
            # # else:
            # #     print("already logger was set")
            lst_new=[]
            self.load_names_csv()
            # print(self.emp_names_dict)
            # print(id_to_train)
            if not os.path.exists('face_enc'):
                # print("face_enc not exist-ing")
                self.logger.warning('{0}'.format("face_enc not exist-ing"))
                self.dir_lst_img_fldr = os.listdir(self.img_fldr)
                self.logger.warning('{0}'.format(self.dir_lst_img_fldr))
                self.train_faces(self.dir_lst_img_fldr)
            else:
                lst_new.append(str(id_to_train))
                # print(lst_new)
                self.logger.warning('face encoding existing {0}'.format(lst_new))
                if len(lst_new)==1 and lst_new[0]!='':
                    # lst_new=[str(id_to_train)]#['16','17','18']
                    # print("face_enc existing")
                    self.logger.warning('{0}'.format("face_enc existing"))
                    face_data = pickle.loads(open('face_enc', "rb").read())
                    self.known_encodings = face_data['encodings']
                    self.known_names = face_data['names']
                    # print(len(lst_new),lst_new)
                    self.logger.warning('{0}{1}'.format(len(lst_new),lst_new))

                    if len(lst_new) >=1:
                        # print("training the new data")
                        self.logger.warning('{0}'.format("training the new data"))
                        self.train_faces(lst_new)
                    else:
                        # print("NO new data to train the model")
                        self.logger.warning('{0}'.format("NO new data to train the model"))
            idnt_face = self.face_detector_mrk_auto_attnd()
        except Exception as e:
            self.logger.error(e)
        return idnt_face