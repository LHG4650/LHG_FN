import os
import cv2
from io import BytesIO
from ftplib import FTP
'''
write : 23.03.15
ver 1.2.0 애러처리 잘 되어있음~ 문제제보바람~
### 1.2.0
    변수명 함수명 변경
    PEP8
### 1.1.4
    폴더 옮기는 모듈 추가
### 1.1.3
    나스와 워크스테이션 함께 쓸 수 있음
    함수 실행전 현재 접속이 정상인지부터 확인
    접속이 원활하지 않으면 안된다고 print문이 생김

'''
class HG_FTP:
    def __init__(self) -> None:
        print('HgFtp 로 변경되었으니 변경하세요~ ')


class HgFtp:
    def __init__(self):
        ''' 이닛 없음다~ 
        기본 사용법 
        ### 선언 ###
        NAS = HgFtp()
        address = '123,123,123,123'
        ID = 'user'
        PW = 'hghg'
        NAS.set_login(address,ID,PW)
        print(NAS.get_directory())
        
        ### 함수규칙 ###
        함수는 경로가 중요합니다 server_path 와 local_path로 구분되는데
        server_path가 항상 먼저 사용됩니다.
        '''
        self.ftp = False

    def set_login(self, address, user, pw):
        self.address = address
        self.user_id = user
        self.__user_pw = pw

        if all([self.address, self.user_id, self.__user_pw]):
            self.__connection()
        else:
            print('address, ID, PW 가 유효하지 않습니다.')

    def __connection(self):
        try:
            self.ftp = False
            ftp = FTP(self.address)
            ftp.login(self.user_id,self.__user_pw)
            ftp.set_pasv(True)
            ftp.encoding = 'utf-8'
            self.ftp = ftp
            print('접속 성공')
        except:
            print('ftp 접속 실패했습니다.')

    def __check_connection(self):
        try:
            self.ftp.voidcmd('NOOP')
            return True
        except:
            print('ftp 접속이 안되어있습니다. 접속 시도합니다.')
            self.__connection()
            if self.ftp:
                return True
            else:
                print('해당 함수는 실행되지 않습니다')
                return False

    def get_directory(self, path = '/'):
        '''
        ex)
        dt = NAS.get_directory('/Project')
        [파일 권한?, 권한?, 작성자, 그룹?, 용량?, 월, 일, 시간, 파일명]
        '''
        if self.__check_connection():
            data = []
            self.ftp.cwd(path)
            self.ftp.dir(data.append)
            out_list = []
            for i in data:
                dt = i.split(' ')
                dt = list(filter(None,dt))
                one_dt = dt[:8]
                two_dt = ' '.join(dt[8:])       #파일명에 띄어쓰기 있는경우 합치기
                one_dt.append(two_dt)

                out_list.append(one_dt)
            return out_list
        else:
            return False

    def download_file(self, server_data_path, local_save_path):
        ''' ex) NAS.download_file('/Project/폴더 설명.txt','폴더 설명.txt')
                NAS.download_file('nas 다운로드 파일명 ','파일 저장경로') '''
        if self.__check_connection():
            server_data_path = self.__path_check(server_data_path,'server')
            fd = open(local_save_path, 'wb')
            self.ftp.sendcmd('OPTS UTF8 ON')
            self.ftp.retrbinary("RETR "+server_data_path, fd.write)
            fd.close()

    def upload_file(self,server_save_path,local_file_path):
        ''' ex) NAS.upload_file('/Project/text.txt','테스트 업로드.txt')
                NAS.upload_file('NAS 저장경로','올릴 파일 경로',)       '''
        if self.__check_connection():
            server_save_path = self.__path_check(server_save_path,'server')
            local_file_path = self.__path_check(local_file_path,'not')
            with open(file=local_file_path, mode='rb') as wf:
                self.ftp.storbinary('STOR '+server_save_path,wf)

    def upload_img_directly(self,server_save_path,cv_img):
        ''' ex) NAS.upload_img_directly('/Project/text.txt'cv_img)
                NAS.upload_img_directly('NAS 저장경로', cv2 이미지 바로 ) '''
        if self.__check_connection():
            server_save_path = self.__path_check(server_save_path,'server')
            _retval, buffer = cv2.imencode('.jpg', cv_img)
            flo = BytesIO(buffer)
            self.ftp.storbinary('STOR '+server_save_path,flo)

    def __path_check(self, path, mode = 'server' ):
        #print('get',path)
        if '\\' in path:            #나스경로에 역슬래시 안먹음. #os.path.join 쓰면 역슬래시 생김ㅋㅋ 
            path = path.replace('\\','/',200)

        if mode == 'server':           #나스경로는 처음이 슬래시로 시작해야함
            if path[0] == '/':
                pass
            else: 
                path = '/' + path
        #print('get',path)
        return path

    def upload_folder(self, server_save_folder_path, local_data_folder_path):
        server_save_folder_path = self.__path_check(server_save_folder_path, mode = 'server')
        local_data_folder_path  = self.__path_check(local_data_folder_path, mode = 'not')

        for name in os.listdir(local_data_folder_path):
            
            server_file_path = os.path.join(server_save_folder_path, name)
            local_file_path = os.path.join(local_data_folder_path, name)
            server_file_path = self.__path_check(server_file_path, mode = 'server')
            local_file_path = self.__path_check(local_file_path, mode = 'not')

            if os.path.isfile(local_file_path):
                self.upload_file(server_file_path,local_file_path)
                
            elif os.path.isdir(local_file_path):
                # 서버에 새로운 폴더 생성
                self.ftp.mkd(server_file_path)

                # 하위 폴더 및 파일 재귀적으로 업로드
                self.upload_folder(server_file_path, local_file_path)

