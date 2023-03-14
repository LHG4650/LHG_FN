import cv2
from ftplib import FTP
from io import BytesIO

'''
write : 23.03.14
ver 1.1.3
애러처리 잘 안되어있음~ 제보바람~
### 1.1.3
나스와 워크스테이션 함께 쓸 수 있음
함수 실행전 현재 접속이 정상인지부터 확인
접속이 원활하지 않으면 안된다고 print문이 생김
'''

class HG_FTP:
    def __init__(self):
        ''' 이닛 없음다~ 
        기본 사용법 
        ### 선언 ###
        NAS = HG_FTP()
        address = '123,123,123,123'
        ID = 'user'
        PW = 'hghg'
        NAS.Set_login(address,ID,PW)
        print(NAS.get_directory())
        
        ### 함수규칙 ###
        함수는 경로가 중요합니다 server_path 와 local_path로 구분되는데
        server_path가 항상 먼저 사용됩니다.
        '''
        self.ftp = False

    def Set_login(self, address, user, pw):
        self.Address = address
        self.User_ID = user
        self.User_PW = pw

        if all([self.Address, self.User_ID, self.User_PW]):
            self._connection()
        else:
            print('address, ID, PW 가 유효하지 않습니다.')

    def _connection(self):
        try:
            self.ftp = False
            ftp = FTP(self.Address)
            ftp.login(self.User_ID,self.User_PW)
            ftp.set_pasv(True)
            ftp.encoding = 'utf-8'
            self.ftp = ftp
            print('접속 성공')
        except:
            print('ftp 접속 실패했습니다.')

    def _check_connection(self):
        try:
            self.ftp.voidcmd('NOOP')
            return True
        except:
            print('ftp 접속이 안되어있습니다. 접속 시도합니다.')
            self._connection()
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
        if self._check_connection():
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
        if self._check_connection():
            server_data_path = self.path_check(server_data_path,'server')
            fd = open(local_save_path, 'wb')
            self.ftp.sendcmd('OPTS UTF8 ON')
            self.ftp.retrbinary("RETR "+server_data_path, fd.write)
            fd.close()

    def upload_file(self,server_save_path,local_file_path):
        ''' ex) NAS.upload_file('/Project/text.txt','테스트 업로드.txt')
                NAS.upload_file('NAS 저장경로','올릴 파일 경로',)       '''
        if self._check_connection():
            server_save_path = self.path_check(server_save_path,'server')
            local_file_path = self.path_check(local_file_path,'not')
            with open(file=local_file_path, mode='rb') as wf:
                self.ftp.storbinary('STOR '+server_save_path,wf)

    def upload_img_directly(self,server_save_path,cv_img):
        ''' ex) NAS.upload_img_directly('/Project/text.txt'cv_img)
                NAS.upload_img_directly('NAS 저장경로', cv2 이미지 바로 ) '''
        if self._check_connection():
            server_save_path = self.path_check(server_save_path,'server')
            _retval, buffer = cv2.imencode('.jpg', cv_img)
            flo = BytesIO(buffer)
            self.ftp.storbinary('STOR '+server_save_path,flo)

    def path_check(self, path, mode = 'server' ):
        if '\\' in path:            #나스경로에 역슬래시 안먹음. #os.path.join 쓰면 역슬래시 생김ㅋㅋ 
            path = path.replace('\\','/',200)

        if mode == 'server':           #나스경로는 처음이 슬래시로 시작해야함
            if path[0] == '/':
                pass
            else: 
                path = '/' + path
        return path
            


