from ftplib import FTP

'''
write : 22.11.17
ver 1.1.1
애러처리 잘 안되어있음~ 제보바람~
'''

class AI_NAS():
    def __init__(self,NAS_Address, NAS_User, NAS_Password):
        ''' 
            EX)
            FTPADDRESS = '10.200.109.217' <<AI추진팀 나스
            FTPUSER  = '[user_id]'
            FTPPASSWORD = '[password]'
            NAS = AI_nas(FTPADDRESS,FTPUSER,FTPPASSWORD)
        '''
        ftp = FTP(NAS_Address)
        ftp.login(NAS_User,NAS_Password)
        self.ftp = ftp
        self.ftp.encoding = 'utf-8'    #파일명에 한글이 들어있거나 하면 꺠짐 --> 동원ai NAS utf-8 코딩으로 설정

    def get_directory(self, path = '/'):
        '''
        ex)
        dt = NAS.get_directory('/Project')
        [파일 권한?, 권한?, 작성자, 그룹?, 용량?, 월, 일, 시간, 파일명]
        '''
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

    def download_file(self,nas_foler_path, nas_file_path, new_file_path):
        '''
        ex)
        NAS.download_file('/Project','폴더 설명.txt','폴더 설명.txt')
        NAS.download_file('나스폴더경로','나스 파일명','다운로드할 파일경로')
        '''

        self.ftp.cwd(nas_foler_path)
        fd = open(new_file_path, 'wb')
        self.ftp.encoding = 'utf-8'
        self.ftp.sendcmd('OPTS UTF8 ON')
        self.ftp.retrbinary("RETR "+nas_file_path, fd.write)
        fd.close()

    def upload_file(self,nas_foler_path,upload_file_path,save_name):
        '''
        ex)
        NAS.upload_file('/Project','테스트 업로드.txt','text.txt')
        NAS.upload_file('NAS 폴더경로','올릴 파일 경로','NAS에 저장될 파일명')
        '''
        self.ftp.cwd(nas_foler_path)
        with open(file=upload_file_path, mode='rb') as wf:
            self.ftp.storbinary('STOR '+save_name,wf)
        


