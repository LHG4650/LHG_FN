import cv2
import matplotlib
import matplotlib.pyplot as plt
#ㅜ 아래 라인이 입력되야 vscode에서 셀 밑 라인에 표시가 된다.
#%matplotlib inline                              

plt.rcParams['font.family'] ='Malgun Gothic'        #<<< 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] =False           #<<< -(마이너스) 부호 깨짐 방지

class ImgWindow():
    def __init__(self):
        print('주피터 노트북이라면 아래를 라인에 추가하세요')
        print('%matplotlib inline')
        self.figure_size = (25,5)      #총 이미지창의 *(가로, 높이) 가로는 25가 적당, 높이는 투입된 이미지 수에 달려있음
        self.img_dic = {}
        self.img_list = []
        self.count =0
        self.a_img_high = 5 #이미지 한장의 높이
    
    def set_img(self, name, img):
        ''' 이름, 이미지 넣으세요'''
        self.count +=1
        self.img_dic[name+ str(self.count)] = img
        self.img_list.append(name + str(self.count))
    
    def show_img(self):
        img_count = self.img_list.__len__()
        
        figure_high =(img_count//4 + 1)*self.a_img_high
        fig_h_num = figure_high/self.a_img_high
        self.figure_size = (self.figure_size[0], figure_high)           #전체 이미지의 크기( 가로 25 초기값 고정 , 세로 총이미지장수 /4의 몫 +111)
        plt.figure(figsize=self.figure_size)
        print(self.figure_size)
        count = 0
        for i in self.img_list:
            
            count += 1
            coordinate = '4' + str(int(fig_h_num)) + str(count)
            count = int(count)
            #print(coordinate)
            plt.subplot(int(fig_h_num),4,count)                     #subplot 이미지는>> 정의된다. ( 이미지 세로갯수, 가로갯수, 위치)  ex) (2,4,1) / (2,4,2) / (2,4,3) / (2,4,4)
            plt.title(i)                                                                                                       #      (2,4,5) / (2,4,6) / (2,4,7) / (2,4,8)
            #print(i)
            a_img = cv2.cvtColor(self.img_dic[i],cv2.COLOR_BGR2RGB)
            plt.imshow(a_img)

        plt.show()

