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

def img_compare(img1,img2,methods=0):
    ''' 1. cv2.HISTCMP_CORREL : 상관관계
            1: 완전 일치, -1: 완전 불일치, 0: 무관계
            빠르지만 부정확
        2. cv2.HISTCMP_CHISQR : 카이제곱 검정(Chi-Squared Test)
            0: 완전 일치, 무한대: 완전 불일치
        3. cv2.HISTCMP_INTERSECT : 교차
            1: 완전 일 치, 0: 완전 불일치(히스토그램이 1로 정규화된 경우)
        4. cv2.HISTCMP_BHATTACHARYYA : 바타차야 거리
            0: 완전 일치, 1: 완전 불일치
            느리지만 가장 정확
        5. EMD
            직관적이지만 가장 느림
    '''
    imgs = [img1,img2]
    #methods = 'CORREL'
    hists = []
    for img in imgs:
        # BGR 이미지를 HSV 이미지로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 히스토그램 연산(파라미터 순서 : 이미지, 채널, Mask, 크기, 범위)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # 정규화(파라미터 순서 : 정규화 전 데이터, 정규화 후 데이터, 시작 범위, 끝 범위, 정규화 알고리즘)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        # hists 리스트에 저장
        hists.append(hist)
    
    ret = cv2.compareHist(hists[0], hists[1], methods)
    return ret

def variance_of_laplacian(img2):
    # 흐림의 정도 확인
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    gray = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _BGR2RGB(BGR_img):
    # turning BGR pixel color to RGB
    rgb_image = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
    return rgb_image

def blurrinesDetection(directories,threshold):
    columns = 3
    rows = len(directories)//2
    fig=plt.figure(figsize=(5*columns, 4*rows))
    for i,directory in enumerate(directories):
        fig.add_subplot(rows, columns, i+1)
        img = cv2.imread(directory)
        text = "Not Blurry"
        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry
        fm = variance_of_laplacian(img)
        if fm < threshold:
            text = "Blurry"
        rgb_img = _BGR2RGB(img)
        cv2.putText(rgb_img, "{}: {:.2f}".format(text, fm), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        plt.imshow(rgb_img)
    plt.show()
    
def crop_yolo_img(img,pred):
    pred = pred[0][0].int().cpu().tolist()
    pred = np.clip(pred,0,640)
    img = img[pred[1]:pred[3],pred[0]:pred[2]]
    return img

if __name__ == "__main__":