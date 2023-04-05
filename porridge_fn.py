from re import L
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
import cv2
import copy

def pred_2_input(pred,mi):
    '''
        yolo의 pred 를 인풋으로 받아
        Class 가 0이면 스푼
        Class 가 1이면 참기름으로 하여
        한 객체당
        12개의 변수로 저장된다.
        [x1,x2,y1,y2,prob,class] + [w,h,area,tx,ty,dist]
        w = 넓이 x2 - x1
        h = 높이 y2 - y1
        area = 부피 w*h
        tx = (x1+x2)/2
        ty = (y1+y2)/2
        dist = [tx,ty] <-> [mix,miy] 의 거리의 제곱

        숫가락과 참기름은 최대 2객채가지 저장하고
        객체가 없는경우 0을 넣어둔다

        4객체 * 12변수 + 2(원의중심변수) = 50변수로 인풋을 만들어준다.
    
    '''


    sp1 = [0,0,0,0, 0,0] # x1,y1,x2,y2, prob,class,  //  w,h,  area, tx,ty, dist,
    sp2 = [0,0,0,0, 0,0] 
    cham1 =[0,0,0,0, 0,0] 
    cham2 = [0,0,0,0, 0,0] 

    #print(pred.tolist())
    for i in pred.tolist():
        #print('---i',i)
        if int(i[5]) == 0:
            if sum(sp1) == 0:
                sp1 = i
            else:
                sp2 = i

        if int(i[5]) == 1:
            if sum(cham1) == 0:
                cham1 = i
            else:
                cham2 = i
    result = []
    for i in [sp1,sp2,cham1,cham2]:
        w = i[2] - i[0]
        h = i[3] - i[1]
        area = w*h
        tx = (i[2] + i[0])/2
        ty = (i[3] + i[1])/2
        dist = (tx-mi[0])*(tx-mi[0]) + (ty-mi[1])*(ty-mi[1])
        result = result + i + [w,h,area,tx,ty,dist]
        #print([w,h,area,tx,ty,dist],'prprprp',[w,h,area,tx,ty,dist].__len__())
        #print(i.__len__(),'??',i)
        #print('target',result.__len__())
    result = result + mi

    return torch.Tensor(result)

def find_out_role(result):
    LR_check = 0
    midx = result[48]
    
    role_result = 1

    if (midx > 200)&(midx < 330):
        LR_check = "L"
    elif (midx > 375)&(midx < 450):
        LR_check = "R"
    else:
        LR_check = 0

    if LR_check == "L":             #L측기준        단측 95%
        if result[11] > 3776:       #숫가락 넘어감
            role_result = 0
        elif result[35] > 4637:     #참기름 넘어감
            role_result = 0
        elif result[32] > 13333:    #참기름 크기 넘어감
            role_result = 0

    if LR_check == "R":             #R측 기준
        if result[11] > 5027:       #숫가락 넘어감
            role_result = 0
        elif result[35] > 4680:     #참기름 넘어감
            role_result = 0
        elif result[32] > 29888:    #참기름 크기 넘어감
            role_result = 0
    
    if LR_check == 0:
        role_result = 0

    return role_result

def Check_accuracy(x,y,fn):
    View_table = [[0,0],[0,0]]

    for i in range(x.index.tolist().__len__()):
        question = x.loc[x.index.tolist()[i]].tolist()
        pred = fn(question)
        answer = y.loc[y.index.tolist()[i]].tolist()
        
        View_table[int(pred)][int(answer)]+=1
    
    TP = View_table[1][1]
    FP = View_table[1][0]
    FN = View_table[0][0]
    TN = View_table[0][1]
    ALL = (TP+FP+FN+TN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    Accuracy = (TP+FN)/ALL
    F1_Score = 2*(Precision*Recall)/(Precision+Recall)

    print('         False / True / sum')
    print('pred(0)/',View_table[0],'|',FN+TN)
    print('pred(1)/',View_table[1],'|',TP+FP)
    print('sum    / ',FN+FP,TN +TP,' |',ALL )
    print('TP True Positive  : ',TP,'\t/',round(TP/ALL*100,2),'%')
    print('FP False Positive : ',FP,'\t/',round(FP/ALL*100,2),'%','\t 2종오류')
    print('FN False Negative : ',FN,'\t/',round(FN/ALL*100,2),'%')
    print('TN True Negative  : ',TN,'\t/',round(TN/ALL*100,2),'%','\t 1종오류')
    print('')
    print('Precision 정밀도 :',round(Precision*100,2),"%")
    print('예측모델이 맞았다고한게 정말 맞을 확률')
    print('Recall 재현율    :',round(Recall*100,2),"%")
    print('정상중에 예측모델이 맞앗다고 할 확률')
    print('Accuracy 정확도  :',round(Accuracy*100,2),"%")
    print('모델의 정확도')
    print('F1 Score         :',round(F1_Score*100,2),"%")
    print('Precision과 Recall의 조화평균')

class Porridge_NN(torch.nn.Module):
    def __init__(self, input = 20, h1 = 8, h2 = 9, out_put = 2):
        super().__init__()

        self.fc1 = nn.Linear(input, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_put)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
    
        return x
        
def get_item_info(sp_pred, mid):
    if torch.is_tensor(sp_pred)== 1:
        sp_pred = sp_pred.cpu().numpy()
    mid = mid[0],mid[1]
    sp_midx = (sp_pred[0]+sp_pred[2])/2
    sp_midy = (sp_pred[1]+sp_pred[3])/2
    sp_dist = math.dist([sp_midx,sp_midy], mid)

    point_dist_list =[]
    for i in [sp_pred[0],sp_pred[2]]:
        for j in [sp_pred[1],sp_pred[3]]:
            point_dist_list.append(math.dist([i,j],mid))
       
    point_dist_list.sort(reverse=True)
    point_dist_list = point_dist_list[:2]

    sp_w = (sp_pred[2]+sp_pred[0])/2
    sp_h = (sp_pred[3]+sp_pred[1])/2
    sp_area = sp_w * sp_h

    sp_info = [sp_dist] + point_dist_list +[sp_w,sp_h,sp_area]
    #print(sp_info)
    return sp_info

def pred_to_input(pred,mid):

    sp_count = 0
    cham_count = 0

    input = []
    for i in pred:
        if i[5] == 0:
            #print(i)
            input = get_item_info(i,mid)
            sp = i
            sp_count +=1
            break
            
    for i in pred:
        if i[5] == 1:
            if cham_count <3 :
                cham_count +=1
                input += get_item_info(i,mid)

    if cham_count == 1:
        input += get_item_info([0,0,0,0,0,1],mid)

    if cham_count == 0:
        input += get_item_info([0,0,0,0,0,1],mid)
        input += get_item_info([0,0,0,0,0,1],mid)

    if sp_count == 0:
        input += get_item_info([0,0,0,0,0,1],mid)
    
    input += [mid[0],mid[1]]

    return input

def red_find_circle_v1(img):
    #이미지의 원을 찾는 함수
    #처음 들어오는 이미지는 전처리가 없다는 가정으로 한다.
    #이미지전처리
    val = 60
    alpha = 1.0
    color = 10
    range = 10
    red_mul = 2
    
    redNorigin_addWeighted = 150/2

    img_origin = img
    #print(img_origin.shape)
    #기본전처리 +60 / clip alpha1 보정
    addarray = np.full((480,640,3), (val,val,val), dtype= np.uint8)    # val 로 이루어진 도화지 전체적으로 조금 하얗게 하는게 목적
    
    #view.set_img('+60 add',copy.deepcopy(addarray)) # 확인후 주석처리

    add_img = cv2.add(img_origin,addarray)
    #view.set_img('add_img',copy.deepcopy(add_img))# 확인후 주석처리


    imgmultiple = np.clip((1+alpha)*add_img - 128*alpha, 0, 255).astype(np.uint8)       #clip 이미지형식 (0~255)를 맞게해줌 / alpha는 rgb배율 1> 2배 

    #view.set_img('clip _alpha1',copy.deepcopy(imgmultiple))# 확인후 주석처리

    #빨강색 범위 *2 하여 빨간색 마스크 영역을 구하는게 목적
    hsv = cv2.cvtColor(imgmultiple, cv2.COLOR_BGR2HSV)
    lower_red = np.array([color-range, 30, 30])        # 빨강색 범위
    upper_red = np.array([color+range, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    res2 = cv2.bitwise_and(imgmultiple, imgmultiple, mask=mask2)*red_mul      # 여기까지가 빨강색 구하기
    #res*2
    #view.set_img('HSV 붉은색 추출',copy.deepcopy(res2))# 확인후 주석처리

    redNclahe = cv2.addWeighted(imgmultiple,float(100) * 0.01, res2,float(redNorigin_addWeighted) * 0.01,0)     #여기까지가 학습에 이용된 그림 imgmultiple + res2 *1.5

    #view.set_img('clip _alpha1 + 불은색 추출',copy.deepcopy(redNclahe))# 확인후 주석처리

    #그레이로 만들고
    red2gray = copy.deepcopy(res2)
    red2gray = cv2.cvtColor(red2gray, cv2.COLOR_BGR2GRAY)

    #view.set_img('흑백 변환',copy.deepcopy(red2gray))# 확인후 주석처리

    #가우시안 블러 쓰고
    red2gray = cv2.GaussianBlur(red2gray,(5,5),0)
    #view.set_img('가우시안블러',copy.deepcopy(red2gray))# 확인후 주석처리

    #쓰레쉬 홀드로 날리기 흑백마스크만 남기기  흰부분이 레드임
    res, red2gray = cv2.threshold(red2gray, 40, 255, cv2.THRESH_BINARY)
    #view.set_img('쓰레쉬홀드',copy.deepcopy(red2gray))# 확인후 주석처리

    #원찾기
    contours, hierarchy = cv2.findContours(red2gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    right   = [0,0]
    left    = [640,0]
    up      = [0,480]
    down    = [0,0]

    for i in contours:
        if np.array(i).shape[0] > 200:
            for k in i:
                j= k[0]
                #print(j)
                if j[0] > right[0]:
                    right = copy.deepcopy(j)
                if j[0] < left[0]:
                    left = copy.deepcopy(j)
                if j[1] < up[1]:
                    up = copy.deepcopy(j)
                if j[1] > down[1]:
                    down = copy.deepcopy(j)
    
    mid = [int((left[0]+right[0])/2),int((up[1]+down[1])/2)]        #가로세로 중심

    radiv = right[0]-left[0]             #가로지름
    radih = down[1]-up[1]             #세로지름
    radim = (radih+radiv)/2         #지름평균

    if radim < 0:   #음수 ->원 못찾음 -> 640 최대원으로 만들어줌
        radim = 640

    #try:
        #cv2.circle( redNclahe, mid, int(radim/2), (0,255,0),5)      #반지름 넣어야함
    #except:
        #print('원못찾음')
    #view.set_img('radim',origin)

    #addarray = np.full(img_origin.shape, (0,0,0), dtype= np.uint8)      #검은색 도화지
    #cv2.circle( addarray, mid, int(radim/2), (0,255,0),5)       #도화지에 원 그리기
    #view.set_img('붉은색원',copy.deepcopy(addarray))# 확인후 주석처리
    #cv2.circle( redNclahe, mid, int(radim/2), (0,255,0),5)       #도화지에 원 그리기
    #view.set_img('붉은색원',copy.deepcopy(redNclahe))# 확인후 주석처리

    return redNclahe,  mid, int(radim/2)

        


