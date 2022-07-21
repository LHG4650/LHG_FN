from re import L
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
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
    def __init__(self, input = 50, h1 = 8, h2 = 9, out_put = 2):
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
    
    input += [mid[0],mid[1]]

    return input


        


