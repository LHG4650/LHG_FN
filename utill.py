import time
import os

def time_stamp():
    time_stamp = time.strftime('%y.%m.%d_%H.%M.%S',time.localtime())
    return time_stamp

class HG_Progress():
    '''
    학습이나 반복문 돌릴때 오래걸리고 언제끝나는지 궁금할때 진행현황 보기 위해
    만든 클래스입니다.
    
    ※ 주의 
        반복문 안에 다른 프린트 되는 사항이 있으면 지저분하게 나옵니다.
        시간은 추정값입니다.
    
    ※ 사용법 
        1: 선언시 클래스에 반복하려는 횟수 입력
        2: 반복문 제일 위쪽에 클래스 호출한번하기

    ※ ex)
        epochs = 10000
        HGP = HG_Progress(epochs)
        for i in range(epochs):
            HGP()
            model.train()

    '''
    def __init__(self,total_repeat_count):
        self.start_time = time.time()
        self.t_0 = time.time()
        self.t_1 = time.time()
        self.total_repeat_count = total_repeat_count
        self.idx = 0

    def __call__(self):
        t_1 = time.time()
        self.idx +=1
        elapse_time = (t_1 - self.start_time) # 경과시간
        a_progress_time = elapse_time / self.idx # 개당 걸리는 시간
        remain_time = a_progress_time*(self.total_repeat_count-self.idx) #남은 시간
        #pred_time = (t_1 + remain_time)+60*60*9 #종료 예상시간
        pred_time = self.total_repeat_count * ( elapse_time ) / self.idx + self.start_time +1*60*60*9#종료 예상시간

        progress_txt    =' 진행율 '    + str(round(self.idx/self.total_repeat_count*100,2))+"% /"
        elapse_time_txt =' 경과 시간 ' + time.strftime("%H:%M:%S", time.gmtime(elapse_time)) +" /"
        remain_time_txt =' 잔여 시간 ' + time.strftime("%H:%M:%S", time.gmtime(remain_time)) +" /"
        pred_time_txt   =' 종료 예상 ' + time.strftime("%H:%M:%S", time.gmtime(pred_time))   +" /"

        txt = '\r'+progress_txt+ elapse_time_txt +remain_time_txt+pred_time_txt
        print(txt,end="")

def make_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: creationg direictory.  "  + path)
        
