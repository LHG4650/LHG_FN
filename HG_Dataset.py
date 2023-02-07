import os
import shutil
import cv2
from utill import make_dir,HG_Progress
from DW_AI.DW_detect import DW_detect
from Yolo_train_helper import pred_to_label_txt3
import pandas as pd
import copy
import numpy as np
import math

class SD_Classification_Dataset_Maker:
    def __init__(self, project_folder,yolo_folder,Detect_pt) -> None:
        self.project_folder = project_folder
        self.Detect_pt = Detect_pt
        self.yolo_path = yolo_folder

        self.Detect_yolo = False

        self.set_init_path(project_folder)
        self.set_init_folder(project_folder)
        self.set_init_meta(project_folder)
        self.get_meta_data()
    
    def set_init_path(self,project_folder):
        ''' 메타데이터들 경로 선언 '''
        self.meta_class_path = os.path.join(project_folder,'meta','Class_DB.csv')
        self.meta_data_num_path = os.path.join(project_folder,'meta','Data_num.csv')
        self.meta_data_label_path = os.path.join(project_folder,'meta','Label_data.csv')
        self.meta_local_config = os.path.join(project_folder,'meta','Local_label_config.csv')
        self.meta_Code_PClass_path = os.path.join(project_folder,'meta','Code_PClass.csv')
    
    def set_init_folder(self,project_folder):
        ''' yolo 학습용 데이터 기초 구조 생성 '''
        self.tree_1 = ['images','labels']
        self.tree_2 = ['train','val','test']
        for one in self.tree_1:
            make_dir(os.path.join(project_folder,one))
            for two in self.tree_2:
                make_dir(os.path.join(project_folder,one,two))
        make_dir(os.path.join(project_folder,'meta'))

    def get_meta_data(self):
        ''' 메타데이터 반영'''
        self.Class_DB = pd.read_csv(self.meta_class_path).drop('Unnamed: 0',axis=1)
        self.Data_Num_DB = pd.read_csv(self.meta_data_num_path).drop('Unnamed: 0',axis=1)
        self.Data_label_DB = pd.read_csv(self.meta_data_label_path).drop('Unnamed: 0',axis=1)
        if os.path.exists(self.meta_local_config):
            self.meta_local_DB = pd.read_csv(self.meta_local_config)
        if os.path.exists(self.meta_Code_PClass_path):
            self.Code_PClass = pd.read_csv(self.meta_Code_PClass_path,encoding='cp949')
    
    def set_init_meta(self,project_folder):
        ''' 메타데이터 없으면 초기 양식 생성 '''
        meta_class_path = self.meta_class_path
        meta_data_num_path = self.meta_data_num_path
        meta_data_label_path = self.meta_data_label_path
        meta_local_config = self.meta_local_config
        meta_Code_PClass_path = self.meta_Code_PClass_path

        if not os.path.exists(meta_class_path):
            init = {'PClass' :[],'YoloP' :[] }
            init = pd.DataFrame(init)
            init.to_csv(meta_class_path)

        if not os.path.exists(meta_data_num_path):
            init = {'dir' :[],'PClass' :[],'data_num' :[] }
            init = pd.DataFrame(init)
            init.to_csv(meta_data_num_path)

        if not os.path.exists(meta_data_label_path):
            init = {'dir' :[],'PClass' :[],'base_name' :[] }
            init = pd.DataFrame(init)
            init.to_csv(meta_data_label_path)

        if not os.path.exists(meta_local_config):
            print('Local_label_config DB가 존재하지 않습니다.')
        if not os.path.exists(meta_Code_PClass_path):
            print('Code_PClass DB가 존재하지 않습니다.')


    def add_img(self,img_path,Code):
        ''' img를 데이터셋구조에 넣고 DB를 업데이트함 '''

        if not(self.Detect_yolo):
            self.Detect_yolo_activate()

        base_name = os.path.basename(img_path)
        #print(base_name)
        PClass = self.Code_to_Pclass(Code)
        unknown_status = PClass != 0

        if (not(self._check_img_data(base_name))) and unknown_status:
            #print('타겟에 데이터 ? 없음')
            YoloP = self.PClass_to_YoloP(PClass)
            cv_img = cv2.imread(img_path)
            #print('Code : {} -> PClass : {}'.format(Code,PClass))
            Data_Num_DB_idx, Dir = self.defin_folder(PClass)
            #print('Data_Num_DB_idx : {},  Dir : {}'.format(Data_Num_DB_idx,Dir))
            #print(self.project_folder,'images',Dir,PClass)
            img_folder = os.path.join(self.project_folder,'images',Dir,)
            label_folder = os.path.join(self.project_folder,'labels',Dir)
            make_dir(img_folder)
            make_dir(label_folder)
            _, pred = self.Detect_yolo.AI_detect(cv_img)
            pred_to_label_txt3(label_folder,cv_img,pred,YoloP,base_name)        #라벨넣음
            shutil.copy(img_path,img_folder)    #이미지넣음

            #여기서부터는 DB업데이트임
            self._append_Data_label_DB(Dir,PClass,base_name)
            self._count_add_Data_Num_DB(Data_Num_DB_idx)
            
    def _count_add_Data_Num_DB(self,idx):
        self.Data_Num_DB.loc[idx,'data_num']+=1

    def _append_Data_label_DB(self,dir,PClass,base_name):
        new = pd.DataFrame({'dir' :[dir],'PClass' :[PClass],'base_name' :[base_name] })
        self.Data_label_DB = pd.concat([self.Data_label_DB,new],ignore_index=True)

    def save_db(self):
        self.Data_label_DB.to_csv(self.meta_data_label_path)
        self.Data_Num_DB.to_csv(self.meta_data_num_path)
        self.Class_DB.to_csv(self.meta_class_path)
        self.set_classes()
        print('저장완료')
                
    def _check_img_data(self,img_base_name):
        ''' 이미 데이터에 있는지 확인함 있으면 True 없으면 False'''
        check = (self.Data_label_DB.base_name == img_base_name).sum()
        return bool(check)
            
    def Detect_yolo_activate(self):
        ''' Detect욜로를 활성화시킴 '''
        self.Detect_yolo = DW_detect(self.Detect_pt)
        
    def defin_folder(self,PClass):
        dt = copy.copy(self.Data_Num_DB)
        dt = dt[dt.PClass == PClass][['dir','data_num']]
        data_split = []
        for split in ['train','val','test']:
            target = dt[dt.dir == split].data_num
            if target.__len__():
                data_split.append(int(target))
            else:
                new = pd.DataFrame({'dir' :[split],'PClass' :[PClass],'data_num' :[0] })
                self.view = new
                self.Data_Num_DB = pd.concat([self.Data_Num_DB,new],ignore_index=True)
                data_split.append(0)
        summ = sum(data_split)
        if data_split[0] <= summ*0.8:
            Dir =  'train'
        elif data_split[1] < summ*0.1:
            Dir =  'val'
        elif data_split[2] < summ*0.1:
            Dir =  'test'

        condition_1 = self.Data_Num_DB.dir == Dir
        condition_2 = self.Data_Num_DB.PClass == PClass
        Data_Num_DB_idx = self.Data_Num_DB[condition_1&condition_2].index.values[0]
        return Data_Num_DB_idx, Dir
    
    def Code_to_Pclass(self,Code):
        dt = self.Code_PClass
        result = dt[dt.Code == int(Code)].PClass
        if result.values == "unknown":
            return 0
        else:
            return int(result.values)

    def _check_Class_DB(self,PClass):
        check = (self.Class_DB.PClass == PClass).sum()
        return bool(check)
    
    def PClass_to_YoloP(self,PClass):
        if not(self._check_Class_DB(PClass)):   #DB에 없으면
            num = self.Class_DB.YoloP.__len__()
            new = pd.DataFrame({'PClass' :[PClass],
                                'YoloP' :[num] })
            self.Class_DB = pd.concat([self.Class_DB,new],ignore_index=True)
        dt = self.Class_DB
        result = dt[dt.PClass == int(PClass)].YoloP
        self.view = dt[dt.PClass == int(PClass)]
        return int(result.values)

    def set_classes(self):
        class_list = self.Class_DB.PClass.tolist()
        folder = ['train','val','test']
        for i in folder:
            path = os.path.join(self.project_folder,'labels',i,'classes.txt')
            with open(path,'w')as f:
                for Cls in class_list:
                    f.write(str(Cls) + "\n")

    def make_yaml(self):

        train_path = os.path.join(self.project_folder,'images','train')
        val_path = os.path.join(self.project_folder,'images','val')
        test_path = os.path.join(self.project_folder,'images','test')

        train_list = os.listdir(train_path)
        train_list = list(map(lambda x: os.path.join(train_path,x), train_list))

        val_list = os.listdir(val_path)
        val_list = list(map(lambda x: os.path.join(val_path,x), val_list))

        test_list = os.listdir(test_path)
        test_list = list(map(lambda x: os.path.join(test_path,x), test_list))

        class_list = []
        classes_path = os.path.join(self.project_folder,'labels','train','classes.txt')
        with open(classes_path , 'r')as f:
            a = f.readlines()
            for i in a:
                if '\n' in i:
                    class_list.append(i[:-1])
                else:
                    class_list.append(i)
            f.close

        w_line1 = 'nc: ' + str(class_list.__len__())
        w_line2 = "names: ["
        for i in class_list:
            pre = "'"
            back = "', "
            w_line2 = w_line2 + pre + i + back
        w_line2 = w_line2[:-2] + ']'

        meta_folder = os.path.join(self.project_folder,'meta')
        with open(meta_folder+"/train.txt", 'w')as f:
            f.write('\n'.join(train_list)+'\n')

        with open(meta_folder + "/val.txt", 'w')as f:
            f.write('\n'.join(val_list)+'\n')

        with open(meta_folder + "/test.txt", 'w')as f:
            f.write('\n'.join(test_list)+'\n')

        with open(meta_folder + "/data.yaml", 'w')as f:
            f.write('train: ' + os.path.join(meta_folder,'train.txt') +'\n')
            f.write('val: ' + os.path.join(meta_folder,'val.txt') +'\n')
            f.write('test: '   + os.path.join(meta_folder,'test.txt') +'\n'+'\n')
            f.write(w_line1+'\n')
            f.write(w_line2+'\n')

    def get_train_command(self, yolo_model_select = 's', epochs=200 , name='SD_Classification', batch=256):
        
        assets = ['n', 's', 'm', 'l', 'x']
        if yolo_model_select in assets:
            yolo_model_select = os.path.join(self.yolo_path,'models','yolov5'+yolo_model_select+'.yaml')
        
        yaml_path = os.path.join(self.project_folder,'meta','data.yaml')
        weight_path = os.path.join(self.yolo_path,'yolov5s.pt')
        yolo_train_path = os.path.join(self.yolo_path,'train.py')

        options = { 'img': '640',  # 이미지 높이
                    'batch': str(batch),  # 배치 높이
                    'epochs': str(epochs),
                    'data': yaml_path,
                    'cfg': yolo_model_select,
                    'weight': weight_path,
                    'name': name,
                    'workers': '0'}

        opt = "python " + yolo_train_path
        for i in options.keys():
            pre = ' --'
            back = ' '
            opt = opt + pre + i + back + options[i]
        print('----일반 입력값---------------------')
        print(opt)
        print('----Work_station 입력값---------------------')
        opt = "python -m torch.distributed.run --nproc_per_node 2 "+yolo_train_path
        for i in ['batch', 'epochs', 'data', 'cfg', 'weight', 'name']:
            pre = ' --'
            back = ' '
            opt = opt + pre + i + back + options[i]
        print(opt+' --device 0,1')
        print('--------------------------------')
