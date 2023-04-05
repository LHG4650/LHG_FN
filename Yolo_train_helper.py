import os
from LHG_FN import utill
import torch

def pred_to_label(save_file_name : str , cv_img , pred, class_num = False):
    ''' 
    yolo pred를 txt파일로 만듦 
    save_file_name : 저장할 파일 명, 경로
    cv_img : yolo 이미지, 사이즈를 알아야함
    pred : yolo pred 결과물 -> {x1,y1,x2,y2,prob,cls}
    class_num = yolo 클래스 넣고싶은것 비어있으면 기존 cls가 들어감
    * yolo label은 [cls, midx, midy, wide, high] 로 이루어져있음
    '''
    img_h, img_w, _c = cv_img.shape
    write_labels =[]
    for i in pred:
        lu_x = int(i[0]) / img_w
        lu_y = int(i[1]) / img_h
        rd_x = int(i[2]) / img_w
        rd_y = int(i[3]) / img_h
        midx = str(round((lu_x + rd_x)/2,6))
        midy = str(round((lu_y + rd_y)/2,6))
        pred_w = str(round((rd_x - lu_x),6))
        pred_h = str(round((rd_y - lu_y),6))
        if class_num == False:
            cls = str(int(i[5]))
        else:
            cls = str(class_num)
        label = cls + " " + midx + " " + midy + " " + pred_w + " " + pred_h
        write_labels.append(label)

        with open(save_file_name, 'w' )as f:
            for k in write_labels:
                f.write(k+"\n")

def set_classes(folder_path : str, cls_name_list : list):
    ''' yolo classes를 생성하는 함수 
    folder_path = classes생성할 경로
    cls_name_list = 클래스로 만들고싶은걸 넣어주길 바람'''
    class_list = cls_name_list
    folder = ['train','val','test']
    for i in folder:
        path = os.path.join(folder_path,'labels',i,'classes.txt')
        with open(path,'w')as f:
            for Cls in class_list:
                f.write(str(Cls) + "\n")

def yaml_make(folder, train_rate = 0.9, val_rate = 0.1, test_rate = 0, shuffle = True):
    '''
    folder(str) : 무조건 절대경로로 들어와야함
    '''
    main_folder = folder
    img_folder = os.path.join(main_folder, 'images')
    label_folder = os.path.join(main_folder, 'labels')
    meta_folder = os.path.join(main_folder, 'meta')
    utill.make_dir(img_folder)
    utill.make_dir(label_folder)
    utill.make_dir(meta_folder)

    img_list_dumi = os.listdir(img_folder)
    img_list = [] 
    for i in img_list_dumi:
        if '.jpg' in i:
            img_list.append(os.path.join(img_folder,i))

    if train_rate+val_rate+test_rate != 1:
        print( 'train test val rate 의 합이 1이 아닙니다 다시 하십쇼')
        return False
    
    if shuffle:
        import random
        random.shuffle(img_list)

    img_len = img_list.__len__()
    train_img_list = img_list[:int(img_len*train_rate)]
    val_img_list = img_list[int(img_len*train_rate):int(img_len*train_rate)+int(img_len*val_rate)]
    test_img_lsit = img_list[int(img_len*train_rate)+int(img_len*val_rate):]
    print(f'train :{train_img_list.__len__()}, val :{val_img_list.__len__()}, test :{test_img_lsit.__len__()}')
    
    class_list = []
    with open(folder +"/labels/classes.txt", 'r' )as f:
        a = f.readlines()
        for i in a:
            if '\n' in i:
                class_list.append(i[:-1])
            else:
                class_list.append(i)
        f.close
    
    print(f'class_list :{class_list}')

    w_line1 = 'nc: ' + str(class_list.__len__())
    w_line2 = "names: [" 
    for i in class_list:
        pre = "'"
        back = "', "
        w_line2 = w_line2 + pre + i + back
    
    w_line2 = w_line2[:-2] + ']'


    with open(os.path.join(meta_folder,'train.txt'), 'w' )as f:
        f.write('\n'.join(train_img_list)+'\n')
    with open(os.path.join(meta_folder,'val.txt'), 'w' )as f:
        f.write('\n'.join(val_img_list)+'\n')
    with open(os.path.join(meta_folder,'test.txt'), 'w' )as f:
        f.write('\n'.join(test_img_lsit)+'\n')

    with open(os.path.join(meta_folder,'data.yaml'), 'w' )as f:
        f.write('train: '+os.path.join(meta_folder,'train.txt')+'\n')
        f.write('val: '  +os.path.join(meta_folder,'val.txt')+'\n')
        f.write('test: ' +os.path.join(meta_folder,'test.txt')+'\n'+'\n')
        f.write(w_line1+'\n')
        f.write(w_line2+'\n')

def yolo_train_command(yolo_folder, data_folder, name, model_size, batch, epochs, img_size = False):
    print('모든 경로는 절대경로로 넣으세요')
    
    if img_size == False:
        import cv2
        import random
        img_folder = os.path.join(data_folder,'images')
        img_list = os.listdir(img_folder)
        img_path = os.path.join(img_folder,random.choice(img_list))
        img = cv2.imread(img_path)
        img_size = max(img.shape)
        
        if img_size % 32 != 0:
            img_size += 32- img_size % 32 

    yolo_train_path = os.path.join(yolo_folder, 'train.py')
    opt_weight = os.path.join(yolo_folder,'yolov5s.pt')     # 사전학습모델을 쓸것인가?ㅋㅋ
    opt_data = os.path.join(data_folder,'meta','data.yaml')

    assets = ['n', 's', 'm', 'l', 'x']
    if model_size in assets: 
        model_size = yolo_folder + "/models/yolov5"+model_size + '.yaml'
    else:
        model_size = ""
    options =   {
                'img' : str(img_size),      #이미지 높이
                'batch' : str(batch),      #배치 높이
                'epochs' : str(epochs),
                'data' : opt_data,
                'cfg' : model_size,
                'weight' : opt_weight,
                'name' : name,
                'workers' : '0',
            }

    cuda_count = torch.cuda.device_count()
    if cuda_count <= 1:
        opt = "python " + yolo_train_path
        for i in options.keys():
            pre = ' --'
            back = ' '
            opt = opt + pre + i + back + options[i]
        print(' ----입력값---------------------')
        print(opt)
        print('--------------------------------')
    
    else:
        print(f'그래픽 카드 {cuda_count}개')
        print('배치 사이즈 자동조정하였으니 그래픽카드 여러개라고 배치 곱하지 마세여')
        device = ''.join(str(x)+',' for x in list(range(cuda_count)))[:-1]
        options['batchs'] = str(batch*cuda_count)

        print('workstation 용 -----------------')
        opt = "python -m torch.distributed.run --nproc_per_node 2 "+yolo_train_path
        for i in ['batch','epochs','data','cfg','weight','name']:
            pre = ' --'
            back = ' '
            opt = opt + pre + i + back + options[i]
        print(opt+' --device '+device)
        print('--------------------------------')


def get_newest_yolopt_path(train_name, yolo_path):
    path = yolo_path + "/runs/train"
    train_folder = os.listdir(path)
    target = train_name
    target_folder = []
    for i in train_folder:
        if target in i:
            target_folder.append(i)

    target_folder
    max_opt = 0
    for i in target_folder:
        a = os.path.getmtime(path+"/"+i)
        if a > max_opt:
            max_target = i
    result = path + "/" + max_target + "/weights/best.pt"

    return result

def pred_to_label_txt(folder, cv_img, pred, class_num, file_name):
    ''' 과거 코드용임 '''
    print('pred_to_label 를 사용해주세요 코드 변경됬습니다. 조만간 삭제됩니다.')
    img_h, img_w, _c = cv_img.shape
    write_labels =[]
    for i in pred:
        lu_x = int(i[0]) / img_w
        lu_y = int(i[1]) / img_h
        rd_x = int(i[2]) / img_w
        rd_y = int(i[3]) / img_h
        midx = str(round((lu_x + rd_x)/2,6))
        midy = str(round((lu_y + rd_y)/2,6))
        pred_w = str(round((rd_x - lu_x),6))
        pred_h = str(round((rd_y - lu_y),6))
        cls = str(class_num)
        label = cls + " " + midx + " " + midy + " " + pred_w + " " + pred_h
        write_labels.append(label)

        with open(label_path + "/" + img_name+".txt", 'w' )as f:
            for k in write_labels:
                f.write(k+"\n")


def pred_to_label_txt3(folder, cv_img, pred, class_num, base_name):
    ''' 과거 코드용임 '''
    print('pred_to_label 를 사용해주세요 코드 변경됬습니다. 조만간 삭제됩니다.')
    img_h, img_w, _c = cv_img.shape
    write_labels = []
    for i in pred:
        lu_x = int(i[0]) / img_w
        lu_y = int(i[1]) / img_h
        rd_x = int(i[2]) / img_w
        rd_y = int(i[3]) / img_h
        midx = str(round((lu_x + rd_x)/2, 6))
        midy = str(round((lu_y + rd_y)/2, 6))
        pred_w = str(round((rd_x - lu_x), 6))
        pred_h = str(round((rd_y - lu_y), 6))
        cls = str(class_num)
        label = cls + " " + midx + " " + midy + " " + pred_w + " " + pred_h
        write_labels.append(label)

    save_path = os.path.join(folder, base_name[:-4]+'.txt')
    
    with open(save_path, 'w')as f:
        for k in write_labels:
            f.write(k+"\n")


def pred_to_label_stable(save_file_name : str , cv_img , pred, class_num = False):
    ''' 과거 코드용임 '''
    print('pred_to_label 를 사용해주세요 코드 변경됬습니다. 조만간 삭제됩니다.')
    img_h, img_w, _c = cv_img.shape
    write_labels =[]
    for i in pred:
        lu_x = int(i[0]) / img_w
        lu_y = int(i[1]) / img_h
        rd_x = int(i[2]) / img_w
        rd_y = int(i[3]) / img_h
        midx = str(round((lu_x + rd_x)/2,6))
        midy = str(round((lu_y + rd_y)/2,6))
        pred_w = str(round((rd_x - lu_x),6))
        pred_h = str(round((rd_y - lu_y),6))
        if class_num == False:
            cls = str(int(i[5]))
        else:
            cls = str(class_num)
        label = cls + " " + midx + " " + midy + " " + pred_w + " " + pred_h
        write_labels.append(label)

        with open(save_file_name, 'w' )as f:
            for k in write_labels:
                f.write(k+"\n")