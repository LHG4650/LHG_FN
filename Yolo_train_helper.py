import os
from LHG_FN import utill

def yaml_make(folder):
    '''
    yolo 학슴을 위한 yaml. test. val 파일을 만드는 함수이다.

    아래 네개의 변수 입력해주고 실행시키면 된다.
    목표 폴더에는 imgaes, labels 가 존재하여야 한다.
    target_dir      : 폴더명
    file_formet     : 이미지 확장자명
    dataset_dir     : 폴더가 있는곳 까지의 경로
    class_list      : 라벨된 이미지의 class, 리스트로 나타내야 한다.
    '''

    #dataset_dir ="C:/Users/gusrm/Desktop/porg/FBF8_IC_porridgevision/dataset"
    #target_dir = "yolo_spoonNchamoil_prelab"
    #train_dir = "train_images"
    #val_dir = "val_images"
    #file_formet = "jpg"
    #class_list2 = ['empty','spoon_in','spoon_out','sesame_out','empty_red','soup_out']
    utill.make_dir(folder)
    utill.make_dir(folder+"/train_images")
    utill.make_dir(folder+"/images")
    utill.make_dir(folder+"/val_images")
    utill.make_dir(folder+"/labels")
    

    train_list = os.listdir(folder +"/train_images")
    train_img_list = []  
    for i in train_list:
        #print(i)
        train_img_list.append(folder +"/images/"+i)

    val_list = os.listdir(folder +"/val_images")
    val_img_list = []  
    for i in val_list:
        #print(i)
        val_img_list.append(folder +"/images/"+i)

    print('data 총량 - ',len(train_img_list)+len(val_img_list))
    print('train 총량 - ',len(train_img_list),'test 총량 - ', len(val_img_list))

    class_list = []
    with open(folder +"/labels/classes.txt", 'r' )as f:
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

    with open(folder+"/train.txt", 'w' )as f:
        f.write('\n'.join(train_img_list)+'\n')
    with open(folder+"/val.txt", 'w' )as f:
        f.write('\n'.join(val_img_list)+'\n')
    with open(folder+"/data.yaml", 'w' )as f:
        f.write('train: '+folder+"/train.txt"+'\n')
        f.write('val: '+folder+"/val.txt"+'\n'+'\n')
        f.write(w_line1+'\n')
        f.write(w_line2+'\n')

    print('cls list =',class_list)

def yolo_train_commend(data_folder,yolo_folder,yolo_model_select ,epochs=50,weight=1, name ='noname', auto = False, batch = 4):
    '''
    data_folder : C:/Users/gusrm/Desktop/porg/FBF8_IC_porridgevision/dataset/train_dataset_v2
    yolo_folder : C:/Users/gusrm/Desktop/porg/yolov5
    folder 학습이라고 만들어둔 폴더 해당폴더 내에는
        result pt파일 결과물 저장하는 폴더가 있어야한다.
        data.yaml 트레인파일 물론 사전에 yaml_make() 를 실행시켜야한다

    epochs 반복횟수
    weight 사전 가중치로 사용할 파일 
        1 = 욜로 v5기본 pt
        2 = 같은 네임으로 저장된 pt중 가장 최신
        esle = 따로 선택한 pt파일
    name    #욜로 학습 이름
    auto    #자동으로 학습까지 할껀지 물어봄
    '''
    yolo_train_path = yolo_folder + '/train.py' #상수 변하지않음.
    
    opt_data = data_folder +"/data.yaml"
    if weight ==1:  #기본값
        opt_weight = yolo_folder + "/yolov5s.pt"
    elif weight == 2:       #같은 이름이 포함된 가장 최신의 pt
        opt_weight = get_newest_yolopt_path(name, yolo_folder)
    else:                   #셀프 pt경로 추가
        opt_weight = weight

    assets = ['n', 's', 'm', 'l', 'x']
    if yolo_model_select in assets: 
        yolo_model_select = yolo_folder + "/models/yolov5"+yolo_model_select + '.yaml'
    else:
        yolo_model_select = ""
    options =   {
                'img' : '640',      #이미지 높이
                'batch' : str(batch),      #배치 높이
                'epochs' : str(epochs),
                'data' : opt_data,
                'cfg' : yolo_model_select,
                'weight' : opt_weight,
                'name' : name,
                'workers' : '0',
            }

    opt = "python " + yolo_train_path
    for i in options.keys():
        pre = ' --'
        back = ' '
        opt = opt + pre + i + back + options[i]
    print(' ----입력값---------------------')
    print(opt)
    print('--------------------------------')
    if auto:
        os.system(opt)
        #pt_path = get_newest_yolopt_path(name)
        #new = folder + "/result/" + time_stamp() + "_" + name +".pt"
        #check_N_copy_file(pt_path,new)
    
    #print('--------33333333333333------------------------')


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

def pred_to_label_txt(img, pred, label_path, img_name):
    img_h, img_w, _c = img.shape
    write_labels =[]
    for i in pred:
        lu_x = int(i[0]) / img_w
        lu_y = int(i[1]) / img_h
        rd_x = int(i[2]) / img_w
        rd_y = int(i[3]) / img_h
        midx = str(round((lu_x + rd_x)/2,6))
        midy = str(round((lu_y + rd_y)/2,6))
        pred_w = str(round((rd_x - lu_x)/2,6))
        pred_h = str(round((rd_y - lu_y)/2,6))
        cls = str(int(i[5]))
        label = cls + " " + midx + " " + midy + " " + pred_w + " " + pred_h
        write_labels.append(label)

        with open(label_path + "/" + img_name+".txt", 'w' )as f:
            for k in write_labels:
                f.write(k+"\n")