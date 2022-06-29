import os



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
    

    train_list = os.listdir(folder +"/train_images")
    train_img_list = []  
    for i in train_list:
        train_img_list.append(folder +"/images/"+i)

    val_list = os.listdir(folder +"/val_images")
    val_img_list = []  
    for i in val_list:
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