import os
import PIL
import timm  # noqa
import torch
import numpy as np
import pandas as pd
import torchvision.models as models  # noqa
import cv2
from torchvision import transforms
from patchcore.utils import set_torch_device, fix_seeds
import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler
import patchcore.patchcore
import contextlib
import faiss

# https://jaeniworld.tistory.com/8 OMP pyplot충돌애러
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#######

_BACKBONES = {
    "alexnet": "models.alexnet(pretrained=True)",
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet50": "models.resnet50(pretrained=True)",
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
}

def load(name):
    return eval(_BACKBONES[name])

def create_storage_folder(project_folder , object_name, mode="iterate"):
    ''' 학습 결과물을 저장하는 폴더를 만들어줌 폴더구조는 다음과 같음
    | - project folder
        | - custom
            | - hg_patchcore_model
                | - object_name(_iterate)

    mode 를 iterate로 두면 돌릴때마다 _0부터 카운트를 늘려가며 폴더가 생성됨
    mode가 overwrite 이면 object_name으로 끝남  '''

    os.makedirs(project_folder, exist_ok=True)
    patchcore_folder = os.path.join(project_folder,'custom','hg_patchcore_model')
    os.makedirs(patchcore_folder, exist_ok=True)

    save_path = patchcore_folder
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(patchcore_folder, object_name + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path

def get_sampler(device, percentage, sampler_name):
    '''
        Args:
            device: [int] int of gpu ids. If None, auto find cuda device or cpu, 
                     it is used.
            percentage: [float] patch sample 가져갈 퍼센트 넣으세요 디폴트는 후에 정할것
            sampler_name: [str] 샘플러 종류를 넣으세요 ['identity','greedy_coreset','approx_greedy_coreset']
        
    '''
    if sampler_name == "identity":
        return patchcore.sampler.IdentitySampler()
    elif sampler_name == "greedy_coreset":
        return patchcore.sampler.GreedyCoresetSampler(percentage, device)
    elif sampler_name == "approx_greedy_coreset":
        return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

def get_patchcore(input_shape, sampler, device, backbone:dict = None ,faiss_on_gpu:bool = False,faiss_num_workers:int = 4,
                  pretrain_embed_dimension:int = 1024, target_embed_dimension:int = 384, patchsize:int = 3,
                  anomaly_scorer_num_nn:int = 3):
    '''
        Args:
            backbone: dict 형식으로 들어오길 바람 모델명과 넣고싶은 레이어들의 리스트를 담았으면 함
            ex defalt_backbon_dict = {'wideresnet50' : ['layer2','layer3']}
            faiss_on_gpu: faiss를 gpu로 쓸꺼면 True
                          If set true, nearest neighbour searches are done on GPU.
            faiss_num_workers: Number of workers to use with FAISS for similarity search.
            anomaly_scorer_num_nn: anomaly knn 할 이웃의 수

        Returns:
            loaded_patchcores: [list] 로 내부에 패치코어 클래스가 들어있음. 각 패치코어는 backbon에 들어간 모델의 갯수와 같음
    '''
    layers_to_extract_from_coll = []
    if backbone == None:
        backbone = {'wideresnet50' : ['layer2','layer3']}   #defalt backbone
    backbone_name = list(backbone.keys())

    if backbone_name.__len__() > 1:
        for i in backbone_name:
            layers_to_extract_from_coll.append(backbone[i])
    else: 
        layers_to_extract_from_coll = [backbone[backbone_name[0]]]
    
    # print(f'backbone_name:{backbone_name}')
    # print(f'layers_to_extract_from_coll:{layers_to_extract_from_coll}')

    loaded_patchcores = []
    for backbone_name, layers_to_extract_from in zip(backbone_name, layers_to_extract_from_coll):
        backbone_seed = None

        backbone = patchcore.backbones.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed

        nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

        patchcore_instance = patchcore.patchcore.PatchCore(device)
        # print(f'backbone:{backbone}')
        # print(f'layers_to_extract_from:{layers_to_extract_from}')
        # print(f'device:{device}')
        # print(f'input_shape:{input_shape}')
        # print(f'pretrain_embed_dimension:{pretrain_embed_dimension}')
        # print(f'target_embed_dimension:{target_embed_dimension}')
        # print(f'patchsize:{patchsize}')
        # print(f'sampler:{sampler}')
        # print(f'anomaly_scorer_num_nn:{anomaly_scorer_num_nn}')
        # print(f'nn_method:{nn_method}')

        
        patchcore_instance.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=input_shape,
            pretrain_embed_dimension=pretrain_embed_dimension,
            target_embed_dimension=target_embed_dimension,
            patchsize=patchsize,
            featuresampler=sampler,
            anomaly_scorer_num_nn=anomaly_scorer_num_nn,
            nn_method=nn_method,
        )
        loaded_patchcores.append(patchcore_instance)
    return loaded_patchcores  

def check_faiss_gpu():
    if faiss.get_num_gpus() > 0:
        return True
    else:
        return False
#######

class HgPatchCore:
    def __init__(self,project_folder, object_name):
        self.project_folder = project_folder
        self.object_name = object_name
        self.patchcore = None
        self.transform_img = None
        self.now_model_folder_path = None
        self.test_data_set = None
        self.auto_thres_state = None
        self.a_map_mean = None
        self.a_map_std = None
        self.a_map_thred= None
        pass

    def fit(self, batch_size = 'defalt', device = None, seed = 0, sampler_name = 'approx_greedy_coreset', num_workers:int = 2,
            save_mode:str = 'iterate',
            sampler_persentage:float = 0.1,
            backbone_dic:dict = None, faiss_on_gpu:bool = None ,faiss_num_workers:int = 4, pretrain_embed_dimension:int = 1024,
            target_embed_dimension:int = 384, patchsize:int = 3, anomaly_scorer_num_nn:int= 3,report:bool = True):
        '''
        Args:
            device: [int] int of gpu ids. If None, auto find cuda device or cpu, 
                     it is used.
            save_mode: [str] iterate로 두면 돌릴때마다 _0부터 카운트를 늘려가며 폴더가 생성됨
                             overwrite 이면 object_name으로 끝남
        '''
        if faiss_on_gpu is None:
            faiss_on_gpu = self.faiss_on_gpu


        if backbone_dic == None:
            backbone_dic = {'wideresnet50' : ['layer2','layer3']}   #defalt backbone

        self.set_device(device)

        device_context = (
            torch.cuda.device("cuda:{}".format(self.device.index))
            if "cuda" in self.device.type.lower()
            else contextlib.suppress()
            )
        fix_seeds(seed,self.device)

        dataloader = torch.utils.data.DataLoader(self.train_data_set,
                                                 batch_size=batch_size,
                                                 shuffle = False,
                                                 num_workers = num_workers,
                                                 pin_memory = True
                                                 )

        with device_context:
            torch.cuda.empty_cache()
            image_size = self.train_data_set.img_size
            sampler = get_sampler(self.device,sampler_persentage,sampler_name)
            PatchCore_list = get_patchcore(image_size, sampler, self.device, 
                                           backbone=backbone_dic,
                                           faiss_on_gpu = faiss_on_gpu, 
                                           faiss_num_workers=faiss_num_workers, 
                                           pretrain_embed_dimension=pretrain_embed_dimension,
                                           target_embed_dimension=target_embed_dimension,
                                           patchsize=patchsize,
                                           anomaly_scorer_num_nn=anomaly_scorer_num_nn)
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                PatchCore.fit(dataloader)

        save_path = create_storage_folder(self.project_folder, self.object_name, mode=save_mode)
        for i, PactchCore in enumerate(PatchCore_list):
            prepend = (
                "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                if len(PatchCore_list) > 1
                else "")
            PatchCore.save_to_path(save_path, prepend)

        #config_save
        train_config = {}
        train_config['batch_size']=[batch_size]
        train_config['self.device']=[self.device]
        train_config['device']=[device]
        train_config['seed']=[seed]
        train_config['sampler_name']=[sampler_name]
        train_config['sampler_persentage']=[sampler_persentage]
        train_config['num_workers']=[num_workers]
        train_config['save_mode']=[save_mode]
        train_config['image_size']=[str(image_size)]
        train_config['self.project_folder']=[self.project_folder]
        train_config['self.object_name']=[self.object_name]
        train_config['backbone_dic']=[backbone_dic]
        train_config['faiss_on_gpu']=[faiss_on_gpu]
        train_config['faiss_num_workers']=[faiss_num_workers]
        train_config['pretrain_embed_dimension']=[pretrain_embed_dimension]
        train_config['target_embed_dimension']=[target_embed_dimension]
        train_config['patchsize']=[patchsize]
        train_config['anomaly_scorer_num_nn']=[anomaly_scorer_num_nn]

        dataset_config_dic = self.train_data_set.get_transform_img_config()
        train_config['resize'] = [dataset_config_dic['resize']]
        train_config['IMAGE_MEAN'] = [dataset_config_dic['IMAGE_MEAN']]
        train_config['IMAGE_STD'] = [dataset_config_dic['IMAGE_STD']]



        config_pd = pd.DataFrame(train_config).transpose()
        config_pd.index.name = 'config'
        #config_pd.columns = 'value'
        config_save_path = os.path.join(save_path,'train_config.csv')
        config_pd.to_csv(config_save_path,encoding='cp949')

        self.patchcore = PatchCore
        
        ### set img transform 
        t_resize = dataset_config_dic['resize']
        t_img_size = image_size[1:]
        t_mean = dataset_config_dic['IMAGE_MEAN']
        t_std = dataset_config_dic['IMAGE_STD']

        self.set_img_transform_config(t_resize, t_img_size, t_mean, t_std)
        if report:
            self.get_report()
        pass

    def get_report(self, score_thr_sigma:float = 3,a_map_thr_sigma:float=12):
        assert not(self.test_data_set is None), '데이터를 등록하세요 .set_dataset(path)'    
        import scipy.stats as stats
        import matplotlib.pyplot as plt
        import copy
        self.set_model_folder_path()
        def get_norm_line(data:np.array):
            mu = data.mean()
            sigma = data.std()
            x = np.linspace(data.min(),data.max(),100)
            y = stats.norm(mu,sigma).pdf(x)
            return mu,sigma,x,y
        def get_cdf_in_norm_dist(mean,std,thres):
            norm_dist = stats.norm(mean, std)
            cumulative_prob = norm_dist.cdf(thres)
            return cumulative_prob
        def get_bins(data:np.array):
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            h = 2 * iqr / (len(data) ** (1/3))
            bins = np.arange(data.min(), data.max() + h, h)
            return bins

   

        hist_alpha = 0.5
        dist_alpha = 0.7
        line_alpha = 1


        dataset = self.test_data_set
        pc = self.patchcore
        data_form = {'score':[],'a_map':[],'img_path':[]}
        
        result = {'test_good':{'score':[],'a_map':[],'img_path':[]}, 
                  'bad':{'score':[],'a_map':[],'img_path':[]}}
        for i in range(dataset.__len__()):
            item = dataset.__getitem__(i)
            score,a_map = pc.predict(item['image'].unsqueeze(0))
            score = score[0]
            a_map = a_map[0]

            result[item['anomaly']]['score'].append(score)
            result[item['anomaly']]['a_map'].append(a_map)
            result[item['anomaly']]['img_path'].append(item['image_path'])

        #### make anoamly score hist, norm dist, 6sig threshold
        good_score = np.array(result['test_good']['score'])
        
        bad_score = np.array(result['bad']['score'])
        
        plt.figure(dpi=300)
        bins = get_bins(good_score)
        mu,sigma,x,y = get_norm_line(good_score)
        thres_val = mu+score_thr_sigma*sigma
        alpha = 1-get_cdf_in_norm_dist(mu,sigma,thres_val)
        plt.hist(good_score, alpha=hist_alpha, label='good',color='g',bins = bins) #hist
        plt.plot(x, y, linewidth=1, color='g',alpha=dist_alpha)        #norm
        plt.axvline(mu, color='g', linestyle='--', alpha=line_alpha)              #mean line
        plt.axvline(thres_val, color='black', linestyle='-',label = f'Thres_hold/{score_thr_sigma}sigma : {round(thres_val,2)}')  #trheshold line

        bins = get_bins(bad_score)
        mu,sigma,x,y = get_norm_line(bad_score)
        beta = get_cdf_in_norm_dist(mu,sigma,thres_val)
        plt.hist(bad_score, alpha=hist_alpha, label='bad',color='r',bins = bins) #hist
        plt.plot(x, y, linewidth=1, color='r',alpha=dist_alpha)        #norm
        plt.axvline(mu, color='r', linestyle='--', alpha=line_alpha)              #mean line
        save_path = os.path.join(self.now_model_folder_path,'score_fig.png')
        plt.title('anomaly score report')
        # plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.xlabel(f'score / alpha : {round(alpha*100,2)}%, beta : {round(beta*100,2)}%')
       
        plt.legend(loc='upper right')
        plt.savefig(save_path)
        plt.show()

        ######## hitmap ########
        flattened_data = np.concatenate(result['test_good']['a_map']).ravel()
        f_mean = flattened_data.mean()
        f_std = flattened_data.std()
        bad_flattened_data = np.concatenate(result['bad']['a_map']).ravel()
        top_ten = np.quantile(flattened_data,0.98) #상위 2프로 값
        b_z_min = (top_ten-f_mean)/f_std/2     #젤큰것의 정규화한것의 절반
        #b_z_min = (flattened_data.max()-f_mean)/f_std/2     #젤큰것의 정규화한것의 절반
        b_z_max = b_z_min + a_map_thr_sigma*f_std           #min에서 시그마_val * 표준편차
        #                      세로수, 가로수   가로넓이, 세로넓이
        fig, axs = plt.subplots(5,4,figsize = (20,25)) #세로 5 가로 4
        fig.subplots_adjust(left = 0.1, right = 0.9,wspace=0.05, hspace=0.05) # 여백 최소화
        #정상 히트맵, 원본이미지 불량히트맵 원본이미지
        for i in range(5):
            good_item = result['test_good']
            a_map = good_item['a_map'][i]
            z_map = (a_map-f_mean)/f_std
            axs[i,0].imshow(z_map,cmap='jet', vmin=b_z_min, vmax=b_z_max)
            img = cv2.imread(good_item['img_path'][i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i,1].imshow(img)

            bad_item = result['bad']
            ba_map = bad_item['a_map'][i]
            bz_map = (ba_map-f_mean)/f_std
            axs[i,2].imshow(bz_map,cmap='jet', vmin=b_z_min, vmax=b_z_max)
            bimg = cv2.imread(bad_item['img_path'][i])
            bimg = cv2.cvtColor(bimg, cv2.COLOR_BGR2RGB)
            axs[i,3].imshow(bimg)

        # 각 subplot의 축 숨기기
        for ax in axs.flatten():
            ax.axis('off')
        
        save_path = os.path.join(self.now_model_folder_path,'test_result.png')
        plt.savefig(save_path)
        plt.show()

        #auto_thres
        auto_thres_config = {}

        mu,sigma,x,y = get_norm_line(good_score)
        thres_val = mu+score_thr_sigma*sigma
        auto_thres_config['score_thres']=[thres_val]
        auto_thres_config['a_map_mean']=[f_mean]
        auto_thres_config['a_map_std']=[f_std]
        auto_thres_config['a_map_thred']=[b_z_min]

        config_pd = pd.DataFrame(auto_thres_config).transpose()
        config_pd.index.name = 'config'
        config_save_path = os.path.join(self.now_model_folder_path,'auto_thres_config.csv')
        config_pd.to_csv(config_save_path,encoding='cp949')

        pass
    
    def set_model_folder_path(self,folder_name:str = None):
        latest_time = 0
        if folder_name is None:
            model_folder_path = os.path.join(self.project_folder,'custom','hg_patchcore_model')
            for folder in os.listdir(model_folder_path):
                if self.object_name in folder:
                    if os.path.isdir(os.path.join(model_folder_path, folder)):
                        modified_time = os.path.getmtime(os.path.join(model_folder_path, folder))
                        if modified_time > latest_time:
                            latest_time = modified_time
                            latest_folder = folder
            model_folder_path = os.path.join(model_folder_path,latest_folder)                        
        else:
            model_folder_path = folder_name
        self.now_model_folder_path = model_folder_path

    def load(self,model_folder:str = None, faiss_num_workers:int = None, auto_thres:bool = True):
        self.set_model_folder_path(folder_name = model_folder)


        if faiss_num_workers is None:
            faiss_num_workers = 4
        nn_method = patchcore.common.FaissNN(self.faiss_on_gpu,faiss_num_workers)
        self.patchcore = patchcore.patchcore.PatchCore(self.device)
        self.patchcore.load_from_path(self.now_model_folder_path,self.device,nn_method)

        config_file_path = os.path.join(self.now_model_folder_path,'train_config.csv')
        config = pd.read_csv(config_file_path,encoding='cp949').set_index('config',drop=True)

        ### set img transform 
        t_resize = int(config.loc['resize','0'])
        t_img_size = eval(config.loc['image_size','0'])[1:]
        t_mean = np.fromstring(config.loc['IMAGE_MEAN','0'][1:-1], sep=' ')
        t_std = np.fromstring(config.loc['IMAGE_STD','0'][1:-1], sep=' ')

        self.set_img_transform_config(t_resize, t_img_size, t_mean, t_std)

        ### set auto thres config 
        config_file_path = os.path.join(self.now_model_folder_path,'auto_thres_config.csv')
        if os.path.exists(config_file_path):
            config = pd.read_csv(config_file_path,encoding='cp949').set_index('config',drop=True)
            pass
        else:
            print('auto_thres 파일 없어 setting 불가 get_report() 필요')
            try:
                self.get_report()
                config = pd.read_csv(config_file_path,encoding='cp949').set_index('config',drop=True)
            except:
                auto_thres = False

        if auto_thres:
            self.auto_thres_state = True
            self.a_map_mean = float(config.loc['a_map_mean','0'])
            self.a_map_std = float(config.loc['a_map_std','0'])
            self.a_map_thred = float(config.loc['a_map_thred','0'])


    def set_img_transform_config(self,resize,img_size,mean,std):

        img_transform_step = [
            transforms.Resize(resize),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
           transforms.Normalize(mean=mean, std=std),
        ]
        
        self.transform_img = transforms.Compose(img_transform_step)
    
    def __call__(self, img, img_type:str = 'cv2'):
        if img_type == 'cv2':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)

        img = self.transform_img(img)
        img = img.unsqueeze(0)
        anomaly_score, anomaly_map = self.patchcore.predict(img)
        anomaly_score = anomaly_score[0]
        anomaly_map = anomaly_map[0]
        if self.auto_thres_state:
            anomaly_map = (anomaly_map-self.a_map_mean)/self.a_map_std

        return anomaly_score, anomaly_map
        

    def set_dataset(self, data_folder):
        self.train_data_set = HgPatchCore_DataSet(data_folder, self.object_name,mode = 'train')
        self.test_data_set = HgPatchCore_DataSet(data_folder, self.object_name,mode = 'test')
        pass

    def set_device(self, device_num = None):
        """Returns correct torch.device.

        Args:
            gpu_ids: [int] int of gpu ids. If None, auto find cuda device or cpu, 
                     it is used.
        """
        if device_num == None:
            device_num = torch.cuda.device_count()
        device = list(range(device_num))
        self.device = set_torch_device(device)
        self.faiss_on_gpu = check_faiss_gpu()


class HgPatchCore_DataSet(torch.utils.data.Dataset):
    def __init__(self,data_folder,object_name,
                 resize='defalt', img_size='defalt', mode = 'train', **kwargs):
        '''
        patchcore dataset의 데이터 구조는 다음과 같다
        | - data_folder
            | - object_name
                | - good    (학습할 정상 이미지)
                | - bad     (검증에 사용될 bad 이미지)
                | - test_good   (검증에 사용될 test이미지 (학습되지 않음))
                | - meta    (메타 데이터가 저장될 폴더)
                    | - mean_N_std.csv (good 이미지의 RGB mean과 std가 저장됨)

        mode = train 이면 dataset은 good img만 사용한다
        mode = test 이면 dataset은 good, bad, test_good을 모두 사용한다
        resize : 인풋 이미지 사이즈 변경시킴 (resize,resize,3)
        img_size : 리사이즈된 이미지를 img_size크기로 센터크롭함 (img_size,img_size,3)
        # 리사이즈 크기가 이미지 사이즈보다 작으면 에러 발생 예상함
        # 기본 모듈 사이즈값 resize='256', img_size=224
        32의 배수가 좋을듯 한데?? 

        '''

        super().__init__()
        self.data_folder = data_folder
        self.object_name = object_name
        self.img_folder = os.path.join(data_folder,object_name)

        if (resize == 'defalt') | (img_size == 'defalt'):   #defalt가 있으면
            min_img_size = self.get_img_size()              #이미지 사이즈중 가로세로중 작은값 가져와서
            if resize == 'defalt':
                resize = min_img_size                       #리사이즈가 기본이면 작은값 사용
            if img_size == 'defalt':        
                if resize < min_img_size:                   # 이미지 사이즈가 기본이면 
                    img_size = resize                       # 이미지 사이즈와 민사이즈중 작은걸 사용
                else:
                    img_size = min_img_size                 #효과 디폴트로 두면 이미지 사이즈에 똑같이 사용됨

        self.resize = resize
        self.img_size = img_size
        self.mode = mode

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.get_img_mean_std()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
           transforms.Normalize(mean=self.IMAGE_MEAN, std=self.IMAGE_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)
        self.img_size = (3, img_size, img_size)

    def get_img_mean_std(self):
        mean_std_path = os.path.join(self.img_folder,'meta', "mean_N_std.csv")
        if os.path.isfile(mean_std_path):
            print('있음')
            info = pd.read_csv(mean_std_path,index_col=0)
            self.IMAGE_MEAN = info.loc['mean'].values     
            self.IMAGE_STD = info.loc['std'].values
        else: 
            os.makedirs(os.path.join(self.img_folder,'meta'),exist_ok=True)
            first_transform_img = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor()]
            first_transform_img = transforms.Compose(first_transform_img)
            
            inf = {}
            rgb = ['r','g','b']
            ms = ['mean','std']
            for i in ms:
                inf[i] = {}
                for j in rgb:
                    inf[i][j]=[]

            for img_path in self.imgpaths_per_class[self.object_name]['good']:
                image = PIL.Image.open(img_path).convert("RGB")
                image = first_transform_img(image)

                inf['mean']['r'].append( image[0].mean())
                inf['mean']['g'].append( image[1].mean())
                inf['mean']['b'].append( image[2].mean())
                inf['std']['r'] .append( image[0].std() )
                inf['std']['g'] .append( image[1].std() )
                inf['std']['b'] .append( image[2].std() )

            result = []
            for i in ms:
                dumi = [] 
                for j in rgb:
                    dumi.append(np.mean(inf[i][j]))
                result.append(dumi)
            
            result = pd.DataFrame(result)
            result.columns = rgb
            result.index = ms
            result.index.name = 'idx'
            result.to_csv(mean_std_path)
            self.get_img_mean_std()

    def get_image_data(self):
        
        good_data_folder = os.path.join(self.img_folder, "good")
        bad_data_folder = os.path.join(self.img_folder, "bad")
        test_good_data_folder = os.path.join(self.img_folder, "test_good")
        os.makedirs(good_data_folder, exist_ok=True)
        os.makedirs(bad_data_folder, exist_ok=True)
        os.makedirs(test_good_data_folder, exist_ok=True)

        imgpaths_per_class = {}
        imgpaths_per_class[self.object_name] = {}
        anomaly_files = sorted(os.listdir(good_data_folder))
        
        if self.mode == 'test':
            anomaly_files = sorted(os.listdir(bad_data_folder))
            imgpaths_per_class[self.object_name]['bad'] = [
                    os.path.join(bad_data_folder, x) for x in anomaly_files]
            anomaly_files = sorted(os.listdir(test_good_data_folder))
            imgpaths_per_class[self.object_name]['test_good'] = [
                    os.path.join(test_good_data_folder, x) for x in anomaly_files]
        else:
            imgpaths_per_class[self.object_name]['good'] = [
                    os.path.join(good_data_folder, x) for x in anomaly_files]

        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate

    def get_img_size(self):
        img_folder = os.path.join(self.img_folder,'good')
        img_list = os.listdir(img_folder)
        for file_name in img_list:
            if '.jpg' in file_name:
                img_path = os.path.join(img_folder, file_name)
                img = cv2.imread(img_path)
                img_size = min(img.shape[:2])
        return img_size

    def __getitem__(self,idx):
        classname, anomaly, image_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        return {
            "image": image,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }
        
    def __len__(self):
        return len(self.data_to_iterate)

    def get_transform_img_config(self):
        img_tf_dic = {}
        img_tf_dic['resize'] = self.resize
        img_tf_dic['img_size'] = self.img_size
        img_tf_dic['IMAGE_MEAN'] = self.IMAGE_MEAN
        img_tf_dic['IMAGE_STD'] = self.IMAGE_STD
        return img_tf_dic
        



_BACKBONES = {
    "alexnet": "models.alexnet(pretrained=True)",
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet50": "models.resnet50(pretrained=True)",
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
}


def load(name):
    return eval(_BACKBONES[name])
