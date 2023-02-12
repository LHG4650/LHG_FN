import time
import os
import logging


def time_stamp():
    time_stamp = time.strftime("%y.%m.%d_%H.%M.%S", time.localtime())
    return time_stamp


class HG_Progress:
    """
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

    """

    def __init__(self, total_repeat_count):
        self.start_time = time.time()
        self.t_0 = time.time()
        self.t_1 = time.time()
        self.total_repeat_count = total_repeat_count
        self.idx = 0

    def __call__(self):
        t_1 = time.time()
        self.idx += 1
        elapse_time = t_1 - self.start_time  # 경과시간
        a_progress_time = elapse_time / self.idx  # 개당 걸리는 시간
        remain_time = a_progress_time * (self.total_repeat_count - self.idx)  # 남은 시간
        # pred_time = (t_1 + remain_time)+60*60*9 #종료 예상시간
        pred_time = (
            self.total_repeat_count * (elapse_time) / self.idx
            + self.start_time
            + 1 * 60 * 60 * 9
        )  # 종료 예상시간

        progress_txt = (
            " 진행율 " + str(round(self.idx / self.total_repeat_count * 100, 2)) + "% /"
        )
        elapse_time_txt = (
            " 경과 시간 " + time.strftime("%H:%M:%S", time.gmtime(elapse_time)) + " /"
        )
        remain_time_txt = (
            " 잔여 시간 " + time.strftime("%H:%M:%S", time.gmtime(remain_time)) + " /"
        )
        pred_time_txt = (
            " 종료 예상 " + time.strftime("%H:%M:%S", time.gmtime(pred_time)) + " /"
        )

        txt = "\r" + progress_txt + elapse_time_txt + remain_time_txt + pred_time_txt
        print(txt, end="")


def make_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: creationg direictory.  " + path)


class Txt_saver:
    """
    텍스트 한줄씩 써가는 클래스
    선언시 원하는 파일주소를 입력하면 없으면 만들고 있으면 사용함.
    호출마다 호출에 넣어주는 텍스트를 파일 하단에 한줄 추가해줌
    ex...
    T_S = Txt_saver(txt_filepath):
    T_S('txt')
    """

    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.file_satus = os.path.isfile(file_path)
        if not (self.file_satus):
            self.create(file_path)
        self.open_file()

    def create(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.f.close()

    def open_file(self):
        self.f = open(self.file_path, "a", encoding="utf-8")

    def __call__(self, txt):
        self.open_file()
        self.f.write(txt + "\n")
        self.f.close()


def GetPrinterList():
    import win32print

    printers = win32print.EnumPrinters(2)
    printers_name_list = []
    for i in printers:
        printers_name_list.append(i[2])
    return printers_name_list


def get_printer_job(printer_name):
    import win32print

    """ printer 에 잡혀있는 Queue 리스트를 반환함 [ 프린터 대기열 페이지 반환함 ]
        없으면 []을 반환함 """

    """
    JOB_STATUS_COMPLETE는 프린터에 출력된 문서를 완료했을 때 발생하는 상태를 의미합니다. 

JOB_STATUS_DELETED는 프린터 대기열에 있는 문서가 삭제되었을 때 발생하는 상태를 의미합니다. 

JOB_STATUS_DELETING는 프린터 대기열에 존재하는 문서가 삭제되는 중일 때 발생하는 상태를 의미합니다. 

JOB_STATUS_ERROR는 프린터에 문제가 발생하여 출력이 실패하는 경우 발생하는 상태를 의미합니다. 

JOB_STATUS_OFFLINE는 프린터가 오프라인 상태일 때 발생하는 상태를 의미합니다. 

JOB_STATUS_PAPEROUT는 프린트할 용지가 없을 때 발생하는 상태를 의미합니다. 

JOB_STATUS_PAUSED는 프린트 작업이 일시 중단되었을 때 발생하는 상태를 의미합니다. 

JOB_STATUS_PRINTED는 프린트 작업이 완료되었을 때 발생하는 상태를 의미합니다. 

JOB_STATUS_PRINTING은 프린트 작업이 진행 중일 때 발생하는 상태를 의미합니다. 

JOB_STATUS_RESTART는 프린트 작업이 재시작되었을 때 발생하는 상태를 의미합니다. 

JOB_STATUS_SPOOLING은 프린트 작업이 스풀링 중일 때 발생하는 상태를 의미합니다. 

JOB_STATUS_USER_INTERVENTION은 사용자의 조치가 필요한 경우 발생하는 상태를 의미합니다.
"""

    job_list = []
    for p in win32print.EnumPrinters(win32print.PRINTER_ENUM_LOCAL, None, 1):
        flags, desc, name, comment = p
        if name == printer_name:
            phandle = win32print.OpenPrinter(name)
            print_jobs = win32print.EnumJobs(phandle, 0, -1, 1)
            if print_jobs:
                for job in print_jobs:
                    document = job["pDocument"]
                    page_count = job["TotalPages"]
                    submit_time = job["Submitted"]
                    submit_time = submit_time.Format("%Y-%m-%d %H:%M:%S")
                    status = job["Status"]
                    job_list.append([document, page_count, submit_time, status])

            win32print.ClosePrinter(phandle)
    return job_list


def check_status():
    import torch
    import os

    print(os.popen("nvcc --version").read())
    print("torch_version : ", torch.__version__)
    # python -m pip install --upgrade pip ## pip upgrade
    # cuda == 11.1 << 상대면?
    # conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge
    # https://velog.io/@openjr/Yolo-v5-ROS%EC%97%90-%EC%98%AC%EB%A6%AC%EA%B8%B0  #upsampling 문제 해결
    """
    import torch.nn.modules.upsampling
    def forward(self, input: Tensor) -> Tensor:
        # return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,
        #                      recompute_scale_factor=self.recompute_scale_factor)
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
        """


def DeleteAllFile(folder_path):
    if os.path.exists(folder_path):
        for file in os.scandir(folder_path):
            os.remove(file.path)


def set_logger(project_name, Test_mode=False, log_level=logging.INFO):
    make_dir("logs")

    if Test_mode:
        log_file_path = os.path.join("logs", "TEST" + ".log")
    else:
        log_file_path = os.path.join("logs", project_name + ".log")

    logger = logging.getLogger(project_name)
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

    # StreamHandler
    streamingHandler = logging.StreamHandler()
    streamingHandler.setFormatter(formatter)

    # FileHandler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(streamingHandler)
    logger.addHandler(file_handler)
    return logger, log_file_path
