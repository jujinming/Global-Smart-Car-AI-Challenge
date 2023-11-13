import paddle
from PIL import Image
from clip import tokenize, load_model
import glob, json, os
import cv2
from PIL import Image
from tqdm import tqdm_notebook
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import ssl

# 忽略ssl验证
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

model, transforms = load_model('ViT_B_32', pretrained=True)

# 设置图片路径和标签
img_path = "kobe.jpeg"
labels = ['kobe', 'james', 'Jordan']

# 准备输入数据
img = Image.open(img_path)
#display(img)
image = transforms(Image.open(img_path)).unsqueeze(0)
text = tokenize(labels)

# 计算特征
with paddle.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = paddle.nn.functional.softmax(logits_per_image, axis=-1)

# 打印结果
for label, prob in zip(labels, probs.squeeze()):
    print('该图片为 %s 的概率是：%.02f%%' % (label, prob*100.))

probs[0].numpy()

cn_match_words = {
    "工况描述": ["高速/城市快速路", "城区", "郊区", "隧道", "停车场", "加油站/充电站", "未知"],
    "天气": ["晴天", "雨天", "多云", "雾天", "下雪", "未知"],
    "时间": ["白天", "夜晚", "拂晓/日暮", "未知"],
    "道路结构": ["十字路口", "丁字路口", "上下匝道", "车道汇入", "进出停车场", "环岛", "正常车道", "未知"],
    "一般障碍物": ["雉桶", "水马", "碎石/石块", "井盖", "减速带", "没有"],
    "道路异常情况": ["油污/水渍", "积水", "龟裂", "起伏不平", "没有", "未知"],
    "自车行为": ["直行", "左转", "右转", "停止", "掉头", "加速", "减速", "变道", "其它"],
    "最近的交通参与者": ["行人", "小型汽车", "卡车", "交警", "没有", "未知", "其它"],
    "最近的交通参与者行为": ["直行", "左转", "右转", "停止", "掉头", "加速", "减速", "变道", "其它"],
}

en_match_words = {
"scerario" : ["suburbs","city street","expressway","tunnel","parking-lot","gas or charging stations","unknown"],
"weather" : ["clear","cloudy","raining","foggy","snowy","unknown"],
"period" : ["daytime","dawn or dusk","night","unknown"],
"road_structure" : ["normal","crossroads","T-junction","ramp","lane merging","parking lot entrance","round about","unknown"],
"general_obstacle" : ["nothing","speed bumper","traffic cone","water horse","stone","manhole cover","nothing","unknown"],
"abnormal_condition" : ["uneven","oil or water stain","standing water","cracked","nothing","unknown"],
"ego_car_behavior" : ["slow down","go straight","turn right","turn left","stop","U-turn","speed up","lane change","others"],
"closest_participants_type" : ["passenger car","bus","truck","pedestrain","policeman","nothing","others","unknown"],
"closest_participants_behavior" : ["slow down","go straight","turn right","turn left","stop","U-turn","speed up","lane change","others"],
}

cap = cv2.VideoCapture('./初赛测试视频/41.avi')
img = cap.read()[1]
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
image.resize((600, 300))

submit_json = {
    "author": "jzxd",
    "time": "231113",
    "model": "VitB/32",
    "test_results": []
}

paths = glob.glob('./初赛测试视频/*')
paths.sort()

for video_path in paths:
    print(video_path)

    clip_id = video_path.split('/')[-1]
    print(clip_id)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES,total_frames/2)
    img = cap.read()[1]

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transforms(image).unsqueeze(0)

    single_video_result = {
        "clip_id": clip_id,
        "scerario": "city street",
        "weather": "clear",
        "period": "daytime",
        "road_structure": "normal",
        "general_obstacle": "nothing",
        "abnormal_condition": "nothing",
        "ego_car_behavior": "go straight",
        "closest_participants_type": "passenger car",
        "closest_participants_behavior": "slow down"
    }

    for keyword in en_match_words.keys():
        if keyword not in ["weather", "road_structure"]:
            continue

        texts = np.array(en_match_words[keyword])

        with paddle.no_grad():
            logits_per_image, logits_per_text = model(image, tokenize(en_match_words[keyword]))
            probs = paddle.nn.functional.softmax(logits_per_image, axis=-1)

        probs = probs.numpy()
        single_video_result[keyword] = texts[probs[0].argsort()[::-1][0]]
        
    submit_json["test_results"].append(single_video_result)

    with open('jzxd_result.json', 'w', encoding='utf-8') as up:
        json.dump(submit_json, up, ensure_ascii=False)
