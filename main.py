import glob
import json

import numpy as np
import torch
import clip
from PIL import Image
import cv2
from clip import tokenize
from datetime import datetime

model, preprocess = clip.load("ViT-B/32")

paths = glob.glob('./初赛测试视频/*')
paths.sort()

current_date = datetime.today().date().strftime("%Y%m%d")[2:]

submit_json = {
    "author": "jzxd",
    "time": current_date,
    "model": "VitB/32",
    "test_results": []
}

en_match_words = {
"scerario": [
        "rugged, narrow, hills, mountains, wild animals.",
        "curb, wide and flat road, roadblock, sign, building, streetlight, city street, pedestrians.",
        "designated exits, service areas",
        "tunnel, dark environment.",
        "Underground, parking garage",
        "gas pump nozzle, charging equipment.",
        "unknown"
    ],
"weather" : ["clear","cloudy","raining","foggy","snowy","unknown"],
"period" : ["daytime","dawn or dusk","night","unknown"],
"road_structure" : ["normal","crossroads","T-junction","ramp","lane merging","parking lot entrance","round about","unknown"],
"general_obstacle" : ["nothing","speed bumper","traffic cone","water horse","stone","manhole cover","nothing","unknown"],
"abnormal_condition" : ["uneven","oil or water stain","standing water","cracked","nothing","unknown"],
"ego_car_behavior" : ["slow down","go straight","turn right","turn left","stop","U-turn","speed up","lane change","others"],
"closest_participants_type" : ["passenger car","bus","truck","pedestrain","policeman","nothing","others","unknown"],
"closest_participants_behavior" : ["slow down","go straight","turn right","turn left","stop","U-turn","speed up","lane change","others"],
}
replacement_mapping = {
    "rugged, narrow, hills, mountains, wild animals.": "suburbs",
    "curb, wide and flat road, roadblock, sign, building, streetlight, city street, pedestrians.": "city street",
    "designated exits, service areas": "expressway",
    "tunnel, dark environment.": "tunnel",
    "Underground, parking garage": "parking-lot",
    "gas pump nozzle, charging equipment.": "gas or charging stations",
    "unknown": "unknown"
}

for video_path in paths:
    clip_id = video_path.split('/')[-1]
    print(clip_id)
    keys = ["scerario","weather","period","road_structure","general_obstacle","abnormal_condition","ego_car_behavior","closest_participants_type","closest_participants_behavior"]
    prob_single_video = {key:np.zeros((1,len(en_match_words[key]))) for key in keys}
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
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval=60#每隔多少帧识别一次
    for f in range(0,total_frames,frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES,f)
        img = cap.read()[1]

        image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = preprocess(image).unsqueeze(0)

        for keyword in en_match_words.keys():
            if keyword not in ["scerario", "weather", "period", "road_structure"]:
                continue
            text = clip.tokenize(en_match_words[keyword])
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            probs = np.array(probs)
            prob_single_video[keyword] = prob_single_video[keyword] + probs

    print("各场景概率分别为：",prob_single_video["scerario"])
    for keyword in en_match_words.keys():
        if keyword not in ["weather", "scerario","period", "road_structure"]:
            continue
        if keyword == "scerario":
            texts = np.array([replacement_mapping[val] if val in replacement_mapping else val for val in
                              en_match_words[keyword]])
        else:
            texts = np.array(en_match_words[keyword])
        single_video_result[keyword] = texts[prob_single_video[keyword][0].argsort()[-1]]
    if(single_video_result["scerario"]=="parking-lot"):
        single_video_result["weather"] = "unknown"

    print("scerario:",single_video_result["scerario"],"|| weather:",single_video_result["weather"],"|| period:",single_video_result["period"],"|| road_structure:",single_video_result["road_structure"])
    submit_json["test_results"].append(single_video_result)


    with open('jzxd_result.json', 'w', encoding='utf-8') as up:
        json.dump(submit_json, up, ensure_ascii=False)

