import os
import sys
from deepface import DeepFace

part = sys.argv[1]
backends = ['opencv', 'ssd', 'dlib', 'mtcnn']

profile_img_dir = "../data/profile_img/part_%s" % (part)

if not os.path.exists("../data/deepface"):
    os.makedirs("../data/deepface")

out = open('../data/deepface/deepface_part%s.csv'%(part), 'w+')
for img in os.listdir(profile_img_dir):
    try:
        obj = DeepFace.analyze(img_path = os.path.join(profile_img_dir, img),
                               actions = ['age', 'gender', 'race'],
                               detector_backend = backends[3])

        user_id = img.split('_')[-1].split('.')[0]
        age = obj['age']
        region = obj['region']
        gender = obj['gender']
        race = obj['race']
        dominant_race = obj['dominant_race']
        s = ','
        out_str = s.join([user_id, str(age),
                          str(region['x']), str(region['y']), str(region['w']), str(region['h']),
                          gender,
                          str(race['asian']), str(race['indian']), str(race['black']), str(race['white']), str(race['middle eastern']), str(race['latino hispanic']),
                          dominant_race, '\n'])
        out.write(out_str)
    except:
        pass

out.close()
