import sys
import re
import os

part = sys.argv[1]

if not os.path.exists("../data/vaccine_text_wo_distribution"):
    os.makedirs("../data/vaccine_text_wo_distribution")

distribution_regex1 = re.compile(r'\b(distribution|distributed|distributing|distribute|distributes|deliver|delivered|delivery|batch|shipments|prioritize|prioritizing|prioritizes|1a|1b|1c|eligibility|deployment)\b')
distribution_regex2 = re.compile(r'\breceiv+?(ed\b|es\b|ing\b|e\b).*[0-9].*\bdose+?(s\b|\b)')
f = open('../data/vaccine_text/vaccine_text_part%s.csv'%(part),'r')
wo_dist_out = open('../data/vaccine_text_wo_distribution/vaccine_text_wo_distribution_part%s.csv'%(part), 'w+')
s = '|$|' 
for lines in f.readlines():
    try:
        lst = lines.split('|$|')
        read_text = lst[0]
        clean_text = lst[1]
        read_user_id = lst[2]
        read_tweet_id = lst[3]
        out_str = s.join([read_text, clean_text, read_user_id, read_tweet_id])
        if (distribution_regex1.search(clean_text)) or (distribution_regex2.search(clean_text)) is False:
            wo_dist_out.writelines(out_str)

    except:
        pass
wo_dist_out.close()
f.close()

