from mordecai import Geoparser
import os
import pandas as pd
import sys
sys.path.append('./')
from utilities import clean_str2

geo = Geoparser(threads=False)
geo.geoparse("I traveled from Oxford to Ottawa.")

merged_path = '../data/merged'

part = sys.argv[1]

print('start processing file: %s'%(file))
merged = pd.read_csv(os.path.join(merged_path, 'merged_part%s.csv' % (part)), index_col = 0)
merged['user_location'] = merged['user_location'].apply(str)
parsed_location_lst = []
for location in merged['user_location']:
    if location != 'nan':
        parsed_location = geo.geoparse(clean_str2(location))
        parsed_location_lst.append(parsed_location)
    else:
        parsed_location_lst.append('')
merged['parsed_location'] = parsed_location_lst
merged.to_csv(os.path.join(merged_path, '%s_location_parsed.csv'%(file.split('.csv')[0])))
