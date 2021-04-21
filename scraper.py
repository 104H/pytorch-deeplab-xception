'''
         15 April 2021
         A scraper I made to pull Naeem's results from separate files and prepare a csv
'''

import os, json, re, sys

data = {}

# adding key for best_pred with an empty list
data['foldername'] = []
data['best_pred'] = []
data['accuracy'] = []
data['accuracy_class'] = []
data['mIoU'] = []
data['fwIoU'] = []
data['train_size'] = []
data['val_size'] = []
data['checkpoint'] = []

#for root, dirs, files in os.walk("/home/robot/git/pytorch-deeplab-xception/run/cropweed/deeplab-resnet"):
for root, dirs, files in os.walk("/home/robot/hunaid/pytorch-deeplab-xception/run/cropweed/deeplab-resnet"):
        for d in dirs:
                try:
                        f = open( os.path.join(root, d, "parameters.txt"), 'r' )
                        text = f.read()

                        data['foldername'].append(d)

                        # parse the file as a json object
                        finding = re.findall("(.+):(.+)", text)

                        for f in finding:
                            if f[0] in data.keys():
                                data[ f[0] ].append( f[1] )
                            else:
                                data[ f[0] ] = [ f[1] ]

                        # check if best_pred file exists, in which case add it to the json object
                        try:
                                b = open( os.path.join(root, d, "best_pred.txt") ).read()
                                data['best_pred'].append(b)
                        except:
                                data['best_pred'].append(None)

                        # pick up accuracy measures from validation (without --infer flag and checkpoint)
                        try:
                            b = open( os.path.join(root, d, "output") ).read()

                            checkpoint = re.findall("\/(\w+)\/checkpoint.pth.tar", b)
                            data['checkpoint'].append( checkpoint[0] if len(checkpoint) > 0 else None )

                            data['accuracy'].append( re.findall("Acc:(\d+.\d+),", b)[0] )
                            data['accuracy_class'].append( re.findall("Acc_class:(\d+.\d+),", b)[0] )
                            data['mIoU'].append( re.findall("mIoU:(\d+.\d+),", b)[0] )
                            data['fwIoU'].append( re.findall("fwIoU: (\d+.\d+)", b)[0] )
                            data['train_size'].append( re.findall("Found (\d+) train images", b)[0] )
                            data['val_size'].append( re.findall("Found (\d+) valid images", b)[0] )
                        except:
                            pass

                        # append json object to larger list of objects
                except:
                        pass

print(data)
import pandas as pd
df = pd.DataFrame.from_dict(data)
df = df[ ["foldername","best_pred","accuracy","accuracy_class","mIoU","fwIoU","train_size","val_size","datset","backbone","epoch"] ]
df.to_csv("trainingresults")

