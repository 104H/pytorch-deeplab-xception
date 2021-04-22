'''
         15 April 2021
         A scraper I made to pull Naeem's results from separate files and prepare a csv
'''

import os, re, sys

data = {}

# adding key for best_pred with an empty list
data['foldername'] = []
data['best_pred'] = []
data['accuracy'] = []
data['accuracy_class'] = []
data['accuracy_pixel'] = []
data['mIoU'] = []
data['fwIoU'] = []
data['train_size'] = []
data['val_size'] = []
data['test_size'] = []
data['checkpoint'] = []

data['Weeds_as_Soil'] = []
data['Weeds_as_Crop'] = []
data['Weeds_as_Weeds'] = []

data['Crops_as_Soil'] = []
data['Crops_as_Crop'] = []
data['Crops_as_Weeds'] = []

data['Soil_as_Soil'] = []
data['Soil_as_Crop'] = []
data['Soil_as_Weeds'] = []

'''
data['synthetic_size'] = []
data['real_size'] = []
'''

def appendNoneConfusion():
    data['Weeds_as_Soil'].append(None)
    data['Weeds_as_Crop'].append(None)
    data['Weeds_as_Weeds'].append(None)

    data['Crops_as_Soil'].append(None)
    data['Crops_as_Crop'].append(None)
    data['Crops_as_Weeds'].append(None)

    data['Soil_as_Soil'].append(None)
    data['Soil_as_Crop'].append(None)
    data['Soil_as_Weeds'].append(None)


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
                except:
                        pass

                # check if best_pred file exists, in which case add it to the json object
                try:
                        b = open( os.path.join(root, d, "best_pred.txt") ).read()
                        data['best_pred'].append(b)
                except:
                        data['best_pred'].append(None)

                # pick up data from the output file
                try:
                    b = open( os.path.join(root, d, "output") ).read()

                    # a checkpoint file name, if there is one needs to be identified
                    checkpoint = re.findall("\/(\w+)\/checkpoint.pth.tar", b)
                    data['checkpoint'].append( checkpoint[0] if len(checkpoint) > 0 else None )

                    # if the job was a test (not a validation) there will be some class confusion metrics to pick
                    metrics = re.findall("(\w+) classified as \n ([\w+: \d+.\d+ \n]+)", b)
                    if len(metrics) > 0:
                        for metric in metrics:
                            t = re.findall("(\w+): (\d+.\d+)", metric[1])
                            for _ in t:
                                data[ metric[0] + '_as_' + _[0] ].append( _[1] )
                    else:
                        appendNoneConfusion()

                    _ = re.findall("Acc:(\d+.\d+),", b)
                    data['accuracy'].append( _[0] if len(_) > 0 else None )

                    _ = re.findall("Acc_class:(\d+.\d+),", b)
                    data['accuracy_class'].append( _[0] if len(_) > 0 else None )

                    _ = re.findall("mI[o|O]U:\s*(\d+.\d+)", b)
                    data['mIoU'].append( _[0] if len(_) > 0 else None )

                    _ = re.findall("[f|F][w|W]I[o|O]U: (\d+.\d+)", b)
                    data['fwIoU'].append( _[0] if len(_) > 0 else None )

                    _ = re.findall("Pixel Accuracy: (\d+.\d+)", b)
                    data['accuracy_pixel'].append( _[0] if len(_) > 0 else None )

                    _ = re.findall("Found (\d+) train images", b)
                    data['train_size'].append( _[0] if len(_) > 0 else None )

                    _ = re.findall("Found (\d+) valid images", b)
                    data['val_size'].append( _[0] if len(_) > 0 else None )

                    _ = re.findall("Found (\d+) test images", b)
                    data['test_size'].append( _[0] if len(_) > 0 else None )

                    '''
                    _ = re.findall("Found (\d+) synthetic images", b)
                    data['synthetic_size'].append( _[0] if len(_) > 0 else None )

                    _ = re.findall("Found (\d+) real images", b)
                    data['real_size'].append( _[0] if len(_) > 0 else None )
                    '''

                except Exception as e:
                    print(str(e))

#print(data)

import pandas as pd
df = pd.DataFrame.from_dict(data)

#df['synthetic_percentage'] = df.synthetic_size.astype(int) / ( df.real_size.astype(int) + df.synthetic_size.astype(int) )

df = df[ ["foldername", "checkpoint", "best_pred", "accuracy", "accuracy_class", "accuracy_pixel", "mIoU", "fwIoU", "train_size", "val_size", "test_size", "datset", "epoch", 'Weeds_as_Soil', 'Weeds_as_Crop', 'Weeds_as_Weeds', 'Crops_as_Soil', 'Crops_as_Crop', 'Crops_as_Weeds', 'Soil_as_Soil', 'Soil_as_Crop', 'Soil_as_Weeds'] ]
df.to_csv("trainingresults.csv")

