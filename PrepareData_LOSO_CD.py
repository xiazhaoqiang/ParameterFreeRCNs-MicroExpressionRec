import os, random
import numpy as np

dbtype_dict = {'casme2':0, 'smic':1, 'samm':2}

def main():
    version = 67 # 0, 1, 2, 4
    verFolder = 'v_{}'.format(version)
    alphas = range(0,1)

    dataDir = os.path.join('data', 'MEGC2019', verFolder)
    filePath = os.path.join('dataset', 'megc_meta.csv')
    meta_dict = {'dbtype':[],'subject':[],'filename':[],'emotion':[],'subid':[],'dbid':[]}
    with open(filePath,'r') as f:
        for textline in f:
            texts = textline.strip('\n').split(',')
            meta_dict['dbtype'].append(texts[0])
            meta_dict['subject'].append(texts[1])
            meta_dict['filename'].append(texts[2])
            meta_dict['emotion'].append(int(texts[3]))
            meta_dict['subid'].append(int(texts[4]))
            meta_dict['dbid'].append(int(texts[5]))
    subjects = list(set(meta_dict['subid']))
    sampleNum = len(meta_dict['dbtype'])
    for subject in subjects:
        idx = meta_dict['subid'].index(subject)
        subjectName = meta_dict['subject'][idx]
        # open the training/val/test list file
        filePath = os.path.join('data','MEGC2019', verFolder, '{}_train.txt'.format(subjectName))
        train_f = open(filePath,'w')
        filePath = os.path.join('data','MEGC2019', verFolder, '{}_test.txt'.format(subjectName))
        test_f = open(filePath,'w')
        # traverse each item, totally 442
        for i in range(0,sampleNum):
            for alpha in alphas:
                fileDir = os.path.join(dataDir, 'flow_alpha{}'.format(alpha))
                fileName = '{}_{}_{}.png'.format(meta_dict['dbtype'][i], meta_dict['subject'][i],
                                                 meta_dict['filename'][i])
                filePath = os.path.join(fileDir, fileName)
                if int(meta_dict['subid'][i]) == int(subject):
                    test_f.write('{} {} {}\n'.format(filePath,meta_dict['emotion'][i],meta_dict['dbid'][i]))
                else:
                    train_f.write('{} {} {}\n'.format(filePath,meta_dict['emotion'][i],meta_dict['dbid'][i]))
        print('The {}-th subject: {}.'.format(subject,subjectName))
        train_f.close()
        test_f.close()

if __name__ == '__main__':
    main()