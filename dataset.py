import glob
import random
import re
import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

class CompositeDataset(Dataset):
    def __init__(self, samm_path, phase, str_loso, transform=None, transform_norm=None):
        self.phase = phase
        self.transform = transform
        self.transform_norm = transform_norm
        self.casme_path = '../data/datasets/casme2'
        self.samm_path = samm_path
        self.smic_path = '../data/datasets/SMIC-HC'
        self.of_path = '../data/datasets/com_of'

        SUBJECT_COLUMN = 0
        FILE_COLUMN = 1
        ONSET_COLUMN = 2
        APEX_COLUMN = 3
        OFFSET_COLUMN = 4
        LABEL_AU_COLUMN = 5
        LABEL_EMOTION_COLUMN = 6

        str_dataset = str_loso.split('_')[0]
        num_loso = str_loso.split('_')[1]
        self.str_dataset = str_dataset

        self.subjects = []
        self.file_names = []
        self.onset_frames = []
        self.apex_frames = []
        self.offset_frames = []
        self.labels_au = []
        self.labels_emotion = []

        self.str_datasets = []

        # CASME2
        df = pd.read_excel(os.path.join(self.casme_path, 'CASME2-coding-20190701.xlsx'), usecols=[0, 1, 3, 4, 5, 7, 8])
        df['Subject'] = df['Subject'].apply(str)

        if phase == 'train':
            if str_dataset == 'casme2':
                dataset = df.loc[df['Subject']!=num_loso]
            else:
                dataset = df
        else:
            if str_dataset == 'casme2':
                dataset = df.loc[df['Subject']==num_loso]
                # print(dataset)
            else:
                dataset = df.loc[df['Subject']=='-1']
                # print(dataset)

        subjects = dataset.iloc[:, SUBJECT_COLUMN].values
        file_names = dataset.iloc[:, FILE_COLUMN].values
        onset_frames = dataset.iloc[:, ONSET_COLUMN].values
        apex_frames = dataset.iloc[:, APEX_COLUMN].values
        offset_frames = dataset.iloc[:, OFFSET_COLUMN].values
        labels_au = dataset.iloc[:, LABEL_AU_COLUMN].values
        labels_emotion = dataset.iloc[:, LABEL_EMOTION_COLUMN].values

        for (subject, file_name, onset, apex, offset, label_au, label_emotion) in zip(subjects, file_names, onset_frames, apex_frames, offset_frames, labels_au, labels_emotion):
            if label_emotion in ['happiness', 'repression', 'disgust', 'surprise']:
                self.subjects.append(subject)
                self.file_names.append(file_name)
                self.onset_frames.append(int(onset))
                self.apex_frames.append(int(apex))
                self.offset_frames.append(int(offset))

                self.str_datasets.append('casme2')
                
                # Positive-0 Surprise-1 Negative-2
                if label_emotion == 'happiness':
                    self.labels_emotion.append(0)
                elif label_emotion == 'surprise':
                    self.labels_emotion.append(1)
                elif label_emotion in ['repression', 'disgust']:
                    self.labels_emotion.append(2)
        
        # SAMM
        df = pd.read_excel(os.path.join(self.samm_path, 'SAMM_Micro_FACS_Codes_v2.xlsx'), usecols=[0, 1, 3, 4, 5, 8, 9])
        df['Subject'] = df['Subject'].apply(str)

        if phase == 'train':
            if str_dataset == 'samm':
                dataset = df.loc[df['Subject']!=num_loso]
            else:
                dataset = df
        else:
            if str_dataset == 'samm':
                dataset = df.loc[df['Subject']==num_loso]
                # print(dataset)
            else:
                dataset = df.loc[df['Subject']=='-1']
                # print(dataset)

        subjects = dataset.iloc[:, SUBJECT_COLUMN].values
        file_names = dataset.iloc[:, FILE_COLUMN].values
        onset_frames = dataset.iloc[:, ONSET_COLUMN].values
        apex_frames = dataset.iloc[:, APEX_COLUMN].values
        offset_frames = dataset.iloc[:, OFFSET_COLUMN].values
        labels_au = dataset.iloc[:, LABEL_AU_COLUMN].values
        labels_emotion = dataset.iloc[:, LABEL_EMOTION_COLUMN].values

        for (subject, file_name, onset, apex, offset, label_au, label_emotion) in zip(subjects, file_names, onset_frames, apex_frames, offset_frames, labels_au, labels_emotion):
            if label_emotion in ['Happiness', 'Surprise', 'Anger', 'Disgust', 'Sadness', 'Fear', 'Contempt']:
                self.subjects.append(subject)
                self.file_names.append(file_name)
                self.onset_frames.append(int(onset))
                self.apex_frames.append(int(apex))
                self.offset_frames.append(int(offset))

                self.str_datasets.append('samm')
                
                # Positive-0 Surprise-1 Negative-2
                if label_emotion == 'Happiness':
                    self.labels_emotion.append(0)
                elif label_emotion == 'Surprise':
                    self.labels_emotion.append(1)
                elif label_emotion in ['Anger', 'Disgust', 'Sadness', 'Fear', 'Contempt'] :
                    self.labels_emotion.append(2)

        # SMIC
        list_subject_paths = glob.glob(os.path.join(self.smic_path, '*'))
        list_subjects = [os.path.split(filepath)[-1] for filepath in list_subject_paths]

        if phase == 'train':
            if num_loso in list_subjects:
                list_subjects.remove(num_loso)
        else:
            if num_loso in list_subjects:
                list_subjects = [num_loso]
            else:
                list_subjects = []

        for subject in list_subjects:
            list_clip_paths_positve = glob.glob(os.path.join(self.smic_path, subject, 'micro', 'positive', '*'))
            list_clip_paths_surprise = glob.glob(os.path.join(self.smic_path, subject, 'micro', 'surprise', '*'))
            list_clip_paths_negative = glob.glob(os.path.join(self.smic_path, subject, 'micro', 'negative', '*'))

            for clip_path in list_clip_paths_positve:
                list_file_paths = glob.glob(os.path.join(clip_path, '*'))
                list_file_paths.sort()

                onset_image_path = list_file_paths[0]
                onset = int(os.path.split(onset_image_path)[-1].split('reg_image')[1].split('.bmp')[0])
                apex_image_path = list_file_paths[int(len(list_file_paths) / 2)]
                apex = int(os.path.split(apex_image_path)[-1].split('reg_image')[1].split('.bmp')[0])
                offset_image_path = list_file_paths[-1]
                offset = int(os.path.split(offset_image_path)[-1].split('reg_image')[1].split('.bmp')[0])

                self.subjects.append(subject)
                self.file_names.append(os.path.split(clip_path)[-1])
                self.onset_frames.append(onset)
                self.apex_frames.append(apex)
                self.offset_frames.append(offset)
                self.labels_emotion.append(0)

                self.str_datasets.append('smic')

            for clip_path in list_clip_paths_surprise:
                list_file_paths = glob.glob(os.path.join(clip_path, '*'))
                list_file_paths.sort()
                
                onset_image_path = list_file_paths[0]
                onset = int(os.path.split(onset_image_path)[-1].split('reg_image')[1].split('.bmp')[0])
                apex_image_path = list_file_paths[int(len(list_file_paths) / 2)]
                apex = int(os.path.split(apex_image_path)[-1].split('reg_image')[1].split('.bmp')[0])
                offset_image_path = list_file_paths[-1]
                offset = int(os.path.split(offset_image_path)[-1].split('reg_image')[1].split('.bmp')[0])

                self.subjects.append(subject)
                self.file_names.append(os.path.split(clip_path)[-1])
                self.onset_frames.append(onset)
                self.apex_frames.append(apex)
                self.offset_frames.append(offset)
                self.labels_emotion.append(1)

                self.str_datasets.append('smic')

            for clip_path in list_clip_paths_negative:
                list_file_paths = glob.glob(os.path.join(clip_path, '*'))
                list_file_paths.sort()
                
                onset_image_path = list_file_paths[0]
                onset = int(os.path.split(onset_image_path)[-1].split('reg_image')[1].split('.bmp')[0])
                apex_image_path = list_file_paths[int(len(list_file_paths) / 2)]
                apex = int(os.path.split(apex_image_path)[-1].split('reg_image')[1].split('.bmp')[0])
                offset_image_path = list_file_paths[-1]
                offset = int(os.path.split(offset_image_path)[-1].split('reg_image')[1].split('.bmp')[0])

                self.subjects.append(subject)
                self.file_names.append(os.path.split(clip_path)[-1])
                self.onset_frames.append(onset)
                self.apex_frames.append(apex)
                self.offset_frames.append(offset)
                self.labels_emotion.append(2)

                self.str_datasets.append('smic')

    def __len__(self):
        return len(self.onset_frames)
    
    def __getitem__(self, idx):
        onset = self.onset_frames[idx]
        apex = self.apex_frames[idx]
        offset = self.offset_frames[idx]

        self.str_dataset = self.str_datasets[idx]

        if self.phase == 'train':
            # on0 = str(onset)
            # apex0 = str(random.randint(int(apex - int(0.15 / 4 * (apex - onset))), int(apex + int(0.15 / 4 * (offset - apex)))))
            on0 = str(random.randint(int(onset), int(onset + int(0.15 * (apex - onset) / 4))))
            apex0 = str(random.randint(int(apex - int(0.15* (apex - onset) / 4)), apex))

            subject = str(self.subjects[idx])
            file_name = str(self.file_names[idx])
        else:
            on0 = str(onset)
            apex0 = str(apex)

            subject = str(self.subjects[idx])
            file_name = str(self.file_names[idx])

        if self.str_dataset == 'casme2':
            subject = 'sub' + (subject if len(subject) == 2 else '0' + subject)
            image_on0_name = 'reg_img' + on0 + '.jpg'
            image_on0_path = os.path.join(self.casme_path, 'Cropped-updated/Cropped/', subject, file_name, image_on0_name)
            image_on0 = Image.open(image_on0_path)
            image_apex0_name = 'reg_img' + apex0 + '.jpg'
            image_apex0_path = os.path.join(self.casme_path, 'Cropped-updated/Cropped/', subject, file_name, image_apex0_name)
            image_apex0 = Image.open(image_apex0_path)
            label_emotion = self.labels_emotion[idx]
        elif self.str_dataset == 'samm':
            subject = subject.zfill(3)
            pattern = '006_1_*|011_1_*|013_1_*|012_1_*|014_1_*|016_7_*|018_1_*|018_7_*|020_1_*|023_1_*|026_1_*|030_1_*|033_1_*|034_7_*|035_1_*|035_7_*|036_7_*|'
            num_zfill = 4 if re.match(pattern, file_name).group() == '' else 5

            image_on0_name = subject + '_' + on0.zfill(num_zfill) + '.jpg'
            image_on0_path = os.path.join(self.samm_path, subject, file_name, image_on0_name)
            image_on0 = Image.open(image_on0_path).convert('RGB')
            image_apex0_name = subject + '_' + apex0.zfill(num_zfill) + '.jpg'
            image_apex0_path = os.path.join(self.samm_path, subject, file_name, image_apex0_name)
            image_apex0 = Image.open(image_apex0_path).convert('RGB')
            label_emotion = self.labels_emotion[idx]
        elif self.str_dataset == 'smic':
            label_emotion = self.labels_emotion[idx]
            if label_emotion == 0:
                str_emotion = 'positive'
            elif label_emotion == 1:
                str_emotion = 'surprise'
            elif label_emotion == 2:
                str_emotion = 'negative'

            image_on0_name = 'reg_image' + on0.zfill(6) + '.bmp'
            image_on0_path = os.path.join(self.smic_path, subject, 'micro', str_emotion, file_name, image_on0_name)
            image_on0 = Image.open(image_on0_path)
            image_apex0_name = 'reg_image' + apex0.zfill(6) + '.bmp'
            image_apex0_path = os.path.join(self.smic_path, subject, 'micro', str_emotion, file_name, image_apex0_name)
            image_apex0 = Image.open(image_apex0_path)

        of_u = Image.open(os.path.join(self.of_path, subject+file_name+str(label_emotion)+'u.jpg'))
        of_v = Image.open(os.path.join(self.of_path, subject+file_name+str(label_emotion)+'v.jpg'))

        transform_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        if self.transform is not None:
            image_on0 = self.transform(image_on0)
            image_apex0 = self.transform(image_apex0)

            image_on0 = transform_norm(image_on0)
            image_apex0 = transform_norm(image_apex0)

            of_u = self.transform(of_u)
            of_v = self.transform(of_v)

        if self.transform_norm is not None and self.phase == 'train':
            ALL = torch.cat((image_on0, image_apex0, of_u, of_v), dim=0)
            # print(ALL.shape)
            ALL = self.transform_norm(ALL)
            image_on0 = ALL[0:3, :, :]
            image_apex0 = ALL[3:6, :, :]
            of_u = ALL[6:7, :, :]
            of_v = ALL[7:8, :, :]

        return image_on0, image_apex0, label_emotion, of_u, of_v

def get_dataloaders(dataset='com', data_path='../data/datasets/samm_aligned_wan', batch_size=32, num_workers=8, num_loso='1'):
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])
    data_transforms_norm = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(3),
        transforms.RandomCrop(224, padding=15),
    ])
    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CompositeDataset(data_path, 'train', num_loso, data_transforms, data_transforms_norm)
    val_dataset = CompositeDataset(data_path, 'test', num_loso, data_transforms_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True, 
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=True
    )

    return train_loader, val_loader