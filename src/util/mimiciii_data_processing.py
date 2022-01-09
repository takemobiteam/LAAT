# 47723/1631/3372 (training_size/validation_size/test_size)
# set the connection to PostgreSQL at Line 139

import pandas as pd
import numpy as np
from src.util.preprocessing import RECORD_SEPARATOR
import operator
import os

conn = None
from nltk.tokenize import sent_tokenize, RegexpTokenizer

# keep only alphanumeric
tokenizer = RegexpTokenizer(r'\w+')

CHAPTER = 1
THREE_CHARACTER = 2
FULL = 3
n_not_found = 0


label_count_dict = dict()
n = 50


def read_admission_ids(train_file, valid_file, test_file, outdir, top_n_labels=None):

    global n_not_found
    import csv

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    df_train = pd.read_csv(train_file)
    df_valid = pd.read_csv(valid_file)
    df_test = pd.read_csv(test_file)

    output_fields = ["Patient_Id", "Admission_Id",
                     "Chapter_Labels", "Three_Character_Labels",
                     "Full_Labels", "Text"]

    training_file = open(outdir + "/train.csv", 'w', newline='')
    training_writer = csv.DictWriter(training_file, fieldnames=output_fields)
    training_writer.writeheader()

    valid_file = open(outdir + "/valid.csv", 'w', newline='')
    valid_writer = csv.DictWriter(valid_file, fieldnames=output_fields)
    valid_writer.writeheader()

    test_file = open(outdir + "/test.csv", 'w', newline='')
    test_writer = csv.DictWriter(test_file, fieldnames=output_fields)
    test_writer.writeheader()



    n_not_found = 0
    process_df(df_train, training_writer, top_n_labels)
    print(n_not_found)
    training_file.close()

    n_not_found = 0
    process_df(df_valid, valid_writer, top_n_labels)
    print(n_not_found)
    valid_file.close()

    n_not_found = 0
    process_df(df_test, test_writer, top_n_labels)
    print(n_not_found)
    test_file.close()

    sorted_labels = sorted(label_count_dict.items(), key=operator.itemgetter(1), reverse=True)
    # print(sorted_labels[0:100])
    output = []
    for i in range(n):
        output.append(sorted_labels[i][0])
    return output


def process_df(df, writer, top_n_labels):
    count = 0
    unique_full_labels = set()

    unique_diag_full_labels = set()
    unique_chapter_labels = set()
    unique_three_character_labels = set()

    unique_proc_full_labels = set()

    for patient_id, row in df:
        count += 1
        if count % 100 == 0:
            print("{}/{}, {} - {} - {} diag labels ~ {} proc labels ~ {} all labels".
                  format(count, len(df),
                         len(unique_chapter_labels), len(unique_three_character_labels), len(unique_diag_full_labels),
                         len(unique_proc_full_labels),
                         len(unique_full_labels)))

        text_labels = get_text_labels(row, top_n_labels)

        if text_labels is not None:

            text = text_labels[0]
            diag_labels = text_labels[1]
            proc_labels = text_labels[2]
            labels = text_labels[3]
            unique_full_labels.update(labels[2].split("|"))
            unique_chapter_labels.update(labels[0].split("|"))
            unique_three_character_labels.update(labels[1].split("|"))
            unique_diag_full_labels.update(diag_labels[2].split("|"))

            unique_proc_full_labels.update(proc_labels[2].split("|"))

            row = {"Patient_Id": patient_id, "Admission_Id": id, "Text": text,
                   "Full_Labels": labels[2],
                   "Chapter_Labels": labels[0],
                   "Three_Character_Labels": labels[1]
                   }

            writer.writerow(row)

    print("{}/{}, {} - {} - {} diag labels ~ {} proc labels ~ {} all labels".
          format(count, len(df),
                 len(unique_chapter_labels), len(unique_three_character_labels), len(unique_diag_full_labels),
                 len(unique_proc_full_labels),
                 len(unique_full_labels)))


def get_text_labels(row, top_n_labels):

    global n_not_found

    text = []
    patient_id = None
    normalised_text, length = normalise_text(row['text'])
    text.append(normalised_text)
    diag_chapter_labels, diag_three_character_labels, diag_full_labels = process_codes([row['icd1'], row['icd2']], True, top_n_labels)

    for lb in diag_full_labels:
        if lb in label_count_dict:
            label_count_dict[lb] += 1
        else:
            label_count_dict[lb] = 1

    diag_full_labels = normalise_labels(label_list=diag_full_labels)
    diag_three_character_labels = normalise_labels(label_list=diag_three_character_labels)
    diag_chapter_labels = normalise_labels(label_list=diag_chapter_labels)


    full_labels = diag_full_labels
    three_character_labels = diag_three_character_labels
    chapter_labels = diag_chapter_labels

    if len(text) > 0 and (len(full_labels) + len(three_character_labels) + len(chapter_labels)) > 0:
        return RECORD_SEPARATOR.join(text), \
               ("|".join(diag_chapter_labels), "|".join(diag_three_character_labels), "|".join(diag_full_labels)), \
               ("|".join([]), "|".join([]), "|".join([])), \
               ("|".join(chapter_labels), "|".join(three_character_labels), "|".join(full_labels)), \
               patient_id
    else:
        n_not_found += 1


def process_codes(codes, is_diagnosis, top_n_labels):
    chapter_labels, three_character_labels, full_labels = [], [], []
    for c in codes:
        if c is not None:
            if type(c) == float and np.isnan(c):
                continue
            if top_n_labels is not None and reformat(c, is_diagnosis, FULL) not in top_n_labels:
                continue

            chapter_label = reformat(c, is_diagnosis, CHAPTER)
            if chapter_label is not None:
                chapter_labels.append(str(chapter_label))

            three_character_label = reformat(c, is_diagnosis, THREE_CHARACTER)
            if three_character_label is not None:
                three_character_labels.append(str(three_character_label))

            full_label = reformat(c, is_diagnosis, FULL)
            if full_label is not None:
                full_labels.append(str(full_label))

    return chapter_labels, three_character_labels, full_labels


def normalise_labels(label_list):
    output = []
    check = set()
    for label in label_list:
        if label not in check:
            output.append(label)
            check.add(label)
    output = sorted(output)
    return output


def normalise_text(text):
    output = []
    length = 0

    for sent in sent_tokenize(text):
        tokens = [token.lower() for token in tokenizer.tokenize(sent) if contains_alphabetic(token)]
        length += len(tokens)

        sent = " ".join(tokens)

        if len(sent) > 0:
            output.append(sent)

    return "\n".join(output), length


def contains_alphabetic(token):
    for c in token:
        if c.isalpha():
            return True
    return False


def reformat(code, is_diag, level=FULL):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits,
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))

    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    if level == THREE_CHARACTER:
        return code.split(".")[0]
    elif level == CHAPTER:
        three_chars = code.split(".")[0]
        if len(three_chars) != 2:
            if three_chars.isdigit():
                value = int(three_chars)
                if 139 >= value >= 1:
                    return "D1"
                elif 239 >= value >= 140:
                    return "D2"
                elif 279 >= value >= 240:
                    return "D3"
                elif 289 >= value >= 280:
                    return "D4"
                elif 319 >= value >= 290:
                    return "D5"
                elif 389 >= value >= 320:
                    return "D6"
                elif 459 >= value >= 390:
                    return "D7"
                elif 519 >= value >= 460:
                    return "D8"
                elif 579 >= value >= 520:
                    return "D9"
                elif 629 >= value >= 580:
                    return "D10"
                elif 679 >= value >= 630:
                    return "D11"
                elif 709 >= value >= 680:
                    return "D12"
                elif 739 >= value >= 710:
                    return "D13"
                elif 759 >= value >= 740:
                    return "D14"
                elif 779 >= value >= 760:
                    return "D15"
                elif 799 >= value >= 780:
                    return "D16"
                elif 999 >= value >= 800:
                    return "D17"
                else:
                    print("Diagnosis: {}".format(code))
            else:
                if three_chars.startswith("E") or three_chars.startswith("V"):
                    return "D18"
                else:
                    print("Diagnosis: {}".format(code))
                    return "D0"
        else:  # Procedure Codes http://www.icd9data.com/2012/Volume3/default.htm
            if three_chars.isdigit():
                value = int(three_chars)
                if value == 0:
                    return "P1"
                elif 5 >= value >= 1:
                    return "P2"
                elif 7 >= value >= 6:
                    return "P3"
                elif 16 >= value >= 8:
                    return "P4"
                elif 17 >= value >= 17:
                    return "P5"
                elif 20 >= value >= 18:
                    return "P6"
                elif 29 >= value >= 21:
                    return "P7"
                elif 34 >= value >= 30:
                    return "P8"
                elif 39 >= value >= 35:
                    return "P9"
                elif 41 >= value >= 40:
                    return "P10"
                elif 54 >= value >= 42:
                    return "P11"
                elif 59 >= value >= 55:
                    return "P12"
                elif 64 >= value >= 60:
                    return "P13"
                elif 71 >= value >= 65:
                    return "P14"
                elif 75 >= value >= 72:
                    return "P15"
                elif 84 >= value >= 76:
                    return "P16"
                elif 86 >= value >= 85:
                    return "P17"
                elif 99 >= value >= 87:
                    return "P18"
                else:
                    print("Procedure: {}".format(code))
            else:
                print("Procedure: {}".format(code))
    else:
        return code


if __name__ == "__main__":
    top_n_labels = read_admission_ids(
        train_file="data/mimicdata/mimic3/train_full_hadm_ids.csv",
        valid_file="data/mimicdata/mimic3/dev_full_hadm_ids.csv",
        test_file="data/mimicdata/mimic3/test_full_hadm_ids.csv",
        outdir="data/mimicdata/mimic3/full/")

    read_admission_ids(
        train_file="data/mimicdata/mimic3/train_50_hadm_ids.csv",
        valid_file="data/mimicdata/mimic3/dev_50_hadm_ids.csv",
        test_file="data/mimicdata/mimic3/test_50_hadm_ids.csv",
        outdir="data/mimicdata/mimic3/50/",
        top_n_labels=top_n_labels)



