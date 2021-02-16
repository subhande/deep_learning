import torch
import torch.nn as nn
import transformers
import pandas as pd
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score

from sklearn import metrics, model_selection


import os

from tqdm import tqdm


import yaml




class BERTBaseUncased(nn.Module):

    def __init__(self, num_classes=5):

        super(BERTBaseUncased, self).__init__()

        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased", return_dict=False)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, num_classes)

    def forward(self, ids, mask, token_type_ids):
        _, o_2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        b_o = self.bert_drop(o_2)
        output = self.out(b_o)
        return output


if __name__ == '__main__':

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = config["inference"]

    BASE_DIR = config['base_dir']
    DF = config['input_file']
    OUT_DF = config['output_file']
    feature_col = config['feature_col']
    MODEL_NAME = config['model']
    device = config['device']
    num_classes = config['num_classes']

    if DF.split('.')[-1] == 'xlsx':
      dfx = pd.read_excel(os.path.join(BASE_DIR, DF))
      print("Dataset Size: ", dfx.shape)
    elif DF.split('.')[-1] == 'csv':
      dfx = pd.read_csv(os.path.join(BASE_DIR, DF))
      print("Dataset Size: ", dfx.shape)
    else:
        print(DF)
        print("Enter a valid input file")
        exit()

    # print("Removing duplicates from dataset")
    # dfx = dfx[~dfx.duplicated()]
    # print("Removing null values from dataset")
    # dfx = dfx.dropna()

    softmax = nn.Softmax(dim=1)

    class_dict = {'negative': 0, 'none': 1, 'positive': 2, 'positive+negative': 3, 'positive+wow': 4}
    inv_class_dict = dict([[class_dict[i], i] for i in class_dict.keys()])
    num_classes = len(list(class_dict.keys()))

    label_map = {0: [0, 0, 1], 1: [0, 0, 0], 2: [1, 0, 0], 3: [1, 0, 1], 4: [1, 1, 0], 5: ["NA", "NA", "NA"]}

    model = BERTBaseUncased(num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', MODEL_NAME), map_location=torch.device(device))['state_dict'])
    model.to(device)
    model.eval()

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    preds = []

    for text in tqdm(dfx[feature_col].tolist()):

        row = []

        row.append(text)

        text = str(text)
        text = " ".join(text.split())

        if text.strip() == 'nan':
            row[0] = "NA"
            row.extend(label_map[5])
            row.append("NA")

        else:
            with torch.no_grad():

                inputs = tokenizer.encode_plus(
                    text,
                    None,
                    add_special_tokens = True,
                    max_length = 128,
                    padding = "max_length",
                    truncation =True
                )

                ids = torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
                mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)
                token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(0).to(device)

                y_pred = model(ids, mask, token_type_ids)
                confidence = float(np.amax(softmax(y_pred).detach().cpu().numpy()[0]))
                y_pred = int(torch.argmax(y_pred, dim=1).detach().cpu().numpy()[0])
               
                row.extend(label_map[y_pred])
                row.append(inv_class_dict[y_pred])
                row.append(confidence)
        preds.append(row)


    out_df = pd.DataFrame(preds, columns=['feedback', 'positive', 'wow', 'negative', 'sentiment', 'confidence'])

    # dfx[target_col] = preds

    if OUT_DF.split('.')[-1] == 'xlsx':
        out_df.to_excel(os.path.join(BASE_DIR, OUT_DF), index=False, float_format='%.2f')
    elif OUT_DF.split('.')[-1] == 'csv':
        out_df.to_csv(os.path.join(BASE_DIR, OUT_DF), index=False, float_format='%.2f')
    else:
        print("Enter valid output file")

    print("Predicted dataframe saved......................")




