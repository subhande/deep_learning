import tez
import torch
import torch.nn as nn
import transformers
import pandas as pd


from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score

from sklearn import metrics, model_selection

import os

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import yaml

#=====================================================================================================
# ================================  Dataset ==========================================================
# ====================================================================================================


class BERTDataset:
    def __init__(self, texts, targets=None, max_len=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = str(self.texts[idx])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens = True,
            max_length = self.max_len,
            padding = "max_length",
            truncation =True
        )

        if self.targets is not None:

            resp = {
                "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
                "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
                "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
                "targets": torch.tensor(self.targets[idx], dtype=torch.long),
            }

        else:

            resp = {
                "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
                "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
                "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long)
            }


        return resp




#=====================================================================================================
# ================================  Model ==========================================================
# ====================================================================================================



class BERTBaseUncased(tez.Model):

    def __init__(self, num_classes, num_train_steps):

        super(BERTBaseUncased, self).__init__()

        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased", return_dict=False)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, num_classes)
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"
    
    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=2e-5)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch

    def loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.CrossEntropyLoss()(outputs, targets)
        # return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
    
    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs  = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        # outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
        # targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def forward(self, ids, mask, token_type_ids, targets=None):
        _, o_2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        b_o = self.bert_drop(o_2)
        output = self.out(b_o)
        if targets is None:
            return output, None, None
        loss = self.loss(output, targets)
        metrics = self.monitor_metrics(output, targets)
        return output, loss, metrics


#=====================================================================================================
# ================================  Trainer ==========================================================
# ====================================================================================================


def trainer(config):

    BASE_DIR=config['base_dir']
    DF=config['input_file']
    feature_col=config['feature_col']
    target_col=config['target_col']
    EPOCHS=config['epochs']
    BATCH_SIZE=config['batch_size']
    MODEL_NAME=config['model_name']
    DEVICE = config["device"]
    NUM_CLASSES = config["num_classes"]

    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

    print(f"Reading ................ {DF}")
    if DF.split('.')[1] == 'xlsx':
      dfx = pd.read_excel(os.path.join(BASE_DIR, DF))
    elif DF.split('.')[1] == 'csv':
      dfx = pd.read_csv(os.path.join(BASE_DIR, DF))
    else:
      print("Enter a valid input file")
      return 
    print("Dataset Size: ", dfx.shape)
    print("Removing duplicates from dataset")
    dfx = dfx[~dfx.duplicated()]
    print("Removing null values from dataset")
    dfx = dfx.dropna()

    sentiment = []

    for row in dfx.iterrows():
        if int(row[1].positive) == 1 and  int(row[1].wow) == 1:
            sentiment.append('positive+wow')
        elif int(row[1].positive) == 1 and  int(row[1].negative) == 1:
            sentiment.append('positive+negative')
        elif int(row[1].positive) == 1:
            sentiment.append('positive')
        elif int(row[1].negative) == 1:
            sentiment.append('negative')
        else:
            sentiment.append('none')
    
    dfx[target_col] = sentiment

    dfx = dfx[[feature_col, target_col]]



    print("Classes--------------------------------")
    print(dict(dfx[target_col].value_counts()))
    print("Dataset Size: ", dfx.shape)
    print("----------------------------------------")

    class2idx = {'negative': 0, 'none': 1, 'positive': 2, 'positive+negative': 3, 'positive+wow': 4}
    idx2class = dict([[class2idx[i], i] for i in class2idx.keys()])
    dfx[target_col] = dfx[target_col].apply(lambda x: class2idx[x])

    df_train, df_valid = train_test_split(dfx, test_size=0.10, random_state=42, stratify=dfx[target_col].values)
    print("----------------------------------------")
    print(dict(df_train[target_col].value_counts()))
    print("----------------------------------------")
    print(dict(df_valid[target_col].value_counts()))
    print("----------------------------------------")
    train_dataset = BERTDataset(
        texts=df_train[feature_col].values, targets=df_train[target_col].values
    )

    print(set(dfx[target_col].values))

    valid_dataset = BERTDataset(
        texts=df_valid[feature_col].values, targets=df_valid[target_col].values
    )

    n_train_steps = int(len(df_train) / BATCH_SIZE * EPOCHS)
    print("Number of classes: ", dfx[target_col].nunique())
    model = BERTBaseUncased(num_classes=NUM_CLASSES, num_train_steps=n_train_steps)

    tb_logger = tez.callbacks.TensorBoardLogger(log_dir=os.path.join(BASE_DIR, 'logs'))
    es = tez.callbacks.EarlyStopping(monitor="valid_loss", model_path=os.path.join(BASE_DIR, 'models', f"{MODEL_NAME}_best.pt"), patience=5)
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_bs=BATCH_SIZE,
        device=DEVICE,
        epochs=EPOCHS,
        callbacks=[tb_logger, es],
        fp16=True,
    )
    model.save(os.path.join(BASE_DIR, 'models', f"{MODEL_NAME}.pt"))
    model.load(os.path.join(BASE_DIR, 'models', f"{MODEL_NAME}_best.pt"))
    torch.save({'state_dict': model.state_dict()}, os.path.join(BASE_DIR, 'models', f"{MODEL_NAME}_best_state_dict.pt"))
    print("Best model dict saved ----------------")

    # Predict
    print("Starting prediction")
    model.eval()

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    preds = []

    with torch.no_grad():
        for text in tqdm(dfx[feature_col].tolist()):

            text = str(text)
            text = " ".join(text.split())
            inputs = tokenizer.encode_plus(
                    text,
                    None,
                    add_special_tokens = True,
                    max_length = 128,
                    padding = "max_length",
                    truncation =True
            )

            ids = torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0).to(DEVICE)
            mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).unsqueeze(0).to(DEVICE)
            token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(0).to(DEVICE)

            y_pred, _, _ = model(ids, mask, token_type_ids)
            y_pred = int(torch.argmax(y_pred, dim=1).detach().cpu().numpy()[0])

            preds.append(idx2class[y_pred])

    dfx[target_col + '_predicted'] = preds
    dfx[target_col] = dfx[target_col].apply(lambda x: idx2class[x])
    dfx.to_csv(os.path.join(BASE_DIR, DF.split('.')[0] + "_pred.csv"), index=False)

    print("Predicted dataframe saved......................")




if __name__ == "__main__":

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config['training'])
    print("Start training")
    trainer(config['training'])
    print("Training finished")



    