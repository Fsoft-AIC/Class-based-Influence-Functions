import yaml
import os
import argparse
import torch
import wandb
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import tqdm
from models.BertSequence import build_bert_sequence_model, load_bert_sequence_model
from dataloaders.ner_conll2003 import conll2003_get_dataloader
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from dataloaders.ner_conll2003 import get_labels

def train(model, dataloader, optimizer, mode='train'):
    epoch_loss = 0.0
    preds, labs = [], []
    
    if mode == 'train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    for batch in tqdm.tqdm(dataloader):
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
        
        # gradient clipping
        # torch.nn.utils.clip_grad_norm_(
        #     parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        # )
        
        # backward pass
        if mode == 'train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.module.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        labels = labels.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()

        preds.extend(predictions)
        labs.extend(labels)

    # print(classification_report(labs, preds))
    metrics = {"{}_acc".format(mode): accuracy_score(labs, preds),
               "{}_f1".format(mode): f1_score(labs, preds, average="weighted"),
               "{}_precision".format(mode):  precision_score(labs, preds, average="weighted"),
               "{}_recall".format(mode): recall_score(labs, preds, average="weighted"),
               "{}_loss".format(mode): epoch_loss/len(dataloader)}

    return metrics, preds, labs


if __name__ == '__main__':
    train_file = os.path.join('data', 'conll2003', 'ner', 'noise_BItags_30sen_30tok.txt')
    val_file = os.path.join('data', 'conll2003', 'ner', 'valid.txt')
    dir_checkpoint = os.path.join('checkpoints', 'conll2003', 'SEED4_NER_CoNLL2003_noise_BItags_30sen_30tok')
    run_name = 'SEED4_NER_CoNLL2003_noise_BItags_30sen_30tok'
    batch_size = 64
    device = 'cuda'
    num_labels = len(get_labels())
    num_epochs = 20
    SEED = 4
    name_project = "NER-CoNLL2003"
    learning_rate = 5e-5

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    if not os.path.isdir(dir_checkpoint):
        print(f"Directory {dir_checkpoint} does not exist")
        os.makedirs(dir_checkpoint)
        print(f"Created {dir_checkpoint}")

    if not torch.cuda.is_available() and device == 'cuda':
        print('Your device don\'t have cuda, please check or select cpu and retraining')
        exit(0)

    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")


    # Load data
    train_dataloader = conll2003_get_dataloader(
        file_name=train_file,
        batch_size=batch_size,
        mode='train',
        num_workers=32
    )
    val_dataloader = conll2003_get_dataloader(
        file_name=val_file,
        batch_size=batch_size,
        mode='test',
        num_workers=32
    )
    # Build model
    model = build_bert_sequence_model(num_labels=num_labels,device=device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # config best
    wandb.init(project=name_project, name= run_name)
    wandb.config = {
        "seed": SEED,
        "train_file": train_file,
        "val_file": val_file,
        "num_labels": num_labels,
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size
    }

    best_valid_f1 = (-1.0) * float("Inf")
    best_epoch = 0

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        train_metrics, train_preds, train_labs = train(model, train_dataloader, optimizer, mode='train')
        val_metrics, val_preds, val_labs = train(model, val_dataloader, optimizer, mode='val')

        print("Train metricc:", train_metrics)
        print("Val metrics", val_metrics)

        torch.save(model.state_dict(), os.path.join(dir_checkpoint, f'epoch_{epoch+1}.pt'))

        if best_valid_f1 < val_metrics["val_f1"]:
            best_valid_f1 = val_metrics["val_f1"]
            torch.save(model.state_dict(), os.path.join(dir_checkpoint, f'best.pt'))
            f = open(os.path.join(dir_checkpoint, f'best_epoch.txt'), "w")
            f.write(str(epoch+1))
            f.close()
            print(f"Best model saved at epoch {epoch+1}")

            print("Classification Report on Train Dataset:")
            print(classification_report(train_labs, train_preds))
            print("Classification Report on Valid Dataset:")
            print(classification_report(val_labs, val_preds))
        
        train_metrics.update(val_metrics)
        wandb.log(train_metrics)
        wandb.watch(model)

    print(f'Finished training')
        
