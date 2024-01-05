import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def get_tokenizer(config, dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_examples(dataset, language), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_examples(dataset, language):
    for item in dataset:
        yield item['translation'][language]



def get_dataset(config):

    dataset_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split = 'train')

    # Get Tokenizer
    tokenizer_src = get_tokenizer(config, dataset_raw, config["lang_src"])
    tokenizer_tgt = get_tokenizer(config, dataset_raw, config["lang_tgt"])

    # Training and Validation Split
    train_dataset_size = int(0.9 * len(dataset_raw))
    val_dataset_size = len(dataset_raw) - train_dataset_size

    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_dataset_size, val_dataset_size])


    train_dataset = BilingualDataset(train_dataset_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_dataset = BilingualDataset(val_dataset_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"] )

    max_len_src = 0
    max_len_tgt = 0

    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config["lang_tgt"]]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max Length of Source Sentence: {max_len_src}")
    print(f'Max Length of Target Sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle = True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], config["d_model"])

    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents = True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], eps = 1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f'Preloading Model: {model_filename}')

        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state["global_step"]

    loss_fn =nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"Processing Epoch: {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)    #(B, seq_len)
            decoder_input = batch['decoder_input'].to(device)    #(B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)      #(B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)      #(B, 1, seq_len, seq_len)

            # Run the tensors through the transformers
            encoder_output = model.encode(encoder_input, encoder_mask)  #(B, seq_len. d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)    #(B, seq_len, d_model )
            proj_output = model.project(decoder_output)     #(B, seq_len, tgt_vocab_size)
                       
            label = batch['label'].to(device)    #(B, seq_len)

            #(B, seq_len, tgt_vocab_size)   -->     (B * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"Loss": f"{loss.item(): 6.3f}"})

            # Log the loss
            writer.add_scalar("Train loss", loss.item(), global_step)
            writer.flush()

            #Backpropogate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


import warnings

torch.cuda.empty_cache()
warnings.filterwarnings('ignore')
config = get_config()

train_model(config)