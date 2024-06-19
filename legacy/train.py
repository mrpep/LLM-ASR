from torch.utils.data import DataLoader
from dataset_asr import ASRDataset
from llm_asr import GPTModel, WavLM, LLMASR
from data_handler import collate, librispeech_to_csv
from pytorch_lightning import Trainer
from functools import partial
from pathlib import Path
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl
from peft import LoraConfig, get_peft_model


def train(dataset, llm_model, wavlm_model, batch_size, epochs, lr, precision, split, workers, save_path,
          sample_rate=16000, devices='0',lora_rank=8, grad_acc=1, warmup_steps=100):
    asr_dataset = ASRDataset(dataset, split, sample_rate)
    llm = GPTModel(llm_model)
    llm_model = llm.get_model()
    
    if lora_rank > 0:
        lora_config = LoraConfig(
                        r=lora_rank,
                        lora_alpha=16,
                        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                        lora_dropout=0.1,
                        task_type='CAUSAL_LM'
                    )
        llm_model = get_peft_model(llm_model, lora_config)
    llm.set_model(llm_model)
        
    wavlm = WavLM(wavlm_model)
    llm_asr = LLMASR(llm, wavlm, lr, warmup_steps=warmup_steps)

    if llm.tokenizer.pad_token is None:
        llm.tokenizer.pad_token = llm.tokenizer.eos_token

    #if 'Qwen2' in llm_model:
    #    llm.tokenizer.padding_side  = 'left'
    
    asr_loader = DataLoader(asr_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                            collate_fn=partial(collate, tokenizer=llm.tokenizer))
    # mandarle un logger, e.g. wandb
    devices = [int(x) for x in devices.split(',')]
    trainer = Trainer(max_epochs=epochs,
                      precision=precision,
                      default_root_dir=save_path,
                      devices=devices,
                      logger=CSVLogger(save_path, name='logs_metrics'),
                      strategy=DDPStrategy(find_unused_parameters=True),
                      accumulate_grad_batches=grad_acc,
                      gradient_clip_val=1,
                      callbacks=[pl.callbacks.LearningRateMonitor(), pl.callbacks.ModelCheckpoint()])
    trainer.fit(llm_asr, asr_loader)
    trainer.save_checkpoint(save_path / 'final.ckpt')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mls_spanish_opus')
    parser.add_argument('--audio_format', type=str, default='opus')
    parser.add_argument('--llm_model', type=str, default='DeepESP/gpt2-spanish')
    parser.add_argument('--wavlm_model', type=str, default='microsoft/wavlm-base-plus')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--grad_acc',type=int, default=1)
    parser.add_argument('--warmup_steps',type=int, default=100)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if dataset_path.name == 'mls_spanish_opus' and not (dataset_path / args.split / 'transcripts.csv').exists():
        librispeech_to_csv(dataset_path, args.split, args.audio_format)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    train(dataset_path, args.llm_model, args.wavlm_model, args.batch_size, args.epochs, args.lr, args.precision,
          args.split, args.workers, save_path, devices=args.devices, lora_rank=args.lora_rank, grad_acc=args.grad_acc,
          warmup_steps=args.warmup_steps)

#Tokenizer: agregar separador
#Codigo: modularizar
#Val Loss: calcular