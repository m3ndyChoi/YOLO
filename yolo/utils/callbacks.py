# yolo/utils/callbacks.py
"""
Custom callbacks for YOLO training
"""

from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
import torch
import numpy as np


class BestModelCheckpoint(ModelCheckpoint):
    """Best 모델 저장 (AP@0.5 기준)"""
    def __init__(self, dirpath, monitor='val/AP_50', mode='max', save_top_k=3, **kwargs):
        filename = f'best-{{epoch:02d}}-{{{monitor}:.4f}}'
        super().__init__(
            dirpath=dirpath,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            filename=filename,
            auto_insert_metric_name=False,
            **kwargs
        )
        
    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if self.best_model_path and self.best_model_score:
            current_score = self.best_model_score.item()
            print(f"\n🎯 New best! {self.monitor} = {current_score:.4f}")


class CustomEarlyStopping(EarlyStopping):
    """Early Stopping with progress display"""
    def __init__(self, monitor='val/AP_50', patience=30, mode='max', min_delta=0.001, **kwargs):
        # verbose 제거 - EarlyStopping이 자동으로 처리
        super().__init__(
            monitor=monitor,
            patience=patience,
            mode=mode,
            min_delta=min_delta,
            **kwargs
        )
        
    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        
        current = self.wait_count
        if current > 0:
            print(f"\n⏳ Early stopping: {current}/{self.patience} epochs without improvement")
            print(f"   Best {self.monitor}: {self.best_score:.4f}")
    
    def on_train_end(self, trainer, pl_module):
        if self.stopped_epoch > 0:
            print(f"\n🛑 Training stopped early at epoch {self.stopped_epoch}")
        else:
            print(f"\n✅ Completed all {trainer.max_epochs} epochs")


class FinalModelSaver(Callback):
    """학습 종료 시 final 모델 저장"""
    def __init__(self, dirpath):
        self.dirpath = Path(dirpath)
        
    def on_train_end(self, trainer, pl_module):
        final_path = self.dirpath / 'final.ckpt'
        trainer.save_checkpoint(final_path)
        print(f"\n💾 Final model saved: {final_path}")
