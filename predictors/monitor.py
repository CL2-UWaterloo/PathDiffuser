import os
import gc
import psutil
import torch
from pytorch_lightning import Callback

class SystemMemoryMonitor(Callback):
    def __init__(self, log_interval_steps=10):
        self.log_interval_steps = log_interval_steps
        self.process = psutil.Process(os.getpid())
        self.peak_cpu = 0.0
        self.peak_gpu = 0.0

    def log_memory(self, trainer):
        # CPU memory (RSS)
        gc.collect()
        cpu_mem = self.process.memory_info().rss / 1024 ** 2  # MB
        self.peak_cpu = max(self.peak_cpu, cpu_mem)

        metrics = {
            "cpu_mem_MB": cpu_mem,
            "cpu_peak_mem_MB": self.peak_cpu,
        }

        # GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024 ** 2  # MB
            gpu_reserved = torch.cuda.memory_reserved() / 1024 ** 2  # MB
            self.peak_gpu = max(self.peak_gpu, gpu_mem)

            metrics.update({
                "gpu_mem_MB": gpu_mem,
                "gpu_reserved_MB": gpu_reserved,
                "gpu_peak_mem_MB": self.peak_gpu,
            })

        trainer.logger.log_metrics(metrics, step=trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_interval_steps == 0 and trainer.is_global_zero:
            self.log_memory(trainer)

    def on_train_end(self, trainer, pl_module):
        print(f"[CPU Peak] {self.peak_cpu:.2f} MB")
        print(f"[GPU Peak] {self.peak_gpu:.2f} MB")
