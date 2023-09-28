import os
from os.path import join
import json
import threading
import time
import torchvision
import numbers
from torch import nn

# from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, folder, verbose=True, substep_print_freq=100) -> None:
        self.folder = folder
        self.verbose = verbose
        self.substep_print_freq = substep_print_freq

        if not os.path.exists(folder):
            os.makedirs(folder)
        
        self.log_file_path = join(folder, "log.json")
        print(f'Logger - writing new log-file: {self.log_file_path}')
        with open(self.log_file_path, 'w') as file:
            json.dump(dict(), file)

        self.start_time = time.time()
        self.epoch = 0
        self.use_tboard = False
        

    def init_tboard_writer(self):
        if not os.path.exists(self.folder):
            os.makedirs(join(self.folder, "tboard"))

        self.tboard_writer = SummaryWriter(join(self.folder, "tboard"))
        self.use_tboard = True

    def done(self):
        if self.use_tboard:
            print("Logger - closing tboard writer")
            self.tboard_writer.close()

    def write_config(self, parameters:dict):
        def _to_opt_or_str(object):
            if hasattr(object, "options"):
                out = {}
                for k, v in object.options.items():
                    out[k] = _to_opt_or_str(v)
                return out
            if type(object) == list: # do not convert 
                return [_to_opt_or_str(item) for item in object]
            if type(object) == dict:
                return {key: _to_opt_or_str(value) for key, value in object.items()}
            else:
                if isinstance(object, numbers.Number):
                    return object
                return str(object)

        out = {}
        for k, v in parameters.items():
            out[k] = _to_opt_or_str(v)

        with open(join(self.folder, "opt.json"), "w+") as file:
            file.write(json.dumps(out, indent=4))

    def write_step(self, epoch, metrics:dict={}):
        self.epoch = epoch
        metrics.update(
            {"time": time.time() - self.start_time,
             "epoch": epoch}
        )
        print(f"{threading.current_thread().name} - epoch {epoch}: {metrics}")

        with open(self.log_file_path, 'r') as file:
            output_dict = json.load(file)

        for k, v in metrics.items():
            self._check_write(output_dict, k, v)

        with open(self.log_file_path, 'w') as file:
            json.dump(output_dict, file)

    def print_substep(self, batch, total, substep_metrics={}):
        if batch % self.substep_print_freq != 0: return

        batch = str(batch).zfill(len(str(total)))
        substep_metrics["time"] = time.time() - self.start_time
        metrics = ", ".join([f"{k}: {v}" for k, v in substep_metrics.items()])
        print(f"{threading.current_thread().name} - {batch}/{total}, {metrics}")

        if self.use_tboard:
            substep_metrics["batch"] = int(batch)
            self.tboard_writer.add_scalars("substep", substep_metrics, global_step=self.epoch)

    def _check_write(self, output_dict, key, value):
        if key not in output_dict.keys():
            output_dict[key] = [value]
        else:
            output_dict[key].append(value)
        return 
       
    def tboard_img_out(self, images, id):
        if not self.use_tboard:
            print("Tboard not initialized, cannot log output images")
            return
        grid = torchvision.utils.make_grid(images)
        self.tboard_writer.add_image(id, grid, global_step=self.epoch)
        self.tboard_writer.flush()

    def tboard_model_weights(self, model: nn.Module, id=""):
        """
        based on https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md
        """
        def weight_histograms_linear(weights, layer_name):
            flattened_weights = weights.flatten()
            self.tboard_writer.add_histogram(f"weights_{id}_{layer_name}", flattened_weights, global_step=self.epoch)

        def weight_histograms_conv2d(weights, layer_name):
            weights_shape = weights.shape
            num_kernels = weights_shape[0]
            for k in range(num_kernels):
                flattened_weights = weights[k].flatten()
                # print(flattened_weights.dtype)
                self.tboard_writer.add_histogram(f"weights_{id}_{layer_name}/kernel_{k}", flattened_weights, global_step=self.epoch)

        if not self.use_tboard:
            print("Tboard not initialized, cannot log model weights")
            return

        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight_histograms_conv2d(module.weight, name)
            elif isinstance(module, nn.Linear):
                weight_histograms_linear(module.weight, name)


    
