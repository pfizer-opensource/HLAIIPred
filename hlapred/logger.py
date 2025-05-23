import os
import yaml
import numpy as np
# from torch.utils.tensorboard import SummaryWriter


def save_configs(dir, model_config=None, training_config=None):
    '''Tool for saving configurations into yaml files for record tracking
    '''
    config_dict = {}
    if model_config is not None:
        model_conf_dict = model_config.__dict__
        config_dict["model"] = model_conf_dict
    if training_config is not None:
        tr_conf_dict = training_config.__dict__
        config_dict["training"] = tr_conf_dict
    with open(os.path.join(dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config_dict, f)


# class Logger:
#     def __init__(self, log_dir, titles):
#         self._log_dir = log_dir
#         print('########################')
#         print('logging outputs to ', log_dir)
#         print('########################')
        
#         self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)
#         self.titles = titles
#         self.csv_log_path = os.path.join(log_dir, "log.csv")
#         with open(self.csv_log_path, "w") as f:
#             f.write(",".join([str(item) for item in titles]) + "\n")

#     def log_hparams(self, hparams, metrics):
#         """Log hyperparameters for results comparison and hparam tuning"""
#         self._summ_writer.add_hparams(hparams, metrics)

#     def log_result(self, results):
#         # tensorboard logging
#         epoch = results['epoch']
#         for item in results:
#             if item != "epoch":
#                 self.log_scalar(results[item], item, epoch)
#         # csv logging
#         with open(self.csv_log_path, "a") as f:
#             f.write(",".join([str(results[item]) for item in self.titles]) + "\n")

#     def log_scalar(self, scalar, name, step_):
#         self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

#     def log_scalars(self, scalar_dict, group_name, step, phase):
#         """Will log all scalars in the same plot."""
#         self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

#     def log_image(self, image, name, step):
#         assert(len(image.shape) == 3)  # [C, H, W]
#         self._summ_writer.add_image('{}'.format(name), image, step)

#     def log_video(self, video_frames, name, step, fps=10):
#         assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
#         self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)

#     def log_paths_as_videos(self, paths, step, max_videos_to_save=2, fps=10, video_title='video'):

#         # reshape the rollouts
#         videos = [np.transpose(p['image_obs'], [0, 3, 1, 2]) for p in paths]

#         # max rollout length
#         max_videos_to_save = np.min([max_videos_to_save, len(videos)])
#         max_length = videos[0].shape[0]
#         for i in range(max_videos_to_save):
#             if videos[i].shape[0]>max_length:
#                 max_length = videos[i].shape[0]

#         # pad rollouts to all be same length
#         for i in range(max_videos_to_save):
#             if videos[i].shape[0]<max_length:
#                 padding = np.tile([videos[i][-1]], (max_length-videos[i].shape[0],1,1,1))
#                 videos[i] = np.concatenate([videos[i], padding], 0)

#         # log videos to tensorboard event file
#         videos = np.stack(videos[:max_videos_to_save], 0)
#         self.log_video(videos, video_title, step, fps=fps)

#     def log_figures(self, figure, name, step, phase):
#         """figure: matplotlib.pyplot figure handle"""
#         assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
#         self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

#     def log_figure(self, figure, name, step, phase):
#         """figure: matplotlib.pyplot figure handle"""
#         self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

#     def log_graph(self, model, model_input):
#         """log the pytorch computation graph"""
#         self._summ_writer.add_graph(model, model_input)

#     def log_hist(self, values, name, step):
#         values = np.array(values)
#         self._summ_writer.add_histogram(name, values, step)

#     def dump_scalars(self, log_path=None):
#         log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
#         self._summ_writer.export_scalars_to_json(log_path)

#     def flush(self):
#         self._summ_writer.flush()
