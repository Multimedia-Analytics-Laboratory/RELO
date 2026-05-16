import os
import glob
import torch
import traceback
from lib.train.admin import multigpu
from torch.utils.data.distributed import DistributedSampler


PUBLIC_CHECKPOINT_PREFIXES = {
    "RELO": "RELO",
    "RELOWarmup": "RELO_WARMUP",
}


def is_main_process(settings):
    return bool(getattr(settings, "is_main_process", getattr(settings, "local_rank", -1) in [-1, 0]))


def get_checkpoint_prefix(net):
    """Return the public checkpoint filename prefix for a model."""
    explicit_name = getattr(net, "checkpoint_name", None) or getattr(net, "checkpoint_prefix", None)
    if explicit_name:
        return explicit_name
    return PUBLIC_CHECKPOINT_PREFIXES.get(type(net).__name__, type(net).__name__)


def get_checkpoint_prefixes(net):
    """Return accepted checkpoint prefixes, newest public name first."""
    primary = get_checkpoint_prefix(net)
    aliases = getattr(net, "checkpoint_aliases", ())
    if isinstance(aliases, str):
        aliases = (aliases,)

    prefixes = [primary, *aliases, type(net).__name__]
    unique_prefixes = []
    for prefix in prefixes:
        if prefix and prefix not in unique_prefixes:
            unique_prefixes.append(prefix)
    return unique_prefixes


class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)
        self.settings = settings

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            '''2021.1.4 New function: specify checkpoint dir'''
            if self.settings.save_dir is None:
                self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            else:
                self._checkpoint_dir = os.path.join(self.settings.save_dir, 'checkpoints')
            print("checkpoints will be saved to %s" % self._checkpoint_dir)

            if is_main_process(self.settings):
                if not os.path.exists(self._checkpoint_dir):
                    print("Training with multiple GPUs. checkpoints directory doesn't exist. "
                          "Create checkpoints directory")
                    os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def train(self, max_epochs, load_latest=False, fail_safe=True,
              load_previous_ckpt=None, policy_load_decoder=False):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """

        epoch = -1
        num_tries = 1
        for i in range(num_tries):
            try:
                if load_previous_ckpt is not None:
                    self.load_state_dict(load_previous_ckpt, policy_load_decoder)
                if load_latest:
                    self.load_checkpoint()
                for epoch in range(self.epoch+1, max_epochs+1):
                    self.epoch = epoch

                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        if self.settings.scheduler_type != 'cosine':
                            self.lr_scheduler.step()
                        else:
                            self.lr_scheduler.step(epoch - 1)
                    # only save the last 10 checkpoints
                    save_every_epoch = getattr(self.settings, "save_every_epoch", False)
                    # save every 10 epochs
                    # save_every_epoch = True
                    if epoch > (max_epochs - 10) or save_every_epoch or epoch % 10 == 0:
                        if self._checkpoint_dir:
                            if is_main_process(self.settings):
                                self.save_checkpoint()
            except:
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')

    def train_epoch(self):
        raise NotImplementedError

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = get_checkpoint_prefix(net)
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings
        }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        print(directory)
        if not os.path.exists(directory):
            print("directory doesn't exist. creating...")
            os.makedirs(directory)

        # First save as a tmp file
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path, file_path)

    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        checkpoint_prefixes = get_checkpoint_prefixes(net)
        net_type = checkpoint_prefixes[0]

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted({
                path
                for prefix in checkpoint_prefixes
                for path in glob.glob('{}/{}/{}_ep*.pth.tar'.format(
                    self._checkpoint_dir, self.settings.project_path, prefix))
            })
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_candidates = [
                '{}/{}/{}_ep{:04d}.pth.tar'.format(
                    self._checkpoint_dir, self.settings.project_path, prefix, checkpoint)
                for prefix in checkpoint_prefixes
            ]
            checkpoint_path = next((path for path in checkpoint_candidates if os.path.exists(path)),
                                   checkpoint_candidates[0])
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        try:
            checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            # 老版本没有 weights_only 参数
            checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        assert checkpoint_dict['net_type'] in checkpoint_prefixes, 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch
        # 2021.1.10 Update the epoch in data_samplers
            for loader in self.loaders:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
        return True

    def load_state_dict(self, checkpoint=None, policy_load_decoder=False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        if isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        print("Loading pretrained model from ", checkpoint_path)
        try:
            checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint_dict["net"]
        if policy_load_decoder:
            decoder_state = {k: v for k, v in state_dict.items() if k.startswith("decoder.")}
            for k, v in decoder_state.items():
                if k.startswith("decoder."):
                    new_key = k.replace("decoder.", "policy_model.")
                    state_dict[new_key] = v.clone()

        missing_k, unexpected_k = net.load_state_dict(state_dict, strict=False)
        print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)
        return True
