import os
os.environ["OMP_NUM_THREADS"] = "18"
from collections import defaultdict

from config.rl_config import RL_Config
from symbolic import dataset
from symbolic.models import get_model, get_input_size
from symbolic.args import init_argparse
from symbolic.process import DynamicSample

import torch
from trl.trainer import BaseTrainer
from os.path import join
from tqdm import tqdm as tq
import json

import numpy as np
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class RLTrainer(BaseTrainer):
    """
    The RLTrainer uses a diffusion policy optimization algorithm to optimise diffusion models.
    """
    def __init__(self):
        self.args = init_argparse()
        self.config = RL_Config(self.args)
        self.generate_model = self._create_pipline(self.config.ckpt_path)

        params = list(self.generate_model.parameters())
        self.optimizer = self._setup_optimizer(params)
        self.datasets = dataset.initialize_datasets(self.config.config)
        
    def save_model(self,path,name):
        torch.save(self.generate_model.state_dict(), join(path,f"{name}_generative_model.npy"))
        torch.save(self.optimizer.state_dict(), join(path,f"{name}_optimize.npy"))
    
    def _create_pipline(self, ckpt_path=None):
        '''
        Load model parameters
        
        Returns:
            flow (torch.nn.moudle): The model
        '''
        config = self.config
            
        flow = get_model(config, config.device)
        flow.to(config.device)
        if ckpt_path is not None:
            fn = 'generative_model.npy'
            flow_state_dict = torch.load(ckpt_path+fn, map_location=config.device)
            flow.load_state_dict(flow_state_dict)
        
        return flow
    
    def filter_data(self, logs, data, batch_sample, epoch):
        data_list = []
        sol_list = []
        for i in range(len(data)):
            idx = str(data[i]['idx'])
            if logs[idx][0] != 1.0:
                data_list.append(data[i])
        train_dataset = DynamicSample(data_list, num_pts=len(data_list))
        train_loader = DataLoader(train_dataset, batch_size=batch_sample, shuffle=True)
        return train_loader
        
    def get_sample_num(self, idx, batch_sample, logs, dynamic_list):
        correct_rate_log = []
        sample_num = [0] * batch_sample
        for i in range(batch_sample):
            correct_rate_log.append(logs[str(idx[i].item())][0])
        sorted_log = sorted(enumerate(correct_rate_log), key=lambda x: x[1])
        for i, log in enumerate(sorted_log):
            sample_num[log[0]] = dynamic_list[i]
        return sample_num

    def step(self, epoch: int, global_step: int, root_path: str):
        """
        Perform a single step of training.

        Args:
            epoch (int): The current epoch.
            global_step (int): The current global step.
            root_path (str): The root path to save checkpoints and logs.

        Side Effects:
            - Model weights are updated
            - Logs the statistics.

        Returns:
            global_step (int): The updated global step.

        """
        results = []
        trained_step = 0
        if epoch==0:
            logs = {}
        else:
            log_path = os.path.join(root_path, "epoch"+str(epoch)+"_log.json")
            with open(log_path, "r") as file:
                logs = json.load(file)

        if epoch==0:
            train_loader = DataLoader(self.datasets['train'], batch_size=self.config.sample_batch_size, shuffle=True)
        else:
            train_loader = self.filter_data(logs, self.datasets['train'], self.config.sample_batch_size, epoch)

        for batch_idx, batch in enumerate(train_loader):
            # 1. prepare input
            idx = batch['idx']
            batch_sample = idx.shape[0]
            sample_num = [self.config.init_sample_num]*batch_sample if epoch==0 else self.get_sample_num(idx, batch_sample, logs, self.config.dynamic_sample_num)
            image_size, channels = get_input_size(self.config.config.task, self.config.config.size)
            puzzle = torch.cat([batch['puzzle'][i].repeat([sample_num[i],1,1,1]) for i in range(batch_sample)], dim=0).view(-1, channels, image_size[0], image_size[1]).to(self.config.device)
            mask = torch.cat([batch['mask'][i].repeat([sample_num[i],1,1,1]) for i in range(batch_sample)]).view(-1, channels, image_size[0], image_size[1]).to(self.config.device)
            if self.config.config.task=="sudoku" or self.config.config.task=="warcraft":
                solution = torch.cat([batch['sol'][i].repeat([sample_num[i],1,1]) for i in range(batch_sample)]).view(-1, image_size[0], image_size[1]).to(self.config.device)
            elif self.config.config.task=="sushi":
                solution = torch.cat([batch['sol'][i].repeat([sample_num[i],1]) for i in range(batch_sample)]).to(self.config.device)
            else:
                solution_onehot = torch.cat([batch['sol_onehot'][i].repeat([sample_num[i],1,1,1]) for i in range(batch_sample)]).view(-1, channels, image_size[0], image_size[1]).to(self.config.device)
            batch_size = puzzle.shape[0]
            
            # 2. generate samples
            samples = self._generate_samples(
                iterations=self.config.sample_num_batches_per_epoch,
                batch_size=batch_size,
                mask=mask,
                context=puzzle
            )
            
            # 3. compute rewards
            if self.config.config.task=="sudoku":
                rewards_, accs_ = self.compute_rewards_sudoku(samples, solution)
            elif self.config.config.task=="maze":
                rewards_, accs_ = self.compute_rewards_maze(samples, solution_onehot)
            elif self.config.config.task=="grid":
                rewards_, accs_ = self.compute_rewards_grid(samples, solution_onehot)
            elif self.config.config.task=="sushi":
                rewards_, accs_ = self.compute_rewards_sushi(samples, solution)
            elif self.config.config.task=="warcraft":
                rewards_, accs_ = self.compute_rewards_warcraft(samples, solution)
            accs = torch.tensor(accs_, dtype=float).to(samples['x'].device)
            rewards = torch.tensor(rewards_, dtype=float).to(samples['x'].device)

            # Normalize the rewards to compute the advantages
            split_rewards = torch.split(rewards, sample_num)
            mean_grouped_rewards = torch.cat([group.mean().repeat(n) for group, n in zip(split_rewards, sample_num)])
            std_grouped_rewards = torch.cat([group.std().repeat(n) for group, n in zip(split_rewards, sample_num)])
            
            for i in range(batch_sample):
                logs[str(idx[i].item())] = (torch.split(accs, sample_num)[i].mean().item(), sample_num[i])

            if np.mean(accs_)==1.0 or np.mean(accs_)==0.0:  # no need to train
                continue
            trained_step+=1

            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)

            total_batch_size, num_timesteps, num_class, _, _ = samples["latents"].shape
            num_timesteps -= 1
            
            # 4. reshape matrix for batch train
            samples["advantages"] = advantages
            samples["next_latents"] = samples["latents"][:,1:]
            samples["latents"] = samples["latents"][:,:-1]
            samples["timesteps"] = samples["timesteps"].unsqueeze(0).repeat([total_batch_size,1])
            del samples["x"]
            for inner_epoch in range(self.config.train_num_inner_epochs):
                # shuffle batch
                perm = torch.randperm(total_batch_size, device=self.config.device)
                for key in ["timesteps", "latents", "next_latents", "logps",'advantages','mu','sigma','mask','context']:
                    samples[key] = samples[key][perm]

                # shuffle timesteps
                perms = torch.randperm(num_timesteps, device=self.config.device).unsqueeze(0).repeat([total_batch_size,1])
                for key in ["timesteps", "latents", "next_latents","logps",'mu','sigma']:
                    samples[key] = samples[key][
                        torch.arange(total_batch_size, device=self.config.device)[:, None],
                        perms,
                    ]
                
                original_keys = samples.keys()
                original_values = samples.values()
                # rebatch them as user defined train_batch_size is different from sample_batch_size
                reshaped_values = [v.reshape(-1, batch_size, *v.shape[1:]) for v in original_values]
                # Transpose the list of original values
                transposed_values = zip(*reshaped_values)
                # Create new dictionaries for each row of transposed values
                samples_batched = [dict(zip(original_keys, row_values)) for row_values in transposed_values]
                
                result = self._train_batched_samples(inner_epoch, epoch, global_step, samples_batched)
                result["Reward"] = rewards.mean().item()
                results.append(result)
                
            if trained_step % self.config.save_freq_step == 0:
                self.save_model(root_path, "epoch"+str(epoch+1)+"_"+str(trained_step // self.config.save_freq_step))
        
        # Process training records
        out_log_path = os.path.join(root_path, "epoch"+str(epoch+1)+"_log.json")
        with open(out_log_path, "w") as log_file:
            json.dump(logs, log_file, indent=4)
        avg_acc = sum(v[0] for v in logs.values()) / len(logs)
        
        return results, avg_acc
    
    def _batch_samples(self, batch_size, mask = None, context = None):
        """
        Generate a batch of samples from the model's input distribution and process the resulting data.
        """

        device = self.config.config.device

        # 1. generate sample from dm
        x, latents, logps, timestep, mu, sigma = self.generate_model.sample(batch_size, mask = mask, context = context)
        
        # 2. warp result
        res = {
            "x": x,
            "latents": torch.stack(latents, dim=1),
            "logps": torch.stack(logps, dim=1),
            "timesteps":torch.tensor(timestep).to(self.config.device),
            "mu": torch.stack(mu, dim=1),
            "sigma": torch.stack(sigma, dim=1),
            "mask": mask,
            "context": context
        }
        
        return res
    
    def _generate_samples(self, iterations, batch_size, mask = None, context = None):
        """
        Generate samples from the model

        Args:
            iterations (int): Number of iterations to generate samples for
            batch_size (int): Batch size to use for sampling

        Returns:
            samples (list[dict[str, torch.Tensor]]), prompt_image_pairs (list[list[Any]])
        """
        
        samples = []
        for _ in tq(range(iterations),desc = "Generate samples", unit = "sample",leave=False):
            sample = self._batch_samples(batch_size, mask = mask, context = context)
            samples.append(sample)
            
        ## concat samples
        samples_warped = {}
        for key in samples[0].keys():
            samples_warped[key] = torch.cat([s[key] for s in samples])

        return samples_warped
    
    def compute_rewards_sudoku(self, samples, solutions):
        '''
        use correctness as reward fuction
        '''
        x = samples['x']
        n_samples = x.shape[0]
        x_softmax = F.softmax(x, dim=1)
        pred = torch.argmax(x_softmax, dim=1) + 1

        rewards = []
        accs = []
        
        for i in range(0, n_samples):
            gold = solutions[i]

            if self.config.reward == "board":
                reward = 1.0 if torch.equal(gold, pred[i]) else 0.0
            elif self.config.reward == "rule":
                valid_item = 0
                for row in range(9):
                    valid_item += int(True if torch.equal(torch.unique(pred[i][row]), torch.tensor(range(1, 10)).to(pred[i][row].device)) else False)

                for col in range(9):
                    board_col = torch.stack([pred[i][row,col] for row in range(9)])
                    valid_item += int(True if torch.equal(torch.unique(board_col), torch.tensor(range(1, 10)).to(board_col.device)) else False)

                for idx in range(9):
                    board_cube = torch.stack([pred[i][row,col] for row in range((idx//3)*3,(idx//3)*3+3) for col in range((idx%3)*3,(idx%3)*3+3)])
                    valid_item += int(True if torch.equal(torch.unique(board_cube), torch.tensor(range(1, 10)).to(board_cube.device)) else False)
                
                reward = int(valid_item/27.0)
         
            rewards.append(reward)
            accs.append(1 if reward == 1.0 else 0)
        
        print("Rewards:", np.mean(rewards))
        
        return rewards, accs
    
    def compute_rewards_maze(self, samples, solutions):
        '''
        use correctness as reward fuction
        '''
        x = samples['x']
        n_samples = x.shape[0]
        pred = torch.argmax(x, dim=1)
        gold = torch.argmax(solutions, dim=1)

        rewards = []
        
        for i in range(0, n_samples):
            reward = 1 if torch.equal(pred[i], gold[i]) else 0
            rewards.append(reward)

        print("Rewards:", np.mean(rewards))
        
        return rewards, rewards
    
    def compute_rewards_grid(self, samples, solutions):
        '''
        use correctness as reward fuction
        '''
        x = samples['x']
        n_samples = x.shape[0]
        pred = torch.sign(x)

        rewards = []
        
        for i in range(0, n_samples):
            reward = 1 if torch.equal(pred[i], solutions[i]) else 0
            rewards.append(reward)
        
        print("Rewards:", np.mean(rewards))
        
        return rewards, rewards
    
    def compute_rewards_sushi(self, samples, solutions):
        '''
        use correctness as reward fuction
        '''
        x = samples['x']
        n_samples = x.shape[0]
        pred = (torch.argmax(x, dim=1) + 1).squeeze(1)
        
        rewards = []
        accs = []
        
        for i in range(0, n_samples):
            acc = 1 if torch.equal(pred[i], solutions[i]) else 0
            consistent = 1 if torch.equal(torch.unique(pred[i]), torch.tensor(range(1, 11)).to(pred[i].device)) else 0
            reward = acc*0.5+consistent*0.5
            accs.append(acc)
            rewards.append(reward)
            
        print("Rewards:", np.mean(rewards), " Accs: ", np.mean(accs))
        return rewards, accs
    
    def compute_rewards_warcraft(self, samples, solutions):
        '''
        use correctness as reward fuction
        '''
        def is_connected(grid):
            H, W = grid.shape
            visited = torch.zeros_like(grid, dtype=torch.bool)
            
            directions = [  # 8 directions
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]
            
            def dfs(x, y):
                if not (0 <= x < H and 0 <= y < W):
                    return
                if visited[x, y] or grid[x, y] != 1:
                    return
                visited[x, y] = True
                for dx, dy in directions:
                    dfs(x + dx, y + dy)

            if grid[0, 0] != 1 or grid[11, 11] != 1:
                return False

            dfs(0, 0)
            return visited[11, 11].item()
        
        x = samples['x']
        n_samples = x.shape[0]
        pred = torch.sign(x[:,1])
        pred_path = (torch.sign(x[:,1])+1)/2

        rewards = []
        accs = []
        
        for i in range(0, n_samples):
            consistent = 1 if is_connected((pred[i].int()+1)//2) else 0
            cost = (x[i,0] * pred_path[i]).sum().item()
            acc = 1 if torch.equal(pred[i], solutions[i]) else int((consistent==1 and torch.equal(x[i,0] * pred[i].sum().item(), x[i,0] * solutions[i].sum().item())))
            reward = acc
            rewards.append(reward)
            accs.append(acc)
            
        print("Rewards:", np.mean(rewards), "Accs:", np.mean(accs))
        return rewards, accs
    
    def _train_batched_samples(self, inner_epoch, epoch, global_step, batched_samples):
        """
        Train on a batch of samples. Main training segment

        Args:
            inner_epoch (int): The current inner epoch
            epoch (int): The current epoch
            global_step (int): The current global step
            batched_samples (list[dict[str, torch.Tensor]]): The batched samples to train on

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.

        Returns:
            global_step (int): The updated global step
        """
        info = defaultdict(list)

        self.T = self.config.num_train_timesteps
        for _i, sample in tq(enumerate(batched_samples),desc= "Training", unit="Batch",leave=False):
            for j in tq(range(self.T), desc= "Training Batch", unit="timesteps",leave=False):
                loss, approx_kl, clipfrac, ratio = self.calculate_loss(
                        sample["latents"][:, j],
                        sample["timesteps"][:, j],
                        sample["next_latents"][:, j],
                        sample["logps"][:, j],
                        sample["advantages"],
                        sample["mu"][:, j],
                        sample["sigma"][:, j],
                        sample["mask"],
                        sample["context"],
                    )
                
                info["approx_kl"].append(approx_kl.item())
                info["clipfrac"].append(clipfrac.item())
                info["loss"].append(loss.item())
                info["ratio"].append(ratio.item())
                loss = loss / self.config.num_train_timesteps
                loss.backward()
            clip_grad_norm_(self.generate_model.parameters(),max_norm=1)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        result = {}
        result["KL"] = np.mean(np.array(info["approx_kl"]))
        result["ClipFrac"] = np.mean(np.array(info["clipfrac"]))
        result["Loss"] = np.mean(np.array(info["loss"]))
        result["ratio"] = np.mean(np.array(info["ratio"]))
        result["GlobalStep"] = global_step + 1
        print("loss:",result["Loss"],"ClipFrac:",result["ClipFrac"], "ratio:",result["ratio"])

        return result
    
    def _setup_optimizer(self, trainable_layers_parameters):
        optimizer_cls = torch.optim.AdamW
        return optimizer_cls(
            trainable_layers_parameters,
            lr=self.config.train_learning_rate,
            betas=(self.config.train_adam_beta1, self.config.train_adam_beta2),
            weight_decay=self.config.train_adam_weight_decay,
            eps=self.config.train_adam_epsilon,
        )
        
    def calculate_loss(self, latents, timesteps, next_latents, log_prob_old, advantages, mu_old, sigma_old, mask = None, context = None):
        """
        Calculate the loss for a batch of an unpacked sample

        Args:
            latents (torch.Tensor):
                The latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            timesteps (torch.Tensor):
                The timesteps sampled from the diffusion model, shape: [batch_size]
            next_latents (torch.Tensor):
                The next latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            log_prob (torch.Tensor):
                The log probabilities of the latents, shape: [batch_size]
            advantages (torch.Tensor):
                The advantages of the latents, shape: [batch_size]
            context (torch.Tensor):
                The embedding of context.
        Returns:
            loss (torch.Tensor), approx_kl (torch.Tensor), clipfrac (torch.Tensor)
            (all of these are of shape (1,))
        """
        latents, _, log_prob_current, mu_current, sigma_current = self.generate_model.p_sample(latents, timesteps[0], mask=mask, context=context, prev_sample=next_latents)
        
        ## log_prob is old latents in new policy
        # compute the log prob of next_latents given latents under the current model
        advantages = torch.clamp(
            advantages,
            -self.config.train_adv_clip_max,
            self.config.train_adv_clip_max,
        )

        dif_logp = (log_prob_current - log_prob_old)
        
        ratio = torch.exp(dif_logp)

        approx_kl = 0.5 * torch.mean((log_prob_current - log_prob_old) ** 2)

        loss = self.loss(advantages, self.config.train_clip_range, ratio)
        
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float())
    
        return loss, approx_kl, clipfrac, ratio.mean()
    
    def loss(
        self,
        advantages: torch.Tensor,
        clip_range: float,
        ratio: torch.Tensor,
    ):
        unclipped_loss =    -1.0 * advantages * ratio
        clipped_loss =   -1.0  * advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        
        return torch.mean(torch.maximum(unclipped_loss, clipped_loss))
