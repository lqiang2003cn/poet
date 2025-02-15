# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
logger = logging.getLogger(__name__)
import numpy as np
from poet_distributed.es import ESOptimizer, initialize_worker
from collections import OrderedDict
from poet_distributed.niches.box2d.env import Env_config
from poet_distributed.reproduce_ops import Reproducer
from poet_distributed.novelty import compute_novelty_vs_archive
import json
from poet_distributed.niches.box2d.env_model import EnvModel, get_env_neural_output
from poet_distributed.niches.box2d.env_model import env_model_config
from poet_distributed.noise_module import noise
import uuid

#env:the encodings of the env,i.e., a set of parameters;  niche:the actual env created based on the env parameters
def construct_niche_fns_from_env(args, env, seed):
    def niche_wrapper(configs, seed):  # force python to make a new lexical scope
        def make_niche():
            from poet_distributed.niches import Box2DNiche
            return Box2DNiche(env_configs=configs,
                            seed=seed,
                            init=args.init,
                            recordVideo=args.recordVideo,
                            render_mode=args.render_mode,
                            stochastic=args.stochastic)

        return make_niche

    niche_name = env.name
    configs = (env,)

    return niche_name, niche_wrapper(list(configs), seed)


class MultualESOptimizer:
    #create an
    def __init__(self, args, engines, scheduler, client):
        #initializations about ipp
        self.args = args
        self.engines = engines
        self.engines.block = True
        self.scheduler = scheduler
        self.client = client
        self.optimizers = OrderedDict()
        self.init = 'random'

    def get_random_model_params(self, stdev=0.1):
        return np.random.uniform(0, 1, 2804)  # randn return samples of "standard normal" distribution

    def get_random_env_params(self, stdev=0.1):
        return np.random.uniform(0, 1, 290701) # randn return samples of "standard normal" distribution

    def create_optimizer(self,env_name, seed, created_at=0, model_params=None, is_candidate=False,env_param=None):

        if model_params is not None:
            theta = np.array(model_params)
        else:
            theta=self.get_random_model_params()

        if env_param is None:
            env_param = self.get_random_env_params()

        # after creating the theta, generate env config by neural network
        env_config = get_env_neural_output(theta,env_param,env_name)
        #the optim_id comes from the env.name
        optim_id, niche_fn = construct_niche_fns_from_env(args=self.args, env=env_config, seed=seed)

        assert optim_id not in self.optimizers.keys()

        return ESOptimizer(
            optim_id=optim_id,
            engines=self.engines,
            scheduler=self.scheduler,
            theta=theta,
            env_param=env_param,
            make_niche=niche_fn,
            learning_rate=self.args.learning_rate,
            lr_decay=self.args.lr_decay,
            lr_limit=self.args.lr_limit,
            batches_per_chunk=self.args.batches_per_chunk,
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.eval_batch_size,
            eval_batches_per_step=self.args.eval_batches_per_step,
            l2_coeff=self.args.l2_coeff,
            noise_std=self.args.noise_std,
            noise_decay=self.args.noise_decay,
            normalize_grads_by_noise_std=self.args.normalize_grads_by_noise_std,
            returns_normalization=self.args.returns_normalization,
            noise_limit=self.args.noise_limit,
            log_file=self.args.log_file,
            created_at=created_at,
            is_candidate=is_candidate,
            env_config=env_config)

    #model_params is the parameter of an Agent:each Optimizer corresponds to an Env and Agent
    def add_optimizer(self,env_name, seed, created_at=0, model_params=None,env_param=None):
        o = self.create_optimizer(env_name,seed, created_at, model_params=model_params,env_param=env_param)
        optim_id = o.optim_id
        self.optimizers[optim_id] = o

    def delete_optimizer(self, optim_id):
        assert optim_id in self.optimizers.keys()
        #assume optim_id == env_id for single_env niches
        o = self.optimizers.pop(optim_id)
        del o

    def clean_up_ipyparallel(self):
        logger.debug('Clean up ipyparallel ...')
        #self.client.purge_everything()
        #self.client.purge_results("all")
        #self.client.purge_local_results("all")
        self.client.results.clear()
        self.client.metadata.clear()
        self.client._futures.clear()
        self.client._output_futures.clear()

        self.client.purge_hub_results("all")
        self.client.history = []
        self.client.session.digest_history.clear()

        self.engines.results.clear()
        self.scheduler.results.clear()
        #self.client.results.clear()
        #self.client.metadata.clear()




    def transfer(self, propose_with_adam, checkpointing, reset_optimizer):
        logger.info('Computing direct transfers...')
        for source_optim in self.optimizers.values():
            source_tasks = []
            for target_optim in [o for o in self.optimizers.values()
                                    if o is not source_optim]:
                task = target_optim.start_theta_eval(
                    source_optim.theta)
                source_tasks.append((task, target_optim))

            for task, target_optim in source_tasks:
                stats = target_optim.get_theta_eval(task)

                target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                    source_optim_theta=source_optim.theta,
                    stats=stats, keyword='theta')

        logger.info('Computing proposal transfers...')
        for source_optim in self.optimizers.values():
            source_tasks = []
            for target_optim in [o for o in self.optimizers.values()
                                    if o is not source_optim]:
                task = target_optim.start_step(source_optim.theta)
                source_tasks.append((task, target_optim))

            for task, target_optim in source_tasks:
                proposed_theta, _ = target_optim.get_step(
                    task, propose_with_adam=propose_with_adam, propose_only=True)

                proposal_eval_task = target_optim.start_theta_eval(proposed_theta)
                proposal_eval_stats = target_optim.get_theta_eval(proposal_eval_task)

                target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                    source_optim_theta=proposed_theta,
                    stats=proposal_eval_stats, keyword='proposal')

        logger.info('Considering transfers...')
        for o in self.optimizers.values():
            o.pick_proposal(checkpointing, reset_optimizer)

        self.clean_up_ipyparallel()

    # check if any env's self_evals is above the reproduce threshold,
    # if one does, then add it to the repro_candidates list;
    # plus: this function has nothing to do with the iteration and delete_candidates,
    def check_optimizer_status(self, iteration):
        '''
            return two lists
        '''
        logger.info("health_check")
        repro_candidates, delete_candidates = [], []
        for optim_id in self.env_registry.keys():#find the already registered env
            o = self.optimizers[optim_id]
            logger.info("niche {} created at {} start_score {} current_self_evals {}".format(
                optim_id, o.created_at, o.start_score, o.self_evals))
            if o.self_evals >= self.args.repro_threshold:
                repro_candidates.append(optim_id)

        logger.debug("candidates to reproduce")
        logger.debug(repro_candidates)
        logger.debug("candidates to delete")
        logger.debug(delete_candidates)

        return repro_candidates, delete_candidates


    def pass_dedup(self, env_config):
        if env_config.name in self.env_registry.keys():
            logger.debug("active env already. reject!")
            return False
        else:
            return True

    def pass_mc(self, score):
        if score < self.args.mc_lower or score > self.args.mc_upper:
            return False
        else:
            return True

    def get_new_env(self, list_repro):

        optim_id = self.env_reproducer.pick(list_repro)
        assert optim_id in self.optimizers.keys()
        assert optim_id in self.env_registry.keys()
        parent = self.env_registry[optim_id]
        child_env_config = self.env_reproducer.mutate(parent)

        logger.info("we pick to mutate: {} and we got {} back".format(optim_id, child_env_config.name))
        logger.debug("parent")
        logger.debug(parent)
        logger.debug("child")
        logger.debug(child_env_config)

        seed = np.random.randint(1000000)
        return child_env_config, seed, optim_id

    def get_child_list(self, parent_list, max_children):
        child_list = []

        mutation_trial = 0
        while mutation_trial < max_children:
            new_env_config, seed, parent_optim_id = self.get_new_env(parent_list)
            mutation_trial += 1
            if self.pass_dedup(new_env_config):
                o = self.create_optimizer(new_env_config, seed, is_candidate=True)
                score = o.evaluate_theta(self.optimizers[parent_optim_id].theta)
                del o
                if self.pass_mc(score):
                    novelty_score = compute_novelty_vs_archive(self.env_archive, new_env_config, k=5)
                    logger.debug("{} passed mc, novelty score {}".format(score, novelty_score))
                    child_list.append((new_env_config, seed, parent_optim_id, novelty_score))

        #sort child list according to novelty for high to low
        child_list = sorted(child_list,key=lambda x: x[3], reverse=True)
        return child_list


    def adjust_envs_niches(self, iteration, steps_before_adjust, max_num_envs=None, max_children=8, max_admitted=1):
        if iteration > 0 and iteration % steps_before_adjust == 0:
        #if iteration % steps_before_adjust == 0:
            list_repro, list_delete = self.check_optimizer_status(iteration)
            #if no env is qualified for reproducing, then do nothing and just return
            if len(list_repro) == 0:
                return

            logger.info("list of niches to reproduce")
            logger.info(list_repro)
            logger.info("list of niches to delete")
            logger.info(list_delete)

            child_list = self.get_child_list(list_repro, max_children)

            if child_list == None or len(child_list) == 0:
                logger.info("mutation to reproduce env FAILED!!!")
                return
            #print(child_list)
            admitted = 0
            for child in child_list:
                new_env_config, seed, _, _ = child
                # targeted transfer
                o = self.create_optimizer(new_env_config, seed, is_candidate=True)
                score_child, theta_child = o.evaluate_transfer(self.optimizers)
                del o
                if self.pass_mc(score_child):  # check mc
                    self.add_optimizer(env=new_env_config, seed=seed, created_at=iteration, model_params=np.array(theta_child))
                    admitted += 1
                    if admitted >= max_admitted:
                        break

            if max_num_envs and len(self.optimizers) > max_num_envs:
                num_removals = len(self.optimizers) - max_num_envs
                self.remove_oldest(num_removals)

    def remove_oldest(self, num_removals):
        list_delete = []
        for optim_id in self.env_registry.keys():
            if len(list_delete) < num_removals:
                list_delete.append(optim_id)
            else:
                break

        for optim_id in list_delete:
            self.delete_optimizer(optim_id)

    def optimize(self, iterations=200,
                 steps_before_transfer=25,#
                 propose_with_adam=False,
                 checkpointing=False,
                 reset_optimizer=True):
        #define the number of iteration
        for iteration in range(iterations):
            #adjust the env:randomly generate more envs
            self.adjust_envs_niches(
                iteration,
                self.args.adjust_interval * steps_before_transfer,
                max_num_envs=self.args.max_num_envs)

            for o in self.optimizers.values():
                o.clean_dicts_before_iter()

            #
            self.ind_es_step(iteration=iteration)

            if len(self.optimizers) > 1 and iteration % steps_before_transfer == 0:
                self.transfer(
                    propose_with_adam=propose_with_adam,
                    checkpointing=checkpointing,
                    reset_optimizer=reset_optimizer)

            if iteration % steps_before_transfer == 0:
                for o in self.optimizers.values():
                    o.save_to_logger(iteration)


    #mutually optimize
    def mutual_opt_step(self, iteration,args=None):
        env_num_per_iter = args.batches_per_chunk
        random_state = np.random.RandomState()
        #for each iteration, create 16 new variants of the env and eval agents in them
        env_seeds = np.random.randint(np.int32(2 ** 31 - 1), size=env_num_per_iter)
        new_env_optim_ids =['theEnv']
        env_noise_indexs = []
        for i in range(0,env_num_per_iter):
            theEnvOpt = self.optimizers['theEnv']
            #print(theEnvOpt.env_param)
            random_state.seed(env_seeds[i])
            #random noise indexs for the env_params
            noise_ind = noise.sample_index(random_state, len(theEnvOpt.env_param))
            #call it twice because its pos and neg
            env_noise_indexs.append(noise_ind)
            env_noise_indexs.append(noise_ind)
            #print('noise index is:',noise_inds)
            #generate new env_params based on the noise_inds
            pos_env_param =theEnvOpt.env_param + theEnvOpt.noise_std * noise.get(noise_ind, len(theEnvOpt.env_param))
            seed = np.random.randint(10000000)
            env_name = str(i)+'0_mut_pos_theEnv_'+ str(uuid.uuid1())
            new_env_optim_ids.append(env_name)
            #use the current theta as input,
            self.add_optimizer(env_name=env_name,seed=seed,model_params=theEnvOpt.theta,env_param=pos_env_param)

            neg_env_param =theEnvOpt.env_param - theEnvOpt.noise_std * noise.get(noise_ind, len(theEnvOpt.env_param))
            seed = np.random.randint(10000000)
            env_name = str(i)+'1_mut_neg_theEnv_' + str(uuid.uuid1())
            new_env_optim_ids.append(env_name)
            # use the current theta as input,
            self.add_optimizer(env_name=env_name, seed=seed, model_params=theEnvOpt.theta, env_param=neg_env_param)

        #generate thetas for the population
        new_theta_list =[]
        agent_num_per_iter = args.batches_per_chunk
        agent_noise_indexs = []
        theta_seeds = np.random.randint(np.int32(2 ** 31 - 1), size=agent_num_per_iter)
        for i in range(0, agent_num_per_iter):
            random_state.seed(env_seeds[i])
            # random noise indexs for the env_params
            noise_ind = noise.sample_index(random_state, len(theEnvOpt.theta))
            agent_noise_indexs.append(noise_ind)
            agent_noise_indexs.append(noise_ind)
            new_pos_theta = theEnvOpt.theta + theEnvOpt.noise_std * noise.get(noise_ind, len(theEnvOpt.theta))
            new_theta_list.append(new_pos_theta)
            new_neg_theta = theEnvOpt.theta - theEnvOpt.noise_std * noise.get(noise_ind, len(theEnvOpt.theta))
            new_theta_list.append(new_neg_theta)

        result_matrix = np.zeros((env_num_per_iter*2,env_num_per_iter*2))
        row_num=0
        for o in self.optimizers.values():
            if o.optim_id != 'theEnv':
                col_num=0
                step_results =[]
                for t in new_theta_list:
                    step_results.append(o.start_mutual_opt_step(t))
                    col_num += 1
                row_num += 1
        result_matrix = np.array([task.get() for task in step_results]).reshape(result_matrix.shape)

        #print(result_matrix)
        updated_theta,updated_env=self.update_by_matrix(result_matrix,agent_noise_indexs,env_noise_indexs,theEnvOpt)

        for opt in new_env_optim_ids:
            self.delete_optimizer(opt)

        #eval the updated params
        seed = np.random.randint(10000000)
        env_name = 'theEnv'
        self.add_optimizer(env_name=env_name, seed=seed, model_params=updated_theta, env_param=updated_env)
        theEnv = self.optimizers['theEnv']
        step_result = theEnv.start_mutual_opt_step(t)
        print('the updated env roughness is:', theEnv.env_config.ground_roughness)
        print('the updated eval reward is:',step_result.get().result)


        self.clean_up_ipyparallel()

    #choose the max theta, choose the middle env
    def update_by_matrix(self,matrix,agent_noise_indexs,env_noise_indexs,theEnvOpt):
        shape = matrix.shape
        row_num = shape[0]
        col_num = shape[1]
        #calculate agent gradient
        sum_of_cols = np.sum(matrix,axis=0)
        max_sum_ind = np.argmax(sum_of_cols)
        agent_grad = noise.get(agent_noise_indexs[max_sum_ind], len(theEnvOpt.theta))
        #agent_grad = agent_grad/theEnvOpt.noise_std

        #calculate env gradient
        row_sum = np.sum(matrix, axis=1)
        row_order_index = np.argsort(row_sum)
        middle_index=row_order_index[int(row_num/2)]
        env_grad = noise.get(env_noise_indexs[middle_index], len(theEnvOpt.env_param))
        #env_grad = env_grad/theEnvOpt.noise_std

        #theta is the updated theta
        _,updated_theta = theEnvOpt.optimizer.update(theEnvOpt.theta, -agent_grad + theEnvOpt.l2_coeff * theEnvOpt.theta)  # theta:the original theta
        _,updated_env_param = theEnvOpt.env_param_optimizer.update(theEnvOpt.env_param, -env_grad + theEnvOpt.l2_coeff * theEnvOpt.env_param)  # theta:the original theta

        theEnvOpt.optimizer.stepsize = max(theEnvOpt.optimizer.stepsize * theEnvOpt.lr_decay, theEnvOpt.lr_limit)
        theEnvOpt.noise_std = max(theEnvOpt.noise_std * theEnvOpt.noise_decay, theEnvOpt.noise_limit)

        return updated_theta,updated_env_param

    #mutually optimization:optimize the agent and env simultaneously, and get the balanced result
    def optimize_mutually(self,iterations,args=None):
        for iteration in range(iterations):
            self.mutual_opt_step(iteration=iteration,args=args)
