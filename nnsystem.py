import tensorflow as tf
import numpy as np
import os
import pickle
from . import utils
import time
import yaml
from gantools.model import discriminator, generator
import functools
# import sys
# sys.path.append("..")
# from .. import data
from gantools.data import fmap


class NNSystem(object):
    """A system to handle Neural Network"""
    def default_params(self):
        d_param = dict()
        d_param['optimization'] = dict()
        d_param['optimization']['learning_rate'] = 1e-4
        d_param['optimization']['batch_size'] = 8
        d_param['optimization']['epoch'] = 100
        d_param['optimization']['batch_size'] = 8

        d_param['net'] = dict()

        d_param['save_dir'] = './checkpoints/'
        d_param['summary_dir'] = './summaries/'
        d_param['summary_every'] = 200
        d_param['print_every'] = 100
        d_param['save_every'] = 10000
        return d_param

    def compute_metrics(self, real, fake):
        from gantools.metric import ganlist
        metric_list = ganlist.cosmo_metric_list()
        d = []
        for metr in metric_list:
            d.append(metr(fake, real))
        # plt.plot(range(0, len(d)), d)
        return (np.mean(np.array(d)), *d)

    def single_metric(self, real, fake):
        return self.compute_metrics(real, fake)[0]


    def __init__(self, model, params={}, name=None, debug_mode=False):
        """Build the TF graph."""
        self._debug_mode=debug_mode
        if self._debug_mode:
            print('User parameters NNSystem...')
            print(yaml.dump(params))

        self._params = utils.arg_helper(params, self.default_params())
        if self._debug_mode:
            print('\nParameters used for the NNSystem..')
            print(yaml.dump(self._params))
        tf.reset_default_graph()
        if name:
            self._net = model(self.params['net'], name=name)
        else:
            self._net = model(self.params['net'])
        self._name = self._net.name
        self._add_optimizer()
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        utils.show_all_variables()
        self._summaries = tf.summary.merge(tf.get_collection("train"))
        self.N = 2000  # Number of samples

        shift = 3
        c = 40000
        self.forward = functools.partial(fmap.stat_forward, shift=shift, c=c)
        self.backward = functools.partial(fmap.stat_backward, shift=shift, c=c)

        self.plc_float = tf.placeholder(tf.float32)
        self.plc_float_r = tf.placeholder(tf.float32)
        self.disc_loss_calc2 = tf.reduce_mean(self.plc_float_r - self.plc_float)

        with tf.variable_scope('worst_calc', reuse=tf.AUTO_REUSE):
            new_opt = tf.train.RMSPropOptimizer(learning_rate=3e-5)
            self.df = discriminator(self._net.X_fake, self._net.params['discriminator'], z=None, reuse=tf.AUTO_REUSE, scope="TMPdisc")
            self.dr = discriminator(self._net.X_real, self._net.params['discriminator'], z=None, reuse=tf.AUTO_REUSE, scope="TMPdisc")
            disc_loss_worst = -tf.reduce_mean(self.dr - self.df)
            t_vars = tf.global_variables()
            d_vars_worst = [var for var in t_vars if 'TMPdisc' in var.name]
            # list_fake = [self._net.X_fake]
            # list_real= [self._net.X_real]
            # gamma = self._net.params['gamma_gp']
            # if not gamma:
            #     # I am not sure this part or the code is still useful
            #     t_vars = tf.trainable_variables()
            #     d_vars = [var for var in t_vars if 'TMPdisc' in var.name]
            #     D_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]
            #     D_gp = tf.constant(0, dtype=tf.float32)
            #     print(" [!] Using weight clipping")
            # else:
            #     bs = tf.shape(list_fake[0])[0]
            #     eps = tf.random_uniform(shape=[bs], minval=0, maxval=1)
            #
            #     x_hat = []
            #     for fake, real in zip(list_fake, list_real):
            #         singledim = [1] * (len(fake.shape.as_list()) - 1)
            #         eps = tf.reshape(eps, shape=[bs, *singledim])
            #         x_hat.append(eps * real + (1.0 - eps) * fake)
            #
            #     D_x_hat = discriminator(*x_hat, params=self._net.params['discriminator'], z=None, reuse=True, scope="TMPdisc")
            #
            #     # gradient penalty
            #     gradients = tf.gradients(D_x_hat, x_hat)
            #     norm_gradient_pen = tf.norm(gradients[0], ord=2)
            #     D_gp = gamma * tf.square(norm_gradient_pen - 1.0)
            self.find_worst_d = new_opt.minimize(disc_loss_worst, var_list=d_vars_worst)


        with tf.variable_scope('worst_calc_gen', reuse=tf.AUTO_REUSE):
            new_opt_gen = tf.train.RMSPropOptimizer(learning_rate=3e-5)
            x_w = generator(self._net.z, self._net.params['generator'], reuse=False, scope="TMPgen")
        self.df_w = discriminator(x_w, self._net.params['discriminator'], z=None, reuse=True, scope="discriminator")
        self.dr_w = discriminator(self._net.X_real, self._net.params['discriminator'], z=None, reuse=True, scope="discriminator")

        with tf.variable_scope('worst_calc_gen', reuse=tf.AUTO_REUSE):
            gen_loss_worst = tf.reduce_mean(self.dr_w - self.df_w)
            t_vars = tf.global_variables()
            g_vars_worst = [var for var in t_vars if 'TMPgen' in var.name]
            self.find_worst_g = new_opt_gen.minimize(gen_loss_worst, var_list=g_vars_worst)

        t_vars = tf.global_variables()
        d_init = [var for var in t_vars if 'worst_calc' in var.name]
        self.init_new_vars_op = tf.initialize_variables(d_init)

        curr_to_tmp = []
        t_vars = tf.global_variables()
        d_vars_tmp = [var for var in t_vars if 'TMPdisc' in var.name and 'RMSProp' not in var.name]
        d_vars_0 = [var for var in t_vars if 'discriminator/' in var.name and 'RMSProp' not in var.name]
        g_vars_tmp = [var for var in t_vars if 'TMPgen' in var.name and 'RMSProp' not in var.name]
        g_vars_0 = [var for var in t_vars if 'generator/' in var.name and 'RMSProp' not in var.name]
        for j in range(0, len(d_vars_tmp)):
            print (d_vars_tmp[j])
            curr_to_tmp.append(d_vars_tmp[j].assign(d_vars_0[j]))
        for j in range(0, len(g_vars_tmp)):
            curr_to_tmp.append(g_vars_tmp[j].assign(g_vars_0[j]))

        self.current_to_tmp = tf.group(*curr_to_tmp)

    def _add_optimizer(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_rate = self._params['optimization']['learning_rate']
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self._optimize = optimizer.minimize(self._net.loss)
        tf.summary.scalar("training/loss", self._net.loss, collections=["train"])

    def _get_dict(self, index=None, **kwargs):
        """Return a dictionary with the argument for the architecture."""
        feed_dict = dict()
        for key, value in kwargs.items():
            if value is not None:
                if index:
                    feed_dict[getattr(self._net, key)] = value[index]
                else:
                    feed_dict[getattr(self._net, key)] = value
        return feed_dict

    def train(self, dataset, resume=False):

        n_data = dataset.N
        batch_size = self.params['optimization']['batch_size']
        self._counter = 1
        self._n_epoch = self.params['optimization']['epoch']
        self._total_iter = self._n_epoch * (n_data // batch_size) - 1
        self._n_batch = n_data // batch_size

        self._save_current_step = False

        # Create the save diretory if it does not exist
        os.makedirs(self._params['save_dir'], exist_ok=True)
        run_config = tf.ConfigProto()

        with tf.Session(config=run_config) as self._sess:
            if resume:
                self.load()
            else:
                self._sess.run(tf.global_variables_initializer())
                utils.saferm(self.params['summary_dir'])
                utils.saferm(self.params['save_dir'])

            self._summary_writer = tf.summary.FileWriter(
                self._params['summary_dir'], self._sess.graph)
            try:
                self._epoch = 0
                self._time = dict()
                self._time['start_time'] = time.time()
                self._time['prev_iter_time'] = self._time['start_time']

                print('Start training')
                while self._epoch < self._n_epoch:
                    epoch_loss = 0.
                    for idx, batch in enumerate(
                            dataset.iter(batch_size)):

                        if resume:
                            self._counter = self.params['curr_counter']
                            resume = False
                        else:
                            self._params['curr_counter'] = self._counter
                        feed_dict = self._get_dict(**self._net.batch2dict(batch))
                        curr_loss = self._run_optimization(feed_dict, idx)
                        # epoch_loss += curr_loss

                        if np.mod(self._counter, self.params['print_every']) == 0:
                            # self._print_log(idx, curr_loss, epoch_loss/idx)
                            self._print_log(idx, curr_loss)

                        if np.mod(self._counter, self.params['summary_every']) == 0:
                            self._train_log(feed_dict)

                        if (np.mod(self._counter, self.params['save_every']) == 0) | self._save_current_step:
                            self._save(self._counter)
                            self._save_current_step = False
                        self._counter += 1

                    # epoch_loss /= self._n_batch
                    # print(" - Epoch {}, train loss: {:f}".format(self._epoch, epoch_loss))

                    self._epoch += 1
                print('Training done')
            except KeyboardInterrupt:
                pass
            self._save(self._counter)


    def calculate_metrics(self, dataset, resume=False):

        n_data = dataset.N
        batch_size = self.params['optimization']['batch_size']
        self._n_batch = n_data // batch_size

        # Create the save diretory if it does not exist
        run_config = tf.ConfigProto()

        with tf.Session(config=run_config) as self._sess:
            if resume:
                self.load()
            else:
                self._sess.run(tf.global_variables_initializer())
                utils.saferm(self.params['summary_dir'])
                utils.saferm(self.params['save_dir'])

            # First randomly initialize the new variables for the optimization of the new D_tmp/G_tmp
            self._sess.run(self.init_new_vars_op)
            # Assign the weights to the new D_tmp/G_tmp to be the those of the current D/G
            self._sess.run(self.current_to_tmp)

            # for fixed G, find the worst D_tmp
            id_batches = list(dataset.iter(64))
            for j in range(0, 500):
                batch_curr = id_batches[j]
                feed_dict = self._get_dict(**self._net.batch2dict(batch_curr))
                self._sess.run(self.find_worst_d, feed_dict=feed_dict)
            # calculate the worst minmax
            feed_dict = self._get_dict(**self._net.batch2dict(batch_curr))
            df_final = self._sess.run(self.df, feed_dict=feed_dict) # here you need to feed z
            dr_final = self._sess.run(self.dr, feed_dict=feed_dict)
            worst_minmax = self._sess.run(self.disc_loss_calc2, feed_dict={self.plc_float: df_final, self.plc_float_r: dr_final})
            # for fixed D, find the worst G_tmp
            for j in range(0, 500):
                batch_curr = id_batches[j]
                feed_dict = self._get_dict(**self._net.batch2dict(batch_curr))
                self._sess.run(self.find_worst_g, feed_dict=feed_dict)
            # calculate the worst maxmin
            feed_dict = self._get_dict(**self._net.batch2dict(batch_curr))
            df_final = self._sess.run(self.df_w, feed_dict=feed_dict)
            dr_final = self._sess.run(self.dr_w, feed_dict=feed_dict)
            worst_maxmin = self._sess.run(self.disc_loss_calc2, feed_dict={self.plc_float: df_final, self.plc_float_r: dr_final})

            # report the metrics
            dualitygap_score = worst_minmax - worst_maxmin
            print ('The duality gap score is: ')
            print ('{0:.16f}'.format(dualitygap_score))
            print ('The minmax is: ')
            print ('{0:.16f}'.format(worst_minmax))
            # # Before computing the statistics, we need to invert the mapping
            raw_images = self.backward(dataset.get_samples(dataset.N))
            gen_sample = np.squeeze(self.generate(N=self.N, sess=self._sess))
            gen_sample_raw = self.backward(gen_sample)
            metric_number = self.single_metric(raw_images, gen_sample_raw)
            print("The global metric is {}".format(metric_number))


    def _run_optimization(self, feed_dict, idx):
            if idx==0:
                self._epoch_loss = 0
            curr_loss = self._sess.run([self.net.loss, self._optimize], feed_dict)[0]
            self._epoch_loss += curr_loss
            return curr_loss

    def _print_log(self, idx, curr_loss):
        current_time = time.time()
        batch_size = self.params['optimization']['batch_size']
        print("    * Epoch: [{:2d}] [{:4d}/{:4d}] "
              "Counter:{:2d}\t"
              "({:4.1f} min\t"
              "{:4.3f} examples/sec\t"
              "{:4.2f} sec/batch)\t"
              "Batch loss:{:.8f}\t"
              "Mean loss:{:.8f}\t".format(
              self._epoch, 
              idx+1, 
              self._n_batch,
              self._counter,
              (current_time - self._time['start_time']) / 60,
              self._params['print_every'] * batch_size / (current_time - self._time['prev_iter_time']),
              (current_time - self._time['prev_iter_time']) / self._params['print_every'],
              curr_loss,
              self._epoch_loss/(idx+1)))
        self._time['prev_iter_time'] = current_time

    def _train_log(self, feed_dict):
        summary = self._sess.run(self._summaries, feed_dict=feed_dict)
        self._summary_writer.add_summary(summary, self._counter)


    def _save(self, step):
        if not os.path.exists(self.params['save_dir']):
            os.makedirs(self.params['save_dir'])

        self._saver.save(
            self._sess,
            os.path.join(self.params['save_dir'], self._net.name),
            global_step=step)
        self._save_obj()
        print('Model saved!')

    def _save_obj(self):
        # Saving the objects:
        if not os.path.exists(self.params['save_dir']):
            os.makedirs(self.params['save_dir'], exist_ok=True)

        path_param = os.path.join(self.params['save_dir'], 'params.pkl')
        with open(path_param, 'wb') as f:
            pickle.dump(self.params, f)

    def load(self, sess=None, checkpoint=None):
        '''
        Given checkpoint, load the model.
        By default, load the latest model saved.
        '''
        if sess:
            self._sess = sess
        elif self._sess is None:
            raise ValueError("Session not available at the time of loading model!")

        if checkpoint:
            file_name = os.path.join(
                self._savedir,
                self.net.name+ '-' + str(checkpoint))
        else:
            file_name = None

        print(" [*] Reading checkpoints...")
        if file_name:
            self._saver.restore(self._sess, file_name)
            print ("Found checkpt")
            return True

        checkpoint_dir = self.params['save_dir']
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
            return True

        return False

    def outputs(self, checkpoint=None, **kwargs):
        outputs = self._net.outputs

        with tf.Session() as self._sess:

            if self.load(checkpoint=checkpoint):
                print("Model loaded.")
            else:
                raise ValueError("Unable to load the model")

            self._sess.run([tf.local_variables_initializer()])
            feed_dict = self._get_dict(**kwargs)

            return self._sess.run(outputs, feed_dict=feed_dict)

    def loss(self, dataset, checkpoint=None):
        with tf.Session() as self._sess:

            if self.load(checkpoint=checkpoint):
                print("Model loaded.")
            else:
                raise ValueError("Unable to load the model")
            loss = 0
            batch_size = self._params['optimization']['batch_size']
            for idx, batch in enumerate(dataset.iter(batch_size)):
                feed_dict = self._get_dict(**self.net.batch2dict(batch))
                loss += self._sess.run(self.net.loss, feed_dict)
        return loss/idx
    @property
    def params(self):
        return self._params

    @property
    def net(self):
        return self._net
        

class ValidationNNSystem(NNSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validation_loss = tf.placeholder(tf.float32, name='validation_loss')
        tf.summary.scalar("validation/loss", self._validation_loss, collections=["validation"])
        self._summaries_validation = tf.summary.merge(tf.get_collection("validation"))


    def train(self, dataset_train, dataset_validation, resume=False):
        self._validation_dataset = dataset_validation
        super().train(dataset_train, resume=resume)

    def _train_log(self, feed_dict):
        super()._train_log(feed_dict)
        loss = 0
        batch_size = self._params['optimization']['batch_size']
        for idx, batch in enumerate(
            self._validation_dataset.iter(batch_size)):

            feed_dict = self._get_dict(**self._net.batch2dict(batch))
            loss += self._sess.run(self._net.loss, feed_dict)
        loss = loss/idx
        print("Validation loss: {}".format(loss))
        summary = self._sess.run(self._summaries_validation, feed_dict={self._validation_loss: loss})
        self._summary_writer.add_summary(summary, self._counter)
