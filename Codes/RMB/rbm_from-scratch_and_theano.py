from __future__ import print_function

import timeit

try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy

import theano
import theano.tensor as T
import os

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import tile_raster_images
from logistic_sgd import load_data




# First we construct RBM Class 

class RBM(object):

# This init function is in fact RBM's constructor

    def __init__(
        self,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None,
        numpy_rng=None,
        theano_rng=None
    ):
    
    #Analogous to this in java
    
    self.n_visible = n_visible
    self.n_hidden = n_hidden

    if(numpy_rng is None):
            # create a number generator
            numpy_rng = numpy.random.RandomState(1254)

        if(theano_rng is None):
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if(W is None):
            # W is initialized with `initial_W` which is uniformely sampled from -4*sqrt(6./(n_visible+n_hidden)) which is the type of initialization
            
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases (yes! we are using theano!)
            
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if(hbias is None):
            # create shared variable for hidden units bias, initialized to zeros
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if(vbias is None):
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if(not input):
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        self.params = [self.W, self.hbias, self.vbias]
        
        #Next we construct symbolic graphs given by p(v/h) and p(h/v) together
        
        def propup(self, vis):
            
            #Forward Prop from visible to hidden layer
            
            pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
            return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
        
         def sample_h_given_v(self, v0_sample):
                
                # Look at the code below carefully
                pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
                h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
                return [pre_sigmoid_h1, h1_mean, h1_sample]
            
         def propdown(self, hid):
            
            #Backward propagation (not to be confused with it being same as in supervised learning. We are semi-supervising here!)
            
                pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
                return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
            
        def sample_v_given_h(self, h0_sample):
            
            pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
            v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
            return [pre_sigmoid_v1, v1_mean, v1_sample]
        
        #Defining symbolic graphs for gibbs sampling step
        
        def gibbs_hvh(self, h0_sample):
        ''' one step of Gibbs sampling, starting from the hidden state'''
            pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
            pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
            return [pre_sigmoid_v1, v1_mean, v1_sample,
                    pre_sigmoid_h1, h1_mean, h1_sample]
    
        def gibbs_vhv(self, v0_sample):
        ''' one step of Gibbs sampling, starting from the visible state'''
            pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
            pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
            return [pre_sigmoid_h1, h1_mean, h1_sample,
                    pre_sigmoid_v1, v1_mean, v1_sample]
        
        """Whenever you compile a Theano function, the computational graph that you pass as input gets optimized for speed and stability. This is done by changing several parts of the subgraphs with others."""
        
        #Let us now calculate the free energy of the samples (RBM is a subclass of EBM after all!)
        
        def free_energy(self, v_sample):
           
            wx_b = T.dot(v_sample, self.W) + self.hbias
            vbias_term = T.dot(v_sample, self.vbias)
            hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
            return -hidden_term - vbias_term
        
        #Generating symbolic gradients for CD-K AND PCD-K updates
        
        
        def get_cost_updates(self, lr=0.1, persistent=None, k=1):
            pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
            
            # Only necessary for PCD
            
            if (persistent is None):
                chain_start = ph_sample
            else:
                chain_start = persistent
                
        #Now we use scan operation from theano. Please refer to its ducumentation
        
            (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
            ) = theano.scan(
                self.gibbs_hvh,
                # the None are place holders, saying that
                # chain_start is the initial state corresponding to the
                # 6th output
                outputs_info=[None, None, None, None, None, chain_start],
                n_steps=k,
                name="gibbs_hvh"
            )
            
            #Now determine gradients on RBM parameters
            
            chain_end = nv_samples[-1]

            cost = T.mean(self.free_energy(self.input)) - T.mean(
                self.free_energy(chain_end))
            # We must not compute the gradient through the gibbs sampling
            gparams = T.grad(cost, self.params, consider_constant=[chain_end])
            
            #Now we update the parameters
            
            for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
                updates[param] = param - gparam * T.cast(
                    lr,
                    dtype=theano.config.floatX
                )
            if (persistent):
                # Note that this works only if persistent is a shared variable
                updates[persistent] = nh_samples[-1]
                # pseudo-likelihood is a better proxy for PCD
                monitoring_cost = self.get_pseudo_likelihood_cost(updates)
            else:
                # reconstruction cross-entropy is a better proxy for CD
                monitoring_cost = self.get_reconstruction_cost(updates,
                                                               pre_sigmoid_nvs[-1])

            return monitoring_cost, updates
        
        
        #Stochastic approximation to the pseudo-likelihood
        
         def get_pseudo_likelihood_cost(self, updates):
        
            bit_i_idx = theano.shared(value=0, name='bit_i_idx')

            #rounding to nearest integer
            xi = T.round(self.input)

            # calculate free energy for the given bit configuration
            fe_xi = self.free_energy(xi)

            
            xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

            # calculate free energy with bit flipped
            fe_xi_flip = self.free_energy(xi_flip)

            
            cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                                fe_xi)))

            # increment bit_i_idx % number as part of updates
            updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

            return cost
        
        #We train the RBM using PCD
        
        train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in range(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))
    
    #Sampling
    
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        numpy.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )
    
    plot_every = 1000
    
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every,
        name="gibbs_vhv"
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = numpy.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1),
        dtype='uint8'
    )
    for idx in range(n_samples):
        
        vis_mf, vis_sample = sample_fn()
        print(' ... plotting sample %d' % idx)
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    # construct image
    image = Image.fromarray(image_data)
    image.save('samples.png')

            

