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
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) which is the type of initialization
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
