"""
Methods and classes for data-generating motif and contextual motif methods.
"""

import numpy as np
import scipy
import sklearn
import math
from sklearn import feature_extraction, mixture
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

class Motif:
    """
    A simple class for abstracting motifs. A motif is sequence of distributions.
    For now, I'm assuming very simple initialization: a given, fixed std and mean drawn
    uniformly in some given magnitude range. May need some brushing up.
    """
    def __init__(self, *args, **kwargs):
        self.w = kwargs.get('w', 8)
        self.std = kwargs.get('std', 1)
        self.thresh = kwargs.get('thresh', 10)
        if kwargs.get('mu', None) is not None:
            self.mu = kwargs.get('mu')
        else:
            self.mu = []
            for i in range(self.w):
                self.mu.append(np.random.uniform(-self.thresh, self.thresh))
        self.dists = [scipy.stats.norm(self.mu[i], self.std) for i in range(len(self.mu))]
    
    def __repr__(self):
        return 'mu:%s, std:%s\n' % (self.mu, self.std)
    def sample(self):
        '''
        Draws one sample of the motif.
        '''
        sample = []
        for i in range(self.w):
            sample.append(self.dists[i].rvs())
        return sample
    
    def log_likelihood(self, data):
        """
        returns log likelihood of data given the motif parameters
        """
        loglike = 0
        for i in range(self.w):
            loglike += self.dists[i].logpdf(data[i])
        return loglike
    
class MotifMixtureModel:
    """
    Implements the motif mixture model generative process. A model instance is created by
    initializing the mixing parameter (gamma) and motif paramters (theta). 
    After that signals can be sampled.
    User supplied parameters:
    num_motifs = 1, number of proper motifs (does not include background)
    thresh = 10, motif means are drawn uniformly at random from mu-thresh to mu+thresh
    w = 8, motif length
    std = 1, standard deviation used for all proper motifs
    background_mult = 10, multiplier used for background motif std
    background_ratio = 0.5, weight in gamma devoted to background
    """
    def __init__(self, *args, **kwargs):
        # Initialize theta
        self.num_motifs = kwargs.get('num_motifs', 1) # does not include background motif
        self.thresh = kwargs.get('thresh', 10)
        self.w = kwargs.get('w', 8)
        self.std = kwargs.get('std', 1)
        self.bkg = kwargs.get('background_mult', 10)
        self.bkg_rat = kwargs.get('background_ratio', 0.5)
        motifs = []
        # create normal motifs
        for i in range(self.num_motifs):
            motifs.append(Motif(w=self.w, std=self.std, thresh=self.thresh))
        # create background motif
        motifs.append(Motif(w=self.w, std=self.std*self.bkg, thresh=0))
        self.theta = motifs
        
        # Initialize gamma
        # assuming weak prior of proper motif ratios with preset background ratio
        gamma_proper = np.random.dirichlet(np.ones(self.num_motifs))
        gamma = gamma_proper*(1-self.bkg_rat)
        gamma = np.append(gamma, self.bkg_rat)
        self.gamma = gamma
        self.fit_model = None
        self.alignment = np.arange(self.num_motifs+1)
    
    def __repr__(self):
        return 'MotifMixtureModel:\n num_motifs:%s,\n gamma:%s,\n theta:%s'% (self.num_motifs+1, self.gamma, self.theta)
    
    def sample(self, x_len=288):
        """
        Generates a signal no shorter than x_len using gamma and theta.
        """
        num_windows = math.ceil(x_len/self.w)
        signal = np.array([])
        truth = []
        for i in range(num_windows):
            # draw m
            m = np.random.choice(self.theta, p = self.gamma)
            m_i = self.theta.index(m)
            for i in range(self.w):
                truth.append(m_i)
            sample = m.sample()
            signal = np.append(signal, sample)
            
        return signal, truth
    
    def create_dataset(self, n_data=100):
        """
        Draws a set of samples from the model. Could be used for patient level modelling, 
        here it's just a wrapper.
        Note: assuming always want default x_len. 
        """
        x, y = [], []
        for i in range(n_data):
            signal, truth = self.sample()
            x.append(signal)
            y.append(truth)
        return x, y
    
    def fit(self, data):
        """
        Fit the motif mixture model to data. Assumes all parameters except for gamma
        and theta are correct (could use more flexible fitting procedures in future).
        """
        x = []
        for day in data:
            x += [day[i:i+self.w] for i in range(0, len(day), self.w)]
        x = np.array(x)
        init_precisions = [self.std**-1 for i in range(self.num_motifs+1)] #+1 for backward compatibility
        motif_mix = mixture.GaussianMixture(n_components= self.num_motifs + 1, 
                                            covariance_type='spherical')
        motif_mix.fit(x)
        # is there some way to recover ordering?
        self.gamma = motif_mix.weights_
        mu = motif_mix.means_
        std = list(map(math.sqrt, motif_mix.covariances_))
        motifs = []
        for i in range(self.num_motifs + 1):
            motifs.append(Motif(w=self.w, mu=mu[i], std = std[i]))
        self.theta = motifs
        self.fit_model = motif_mix
        return self
    
    def predict(self, data):
        """
        Assigns cluster labels to data. Accounts for any reordering that's been done, but
        hacky.
        """
        x = []
        for day in data:
            x += [day[i:i+self.w] for i in range(0, len(day), self.w)]
        x = np.array(x)
        pred_ind = self.fit_model.predict(x)
        try:
            return [self.alignment[pred_ind[i]] for i in range(len(pred_ind))]
        except:
            print(self.alignment, pred_ind)
            raise
    def reorder(self, alignment):
        """
        Reorders the motif mixture model parameters to match alignment.
        Kind of hacky.
        """
        # can only do one reorder per model, otherwise errors with prediction I think
        # can rework if becomes a problem
        assert(np.all(np.equal(sorted(self.alignment),self.alignment)))
        self.theta = list(np.array(self.theta)[alignment])
        self.gamma = list(np.array(self.gamma)[alignment])
        self.alignment = alignment

class FixedLengthContextWindow(MotifMixtureModel):
    """
    The generative process for the fixed length context window model. 
    We add a context mixing parameter, and use a context specific gamma when
    sampling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.gamma
        # For right now, we'll use the weak dirichlet prior method from mtm
        # to define contexts. Might not be ideal, but easy...
        gamma_prior_proper = kwargs.get('gamma_prior', np.ones(self.num_motifs))
        bkg_prior = self.bkg_rat*np.sum(gamma_prior_proper)/(1-self.bkg_rat)
        self.gamma_prior = np.append(gamma_prior_proper, bkg_prior)
        
        # define contexts
        self.context_num = kwargs.get('context_num', 2)
        self.context_len = kwargs.get('context_len', 4)
        # TODO: think I might be lexically overloading alpha here
        self.alpha = kwargs.get('alpha', np.ones(self.context_num)/self.context_num)
        self.gamma = []
        for i in range(self.context_num):
            self.gamma.append(np.random.dirichlet(self.gamma_prior))
        
    def sample(self, x_len=288, verbose=False):
        """
        Generates a signal no shorter than x_len using gamma and theta
        """
        # TODO: May be too much of an overshot
        num_context_windows = math.ceil(x_len/(self.w*self.context_len))
        if verbose:
            print('number of context windows:', num_context_windows)
        signal = np.array([])
        truth = []
        context = []
        for i in range(num_context_windows):
            # draw c
            c_i = np.random.choice(np.arange(len(self.gamma)), p = self.alpha)
            c = self.gamma[c_i]
            for j in range(self.context_len):
                # draw m
                m = np.random.choice(self.theta, p = c)
                m_i = self.theta.index(m)
                for i in range(self.w):
                    truth.append(m_i)
                    context.append(c_i)
                sample = m.sample()
                signal = np.append(signal, sample)
        return signal, truth, context
    
    def create_dataset(self, n_data=100):
        """
        Draws a set of samples from the model. In later iterations
        will be used for patient level modelling, here it's just a wrapper.
        Note: assuming always want default x_len. 
        """
        x, y = [], []
        for i in tqdm(range(n_data)):
            signal, truth, context = self.sample()
            x.append(signal)
            y.append((truth, context))
        return x, y
    
def one_hot(point, rng):
    """
    Encodes value point in range rng to one-hot vector
    """
    vec = np.zeros(rng)
    try:
        vec[point] = 1
    except:
        print(point)
        raise
    return vec

def make_protocontext_vectors(data, num_motifs, mpc):
    proto_contexts = []
    for day in data:
        for i in range(int(len(day)/mpc)): # num windows
            cont = np.zeros(num_motifs)
            for j in range(mpc):
                cont += one_hot(day[i*mpc+j], num_motifs)
            proto_contexts.append(cont)
    return proto_contexts

def cluster_protocontext(proto_contexts, cont):
    km = sklearn.cluster.KMeans(n_clusters = cont)
    return km.fit_predict(proto_contexts)

def contextualize_data(contexts, mpc, data):
    # turn context into same shape as day
    data_context = []
    for i in range(len(data)):
        day_context = []
        for j in range(int(len(data[0])/mpc)):
            for k in range(mpc):
                day_context.append(contexts[i*int(len(data[0])/mpc)+j]) # copy mpc times so one context per mpc
        day_data_context = []
        for z in range(len(data[i])):
            day_data_context.append((data[i][z], day_context[z]))
        data_context.append(day_data_context)
    return data_context

def dummy_contextualize(data, mpc, num_cont):
    data_context = []
    for i in range(len(data)):
        day_context = []
        for j in range(int(len(data[0])/mpc)):
            context = np.random.choice(np.arange(num_cont))
            for k in range(mpc):
                day_context.append((data[i][j*mpc + k], context))
        data_context.append(day_context)
    return data_context

def make_motifs_mix(mixture, sd=1):
    motifs = []
    for i in range(len(mixture.theta)):
        m_i = []
        for j in range(len(mixture.theta[i].mu)):
            m_i.append({'mu':mixture.theta[i].mu[j], 'sd':sd})
        motifs.append(m_i)
    return motifs

def make_dataset_from_mmm_uncondensed(data, num_motifs, n_data):
    """Makes dataset using motif mixture model"""
    data_comb = []
    for i in range(len(data)):
        data_comb.append(np.concatenate(data[i][:]))
    cmmm_learn = MotifMixtureModel(num_motifs=num_motifs-1)
    cmmm_learn.fit(data_comb[0:n_data])
    data_mix = []
    mot = make_motifs_mix(cmmm_learn)
    days = []
    for i in tqdm(range(len(data))):
        day = data[i]
        days.append(process_day_mix(day, mot))
    return days

# for evaluating sample runs
# evaluation functions:
def normpdf(x, mu, sigma):
    return scipy.stats.norm._logpdf((x - mu) / sigma) - np.log(sigma)

def make_motifs(tr, burn_in, sd=1):
    """
    Given trained CMMM in form of PyMC3 trace tr, 
    uses ML estimate to assign motif labels to all samples
    """
    motifs = []
    n_sample = tr['theta_mu'].shape[0]
    n_mot = tr['theta_mu'].shape[1]
    len_mot = tr['theta_mu'].shape[2]
    theta_mu_avg = np.mean(tr['theta_mu'][n_sample-burn_in::], 0)
    for i in range(n_mot):
        m_i = []
        for j in range(len_mot):
            m_i.append({'mu':theta_mu_avg[i][j], 'sd':sd})
        motifs.append(m_i)
    return motifs

def make_motifs_mix(mixture, sd=1):
    """
    Given trained MMM in form of MotifMixtureModel class,
    uses ML estimate to assign motif labels to all samples
    """
    motifs = []
    for i in range(len(mixture.theta)):
        m_i = []
        for j in range(len(mixture.theta[i].mu)):
            m_i.append({'mu':mixture.theta[i].mu[j], 'sd':sd})
        motifs.append(m_i)
    return motifs

def motif_likelihood(motifs, sample):
    """
    Finds likelihood of sample under each motif
    """
    bsf = -np.inf
    bsfi = -1
    for i in range(len(motifs)):
        #test motif
        prob = 0
        for j in range(len(sample)):
            try:
                prob += normpdf(sample[j], motifs[i][j]['mu'], motifs[i][j]['sd'])
            except:
                print(sample[j], motifs[i][j]['mu'], motifs[i][j]['sd'])
                raise
            #prob += motifs[i][j].logpdf(sample[j])
        if prob > bsf:
            bsf = prob
            bsfi = i
    return bsfi

def context_likelihood(tr, burn_in, cluster):
    """
    Finds most likely context for any given cluster of motifs
    """
    n_sample = tr['gamma'].shape[0]
    n_gamma = tr['gamma'].shape[1]
    num_mot = tr['gamma'].shape[2]
    gamma_avg = np.mean(tr['gamma'][n_sample-burn_in::], 0)
    
    bsf = 0
    bsfi = -1
    for i in range(n_gamma):
        #test context
        prob = 1
        for j in range(len(cluster)):
            prob *= gamma_avg[i][cluster[j]]
        if prob > bsf:
            bsf = prob
            bsfi = i
    return bsfi

def process_day(day, m_per_c, c_per_x, burn_in, tr):
    # make into motifs
    motifs = make_motifs(tr, burn_in=burn_in, sd=1)
    day_m = []
    for i in range(len(day)):
        day_m.append(motif_likelihood(motifs, day[i]))
    context_split = [day_m[i*m_per_c:(i+1)*m_per_c] for i in range(c_per_x)]
    contexts = []
    for i in range(len(context_split)):
        contexts.append(context_likelihood(tr, burn_in, context_split[i]))
    return day_m, contexts

def process_day_mix(day, motifs):
    day_m = []
    for i in range(len(day)):
        day_m.append(motif_likelihood(motifs, day[i]))
    return day_m 

def vectorize_dataset(dataset):
    """Given array of examples, applies dictvectorizer"""
    dv = feature_extraction.DictVectorizer(sparse=False)
    return dv.fit_transform(dataset)

def categorical_switch_points(arr):
    """
    Given a categorical array, finds points that switch from one category
    to another, returning a sparse representation of the array.
    """
    switch_points = []
    switch_start = 0
    local_cat = arr[0]
    for i in range(len(arr)):
        if arr[i] != local_cat:
            switch_points.append((switch_start, i-1, local_cat))
            switch_start = i
            local_cat = arr[i]
    switch_points.append((switch_start, i, local_cat))
    return switch_points

def plot_regions(signal, regions):
    """
    Given a signal and categorical regions, plots the signal with shading based
    on the regions. Assumes regions are represented by densley distributed ints.
    Won't look good if regions change too often, and does not assume ordering of
    regions (uses qualitative colormap).
    """
    hsv = plt.get_cmap('hsv', max(regions)+2)
    region_order = list(set(regions))
    #region_colors = [sns.xkcd_rgb["pale red"], None] # total hackjob
    alphas = [0.2 for i in range(max(regions)+1)]
    region_colors = [hsv(i) for i in range(max(regions)+1)]
    region_colors[-1] = None
    alphas[-1] = 0
    rsp = categorical_switch_points(regions)
    fig, ax = plt.subplots()
    ax.plot(signal)
    ylim = ax.get_ylim()
    for sp in rsp:
        x = np.arange(sp[0], sp[1]+1)
        y = signal[sp[0]:sp[1]+1]
        plt.fill_between(x, 
                         ylim[0], 
                         ylim[1], 
                         facecolor=region_colors[region_order.index(sp[2])], 
                         alpha=alphas[region_order.index(sp[2])])