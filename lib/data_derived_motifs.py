"""
Functions for finding and evaluating plain and contextual data driven motifs, specifically for CGM data

 Many functions assume data structured as follows
 patients = {patient_1:{'glucose':[session_1, session_2, ..., session_n],
                        other info},
             patient_2:{'glucose':[session_1, session_2, ..., session_n],
                        other info}, 
             ...}
 Where patient_x is a collection of glucose and related clinical data. Each glucose session contains several days
 of CGM data. CGM data consists of glucose readings in mg/dL with values between 40-400 measured at 5 minute intervals.
 np.nan indicates missing value
 
 Of course, while we assume glucose data, this could be used for any form of sequential data. Though it is possible certain threshold values would need to be adjusted.
"""
from lib import glucose_processing as GP
import numpy as np
import sklearn
from tqdm import tqdm
from collections import Counter

hypo_thresh = 1
hyper_thresh = 2


def mini_interpolate(old_arr, max_size):
    '''
    Inefficient method to interpolate missing chunks of data smaller than max_size in arr 
    '''
    arr = np.copy(old_arr)
    bool_arr = np.isnan(arr)
    for i in range(len(bool_arr)-2):
        # look for start False True
        if not bool_arr[i] and bool_arr[i+1]:
            # check to see if can interpolate
            for j in range(1, min(max_size+2, len(bool_arr)-i)): # check for off by one
                if not bool_arr[i+j]:
                    # interpolate this chunk
                    arr[i:i+j+1] = GP.interpolate(arr[i:i+j+1])
                    break
    return arr


#Discretization
def motif_candidates(data, motif_length, stride_length=1):
    '''
    Seperates unequal data chunks into possibly overlapping motif candidates
    Inputs
    data : list of pieces of contiguous data to be turned into candidates
    motif_length : length of motif candidates
    stride_length : amount of overlap between candidates (1 is max, motif_length is none)
    Outputs
    candidate_list : list of motif candidates
    chunk_divs : index in list where chunk changes
    '''
    candidate_list = []
    chunk_div = []
    for chunk in data:
        chunk_div.append(len(candidate_list))
        assert(len(chunk) >= motif_length)
        for i in range(0, len(chunk)-motif_length, stride_length):
            candidate_list.append(chunk[i:i+motif_length])
    return candidate_list, chunk_div


def divisions(dist, vals, nbins):
    '''
    Finds equally sized bins
    Inputs
    dist : distribution of unique values, currently assumes equally spaced
    vals : value for each index of dist, made more sense in integers
    nbins : number of bins to divide into
    Outputs
    div : list of bin edges (beginning and end implicit)
    '''
    goal_for_bin = np.sum(dist)/float(nbins)
    print(goal_for_bin)
    div = []
    tot = 0
    for i in range(len(dist)):
        tot += dist[i]
        if tot >= goal_for_bin:
            div.append(vals[i])
            tot = 0
    print(len(div), nbins)
    return div


# underrepresenting beginning and end of chunks right now due to sliding window
def balancing_SAX(dataset, alph_size, train_div = None):
    '''
    Modification of SAX to ensure balance of characters across dataset.
    Assumes word length = 1
    Inputs
    dataset : list of sequences to be discretized
    alph_size : total number of distinct characters to be used
    Outputs
    sax_dataset : list of discretized sequences in same order as dataset
    '''
    if train_div is None:
        # need to fit balance
        vals = Counter({}) 
        for x in dataset:
            vals.update(x)
        val_set = sorted(vals.keys())
        # this method seems to imply constant distance between values, not true in general
        dist = np.zeros(len(val_set))
        for i in range(len(val_set)):
            dist[i] = vals[val_set[i]]
        div = divisions(dist, val_set, alph_size)
    else:
        div = train_div
    sax_dataset = [0 for i in range(len(dataset))]
    for i in range(len(dataset)):
        # make sure NaN's preserved
        nan_mask = np.isnan(dataset[i])
        digits = np.digitize(dataset[i], div)
        digits_final = []
        for j in range(len(digits)):
            if nan_mask[j]:
                digits_final.append(np.nan)
            else:
                digits_final.append(digits[j])
        sax_dataset[i] = tuple(digits_final)
    return sax_dataset, div


def remove_runs(dataset, chunk_divs):
    '''
    Consolidate runs for more interesting motifs
    Note removed curr_run tracker to keep all motifs same length
    as a result this function is needlessly complicated
    Inputs
    dataset : list of discretized motif candidates
    chunk_divs : index of where noncontiguous motifs lie
    Outputs
    consolidated_dataset : dataset, but with consecutive identical runs removed
    '''
    consolidated_dataset = []
    curr_run = 0
    prev = float('nan')
    for i in tqdm(range(len(dataset))):
        if i in chunk_divs: # check i, not i+/-1
            # switched to new chunk
            if curr_run > 0:
                consolidated_dataset.append(prev+(curr_run,))
                curr_run = 0
        m = dataset[i]
        if len(Counter(m)) == 1:
            if m == prev:
                # continuing run
                curr_run += 1
            else: # this shouldn't actually be a concern as we're using sliding window
                # new run
                if curr_run > 0:
                    consolidated_dataset.append(prev+(curr_run,))
                curr_run = 1
        else:
            if curr_run > 0:
                consolidated_dataset.append(prev+(curr_run,))
                curr_run = 0
            consolidated_dataset.append(m)
        prev = m
    return consolidated_dataset


def get_chunks(a):
    '''
    Split a into all contiguous non-nan chunks
    '''
    return [a[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))]


def sess_to_cand(patients, motif_length, stride_length):
    '''
    Break up contiguous data into motif candidates
    Inputs
    patients : collection of patient data, contains glucose sessions
    motif_length : length of motif candidates
    stride_length : space between motif candidates
    Outputs
    cands : motif candidates
    chunkdiv : index keeping track of where candidates came from
    '''
    chunks = []
    for pat in patients:
        for sess in patients[pat]['glucose']:
            sess_chunk = get_chunks(sess)
            for c in sess_chunk:
                if len(c) >= motif_length: # could contain motif
                    chunks.append(c)
    cands, chunkdiv = motif_candidates(chunks, motif_length, stride_length)
    return cands, chunkdiv


def cand_preprocessing(candidates, center, scale, log=True):
    '''
    Apply preprocessing to candidates. Can't scale unless you center
    '''
    if not center:
        return candidates
    elif not scale:
        centered_candidates = []
        if log:
            for i in tqdm(range(len(candidates))):
                centered_candidates.append(candidates[i]-np.mean(candidates[i]))
        else:
            for i in range(len(candidates)):
                centered_candidates.append(candidates[i]-np.mean(candidates[i]))
        return centered_candidates
    else:
        scaled_candidates = []
        if log:
            for i in tqdm(range(len(candidates))):
                if min(candidates[i]) == max(candidates[i]):
                    scaled_candidates.append(candidates[i]-np.mean(candidates[i]))
                else:
                    scaled_candidates.append(
                        (candidates[i]-np.mean(candidates[i]))/np.std(candidates[i]))
        else:
            for i in range(len(candidates)):
                if min(candidates[i]) == max(candidates[i]):
                    scaled_candidates.append(candidates[i]-np.mean(candidates[i]))
                else:
                    scaled_candidates.append(
                        (candidates[i]-np.mean(candidates[i]))/np.std(candidates[i]))
        return scaled_candidates

    
def protomotifs(dataset, motif_min_count):
    '''
    Turn discretized dataset into protomotifs
    '''
    motif_tups = Counter(dataset).most_common() # equivalent to total projection
    motif_protos = []
    motif_protos_dict = {}
    for i in range(len(motif_tups)):
        if motif_tups[i][1] >= motif_min_count:
            motif_protos.append(motif_tups[i][0])
            motif_protos_dict[motif_tups[i][0]] = motif_tups[i][1]
    motif_proto_set = set(motif_protos)
    return motif_protos, motif_proto_set


def dummy_context(day, i, size):
    '''
    Returns random categorical context
    '''
    return np.random.choice(np.arange(size))


def time_context(day, i, size):
    '''
    Returns context based on time of occurence
    '''
    bins = [((j/size)+(1/size))*288 for j in range(size-1)]
    return np.digitize([i], bins)[0]


def abs_amplitude_context(day, i, size):
    '''
    Returns context based on the amplitude of previous section
    '''
    # could do balance scaling, or expert thresh for interest
    bins = [40+360*((j+1)/size) for j in range(size-1)]
    return np.digitize([np.mean(day[i:i+8])], bins)[0]


def rel_amplitude_context(day, i, size):
    '''
    Returns context based on relative amplitude of motif compared to day 
    '''
    percent = [100*((j+1)/size) for j in range(size-1)]
    bins = np.percentile(day, percent)
    return np.digitize([np.mean(day[i:i+8])], bins)[0]


def thresh_amplitude_context(day, i, size):
    '''
    Returns context in form of whether motif is occuring in hypo/hyper/euglycemic state
    Note size irrelevant
    '''
    bins = [70, 180]
    return np.digitize([np.mean(day[i:i+8])], bins)[0]


def smooth(x, window_radius):
    '''
    Simple mean smoothing to help with noise for trend
    '''
    x_smooth = []
    for i in range(len(x)):
        if i < window_radius:
            window = x[0:i+window_radius+1]
        else:
            window = x[i-window_radius:i+window_radius+1]
        x_smooth.append(np.mean(window))
    return x_smooth


def abs_trend_context(smooth_day, i, bins):
    '''
    Returns context based on window diff compared to whole diff
    Note: depends on globally defined diff_strip, probably bad form.
    Really just poorly structured, needs fixing
    '''
    return np.digitize([np.mean(np.diff(smooth_day[i-8:i]))], bins)[0]


def rel_trend_context(day, i, context_size):
    '''
    Returns context based on window diff compared with rest of day
    '''
    percent = [100*((j+1)/context_size) for j in range(context_size-1)]
    bins = np.percentile(np.diff(day), percent)
    return np.digitize([np.mean(np.diff(day[i-8:i]))], bins)[0]


def precomputed_context(precompute, i, context_size):
    '''
    Dummy function to allow precomputed contexts to fit into function, may make all
    contexts precomputed...
    '''
    return precompute[i]


def hmm_context(day, hmm):
    """
    Given a trained hmm (created in my case using pomegranate), uses predictions
    as context
    """
    return hmm.predict(np.nan_to_num(day).reshape(288, 1))


def day_to_motif_vec(day, 
                     motif_length, 
                     stride_length, 
                     div, 
                     motif_proto_set, 
                     n_clusters, 
                     cluster, 
                     context, 
                     center, 
                     scale):
    '''
    From day get motifs and motif locations
    '''
    day_motif_info = []
    cands, junk = motif_candidates([day], motif_length, stride_length)
    cands = cand_preprocessing(cands, center, scale, log=False)
    disc_cands, junk = balancing_SAX(cands, -1, div) # -1 junk
    for i in range(len(disc_cands)):
        if disc_cands[i] in motif_proto_set:
            day_motif_info.append((i, disc_cands[i]))
    context_func, context_size, precompute_context = context
    if precompute_context is not None:
        day = precompute_context(day) # Hacky
    motif_vec = np.zeros(n_clusters * context_size)
    cont = []
    promotif = []
    for i in range(len(day_motif_info)):
        cont.append(context_func(day, day_motif_info[i][0], context_size))
        promotif.append(day_motif_info[i][1])
    if len(promotif) > 0:
        motif = cluster.predict(promotif)
    else:
        motif = []
    for i in range(len(motif)):
        motif_vec[motif[i] + cont[i] * (motif[i]+1)] += 1 # check
    return motif_vec


def test_motif_rep(X, y_hypo, y_hyper, labels, params, budget, num_splits):
    '''
    Given motif representation look at performance across prediction
    '''
    X = np.array(X)
    y_hypo = np.array(y_hypo)
    y_hyper = np.array(y_hyper)
    labels = np.array(labels)
    logreg = sklearn.linear_model.LogisticRegression()
    # bunch of train/test splits on label
    np.random.seed(10)
    skf = sklearn.model_selection.GroupShuffleSplit(n_splits=num_splits, 
                                                    test_size=0.33,
                                                    random_state=11)
    hypo = []
    hypo_c = []
    hyper = []
    hyper_c = []
    for train, test in tqdm(skf.split(X, y_hypo, groups=labels)):
        X_train = X[train]
        X_test = X[test]
        y_hypo_train = y_hypo[train]
        y_hypo_test = y_hypo[test]
        y_hyper_train = y_hyper[train]
        y_hyper_test = y_hyper[test]
        label_train = labels[train]
        # possibly learning curve training subsample
        # cross validation for hyperparameter selection
        lkf = sklearn.model_selection.GroupKFold(n_splits=5)
        
        rcvs_hyper = sklearn.model_selection.RandomizedSearchCV(
            estimator=logreg, 
            param_distributions=params,
            n_iter=budget, 
            scoring='roc_auc', 
            random_state=12, 
            cv=lkf.split(X_train, 
                         y_hyper_train, 
                         groups=label_train))
        
        rcvs_hypo = sklearn.model_selection.RandomizedSearchCV(
            estimator=logreg, 
            param_distributions=params, 
            n_iter=budget, 
            scoring='roc_auc', 
            random_state=12, 
            cv=lkf.split(X_train, 
                         y_hypo_train, 
                         groups=label_train))
        rcvs_hypo.fit(X_train, y_hypo_train, groups=label_train)
        hypo.append(sklearn.metrics.roc_auc_score(y_hypo_test, 
                                  rcvs_hypo.predict_proba(X_test)[:, 1]))
        hypo_c.append(rcvs_hypo.best_params_['C'])
        rcvs_hyper.fit(X_train, y_hyper_train, groups=label_train)
        hyper.append(sklearn.metrics.roc_auc_score(y_hyper_test, 
                                  rcvs_hyper.predict_proba(X_test)[:, 1]))
        hyper_c.append(rcvs_hyper.best_params_['C'])
    return {'hypo':hypo, 'hyper':hyper}


def get_days_and_events(patients, 
                        max_interp_length, 
                        max_missing_data_x, 
                        max_missing_data_y):
    '''
    Interpolate sessions and break into days
    '''
    X = []
    y_days = []
    y_hypo = []
    y_hyper = []
    y_event = []
    labels = []
    for pat in patients:
        for sess in patients[pat]['glucose']:
            # small gap interpolation, may not even need
            interp_sess = mini_interpolate(sess, max_interp_length)
            days = GP.days(interp_sess, True)
            for i in range(len(days)-1):
                # Note: Currently using ALL data, not appropriate
                if np.count_nonzero(~np.isnan(days[i])) < max_missing_data_x:
                    continue
                if np.count_nonzero(~np.isnan(days[i+1])) < max_missing_data_y:
                    continue
                if GP.hypo_event_num(days[i+1][0:12]) > 0:
                    #probably just a continuing hypo event
                    if GP.hypo_event_num(days[i][276::]) > 0:
                        pass # doesn't matter for motifs, could for contextual
                X.append(days[i])
                labels.append(pat)
                y_hypo.append(GP.hypo_event_num(days[i+1]) >= hypo_thresh)
                y_hyper.append(GP.hyper_event_num(days[i+1]) >= hyper_thresh)
                y_event.append(y_hypo[-1] or y_hyper[-1])
                y_days.append(days[i+1])
    return X, labels, y_hypo, y_hyper, y_event