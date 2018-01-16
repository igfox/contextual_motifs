import numpy as np

#Relevant constants
INTERVAL=5
MINUTES=60
DAY_LENGTH=288
MAX_LENGTH=2544
HYPO=70
HYPER=180
HOUR=12

def hypo_event_num(arr):
    '''
    Returns number of hypoglycemic events in an array
    '''
    hypo = 0
    ht = 0
    max_ht = 0
    for i in arr:
        if i <HYPO:
            ht += 1
        else:
            ht = 0
        if ht == 4:
            hypo += 1
        if ht > max_ht:
            max_ht = ht
    return hypo

def hyper_event_num(arr):
    '''
    Returns number of hyperglycemic events in an array
    '''
    hyper = 0
    ht = 0
    max_ht = 0
    for i in arr:
        if i > HYPER:
            ht += 1
        else:
            ht = 0
        if ht == 4:
            hyper += 1
        if ht > max_ht:
            max_ht = ht
    return hyper

def get_start_index_hypo(arr):
    '''
    Returns first instance of hypoglycemic event
    '''
    for i in range(len(arr)-3):
        if hypo_event_num(arr[i:i+4]) > 0:
            return i
    return -1

def get_start_index_hyper(arr):
    '''
    Returns first instance of hyperglycemic event
    '''
    for i in range(len(arr)-3):
        if hyper_event_num(arr[i:i+4]) > 0:
            return i
    return -1

# strip nan padding
def strip_nans(patient):
    '''
    Remove the NAN alignment used for sessions
    '''
    pat_strip = []
    for session in patient:
        bindex = -1
        eindex = -1
        begin=False
        end=False
        for i in range(len(session)):
            if not np.isnan(session[i]):
                if not begin:
                    bindex = i
                    begin = True
        for i in reversed(range(len(session))):
            if not np.isnan(session[i]):
                if not end:
                    eindex = i
                    end = True
        pat_strip.append(session[bindex:eindex+1])
    return pat_strip   

def days(sess, valid=False):
    '''
    Splits a session into days
    TODO: check to see if alignment problems are happening
    '''
    days = []
    day_length = 288
    for i in range(9):
        days.append(sess[day_length*i:day_length*(i+1)])
    days[8] = np.pad(days[8], (0,len(days[0])-len(days[8])), mode='constant', constant_values=np.NAN)
    v_days = []
    for d in days:
        if len(d[~np.isnan(d)]) > 0 and np.nanmin(d) != np.nanmax(d):
            v_days.append(d)
    if valid:
        return v_days
    return days

def strip_nans(session):
    '''
    Removes all NANS from a session
    '''
    return session[~np.isnan(session)]

def sess_trim(sess):
    '''
    remove leading and trailing nans
    '''
    try:
        beg = np.where(~np.isnan(sess)==True)[0][0]
    except IndexError as e:
        print(len(strip_nans(sess)))
        raise
    end = np.where(~np.isnan(sess)==True)[0][-1]
    tr_sess = sess[beg:end+1]
    return tr_sess

def nan_helper(y):
    '''
    Helper function for interpolate
    '''
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate(sess, trim = True):
    '''
    Performs linear interpolation for missing pieces of a session
    '''
    if np.count_nonzero(~np.isnan(sess)) == 0:
        return sess
    if trim:
        trm = sess_trim(sess).copy()
    else:
        trm = sess.copy()
    nans,x = nan_helper(trm)
    trm[nans]= np.interp(x(nans), x(~nans), trm[~nans], left=float('nan'), right=float('nan'))
    return trm

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
                    arr[i:i+j+1] = interpolate(arr[i:i+j+1])
                    break
    return arr
