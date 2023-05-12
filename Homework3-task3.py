#!/usr/bin/env python
# coding: utf-8

# # Task 3 — Augment the MPHF with a “fingerprint array”

# ##### ref: https://pypi.org/project/bbhash/
# ##### ref https://github.com/dib-lab/pybbhash
# ##### ref: https://piazza.com/class/ld0gqwk4xme2vy/post/87
# ##### ref: https://pypi.org/project/bitarray/

# In[1]:


# fpr = [1/(2**7), 1/(2**8), 1/(2**10)]
# fpr


# In[2]:


# ## smaller experiments to get familiarized with the functions
# import bbhash
# import random
        

# # some collection of 64-bit (or smaller) hashes
# uint_hashes = [i for i in range(1, 100, 4)] #[10, 20, 50, 80]

# num_threads = 16 # hopefully self-explanatory :)
# gamma = 1.0     # internal gamma parameter for BBHash

# mph = bbhash.PyMPHF(uint_hashes, len(uint_hashes), num_threads, gamma)

# # for val in uint_hashes:
# #     print('{} now hashes to {}'.format(val, mph.lookup(val)))

# # can also use 'mph.save(filename)' and 'mph = bbhash.load_mphf(filename)'.

# cnt = 0 
# total = 0
# for i in range(1000, 2000, 4):
#     if mph.lookup(i) is not None:
#         print(mph.lookup(i), i in uint_hashes)
#         cnt += 1
#     total += 1
        
# print('\nFP count', cnt)
# print('FPR', cnt/total)


# In[3]:


from time import time
import random
import os
import numpy as np
import bbhash
import random

num_threads = 16
gamma = 1.0  

### since BBHashTable only accepts integers and we are allowed to use integers, I am taking 10,000 integers for our experiments instead. 
## ref: https://piazza.com/class/ld0gqwk4xme2vy/post/87

# with open('mit_wordlist_10000.txt', 'r') as f:
#     random_unique_words = f.readlines()
random_unique_words = [ random.randint(100, 2**32) for i in range(10_000) ] # [x.strip() for x in random_unique_words]
len(random_unique_words)


# In[4]:


# K_sizes_ = [1_000, 2_000, 3_000, 4_000, 5_000]

# results = -1 * np.ones([10, 5, 5]) # (num_runs, K_sizes, different_values_to_store) 

# size_of_K = K_sizes_[0]
# random.shuffle(random_unique_words)
# print('\n\n')
# print(''.join(["="] * 50))
# print("size_of_K :\t", size_of_K)

# K = random_unique_words[:size_of_K] # input keys
# Kprime_extra = random_unique_words[size_of_K:size_of_K*2] # samples that are in Kprime but not in K


# total_time = 0
# ## setting bbhash
# mph = bbhash.PyMPHF(K, len(K), num_threads, gamma)
# print(mph.lookup(0))

# ## sanity check, ensuring I have no false negatives
# fn_count = 0
# for w in K:
#     t0 = time()
#     if mph.lookup(w) is None:
#         fn_count += 1
#     total_time += (time() - t0)

# if fn_count > 0:
#     print("fn_count:", fn_count)
#     raise Exception


# In[5]:


# num_bits_from_hash = 3
# a = 0

# for i in range(num_bits_from_hash):
#     a |= (1 << i)

# b = 10
# a & b


# In[6]:


def get_bitmask(fingerprints_bits=3):
    from bitarray import bitarray
    a = bitarray('1' * fingerprints_bits)
    return int(a.to01(),2)

bitmask = get_bitmask(fingerprints_bits=7)
bitmask & hash(230)


# In[7]:


"""
For each input dataset, and for each target false positive rate, measure:
(a) what is the observed false positive rate (also, as a sanity check, ensure you have no false negatives) 
(b) what is the total time spent querying K’ and 
(c) what is the total size of the mph filter? 
"""

## dataset1

K_sizes_ = [1_000, 2_000, 3_000, 4_000, 5_000]

results = -1 * np.ones([10, 5, 5, 3]) # (num_runs, K_sizes, different_values_to_store) 


# or each set of keys, consider building your MPHF and fingerprint array with fingerprints consisting of 7, 8 and 10 bits

    
for run_num in range(10):

    print('\n\n\n\n')
    print(''.join(["="] * 50))
    for index_b, fingerprints_bits in enumerate([7, 8, 10]):

        print('\n\n')
        print(''.join(["="] * 50))
        bitmask = get_bitmask(fingerprints_bits)
    
        for index_k, size_of_K in enumerate(K_sizes_):
        
            random.shuffle(random_unique_words)
            print("\n\nsize_of_K :\t", size_of_K)

            K = random_unique_words[:size_of_K] # input keys
            Kprime_extra = random_unique_words[size_of_K:size_of_K*2] # samples that are in Kprime but not in K


            
            ## setting bbhash
            mph = bbhash.PyMPHF(K, len(K), num_threads, gamma)
            print(mph.lookup(0))
            
            ## auxiliary array
            aux_array = np.ones([len(K)]).astype(int) * -1
            

            ## sanity check, ensuring I have no false negatives
            ## and, filling the auxiliary array
            fn_count = 0
            for w in K:
                index_w = mph.lookup(w)
                if index_w is None:
                    fn_count += 1
                else:
                    aux_array[index_w] = (bitmask & hash(w))
            if fn_count > 0:
                print("fn_count:", fn_count)
                raise Exception
                
            fn_count = 0
            for w in K:
                index_w = mph.lookup(w)
                if aux_array[index_w] != (bitmask & hash(w)):
                    fn_count += 1
            if fn_count > 0:
                print("fn_count:", fn_count)
                raise Exception

            
            ## observed false positive rate
            total_time = 0
            fp_count = 0
            for w in Kprime_extra:
                t0 = time()
                index_w = mph.lookup(w)
                if index_w is not None:
                    if aux_array[index_w] == (bitmask & hash(w)):
                        fp_count += 1
                total_time += (time() - t0)

            print("fp_count:", fp_count)
            print("observed_fpr:", fp_count / (size_of_K))
            print("total_time", total_time)

            mph.save('mph.tmp')
            np.save('aux_array', aux_array)
            np.savez_compressed('aux_array', aux_array)
            print(f'{index_k}, size on disk:', os.path.getsize('mph.tmp')/1000 + os.path.getsize('aux_array.npz')/1000)

            results[run_num, 1, index_k, index_b] = fp_count / (size_of_K)
            results[run_num, 2, index_k, index_b] = total_time
            results[run_num, 3, index_k, index_b] = (os.path.getsize('mph.tmp')/1000 + os.path.getsize('aux_array.npz')/1000)
            results[run_num, 4, index_k, index_b] = (os.path.getsize('mph.tmp')/1000 + os.path.getsize('aux_array.npy')/1000)



# In[8]:


mean_results = results.mean(0)


# In[9]:


# results.std(0)
mean_results[1:].shape, mean_results[0]


# In[10]:


from matplotlib import pyplot as plt

## what is the observed false positive rate (also, as a sanity check, ensure you have no false negatives)

plt.plot(K_sizes_, mean_results[1, :, 0], label='b=7')
plt.plot(K_sizes_, mean_results[1, :, 1], label='b=8')
plt.plot(K_sizes_, mean_results[1, :, 2], label='b=10')

plt.xlabel('Size of input dataset')
plt.ylabel('Observed false positive rate')

plt.legend(loc='best', bbox_to_anchor=(.75, .5))

try:
    os.mkdir('results_task3')
except:
    print('results_task3 folder already exists.')
    
plt.savefig('results_task3/observed_false_positive_rate')


# In[11]:


# what is the total time spent querying each K’ 

plt.plot(K_sizes_, mean_results[2, :, 0], label='b=7')
plt.plot(K_sizes_, mean_results[2, :, 1], label='b=8')
plt.plot(K_sizes_, mean_results[2, :, 2], label='b=10')


plt.xlabel('Size of input dataset')
plt.ylabel('Total time to query K\' (in seconds)')

plt.legend()

plt.savefig('results_task3/time_to_query_Kprime')


# In[12]:


# what is the total size of the MPHF + compressed fingerprint array (compressed)

plt.plot(K_sizes_, mean_results[3, :, 0], label='b=7')
plt.plot(K_sizes_, mean_results[3, :, 1], label='b=8')
plt.plot(K_sizes_, mean_results[3, :, 2], label='b=10')

plt.xlabel('Size of input dataset')
plt.ylabel('Size of the MPHF + compressed aux. array \n(in Kilo Bytes)')

plt.legend()
plt.savefig('results_task3/size_of_MPHF+compressed_aux_array')


# In[13]:


# what is the total size of the MPHF + uncompressed fingerprint array

plt.plot(K_sizes_, mean_results[4, :, 0], label='b=7')
plt.plot(K_sizes_, mean_results[4, :, 1], label='b=8')
plt.plot(K_sizes_, mean_results[4, :, 2], label='b=10')

plt.xlabel('Size of input dataset')
plt.ylabel('Size of the MPHF + uncompressed aux. array \n(in Kilo Bytes)')

plt.legend()
plt.savefig('results_task3/size_of_MPHF+uncompressed_aux_array')


# In[14]:


## How do these compare to the bloom filter? 
## -> 


## How does the false positive rate compare to the expectations you had before you performed this experiment?
## -> 


# In[15]:


mean_results[3]


# In[ ]:





# In[ ]:





# In[ ]:




