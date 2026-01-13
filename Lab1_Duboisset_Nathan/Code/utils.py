
import numpy as np


def create_train_dataset():
    N_TRAIN = 100000
    MAX_TRAIN_CARD = 10

    ############## Task 1
    
    training_in = []
    training_out = []
    
    for sample_index in range(N_TRAIN):
        actual_cardinality = np.random.randint(1, MAX_TRAIN_CARD + 1)
        
        digits = np.random.randint(1, 11, size=actual_cardinality)
        
        number_of_zeros = MAX_TRAIN_CARD - actual_cardinality
        padding = np.zeros(number_of_zeros, dtype=int)
        
        final_sample = np.concatenate([padding, digits])
        
        sample_sum = np.sum(digits)
        
        training_in.append(final_sample)
        training_out.append(sample_sum)
    
    X_train = np.array(training_in)
    y_train = np.array(training_out)

    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    SAMPLES_PER_CARDINALITY = 10000
    MIN_CARDINALITY = 5
    MAX_CARDINALITY = 100
    CARDINALITY_STEP = 5
    
    X_test = []
    y_test = []
    
    
    for cardinality in range(MIN_CARDINALITY, MAX_CARDINALITY + 1, CARDINALITY_STEP):
        samples_for_this_cardinality = []
        targets_for_this_cardinality = []
        
        for sample_index in range(SAMPLES_PER_CARDINALITY):
            digits = np.random.randint(1, 11, size=cardinality)
            
            sample_sum = np.sum(digits)
            
            samples_for_this_cardinality.append(digits)
            targets_for_this_cardinality.append(sample_sum)
        
        X_test.append(np.array(samples_for_this_cardinality))
        y_test.append(np.array(targets_for_this_cardinality))
        

    return X_test, y_test
