from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints import PreTransformationConstraint

from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)

from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import SearchMethod
from textattack.transformations import WordSwapWordNet
from textattack.transformations import WordSwapEmbedding


from textattack.shared.attack import Attack

from textattack.attack_recipes import AttackRecipe
from textattack.shared.validators import transformation_consists_of_word_swaps




from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.utils import get_modifiable_indices

import numpy as np
import math
import corpus

np.random.seed(21)


def phi_value(freq):
    return math.log(1+freq)
    
def Union(lst1, lst2): 
    final_list = list(set((set(lst1) | set(lst2)))) 
    return final_list 
    
def get_word_frequencies(attacked_texts,freq_dict,modified_index):
    frequencies = []
    
    for attack_text in attacked_texts:
        word = attack_text.words[modified_index]
        if word in freq_dict:
            frequencies.append(int(freq_dict[word]))
        else:
            frequencies.append(0)
    
    return frequencies
    
def get_word_phi(word,freq_dict):
    
    freq = 0
    if word in freq_dict:
        freq = int(freq_dict[word])
    
    return phi_value(freq)

class FrequencyConstraint(PreTransformationConstraint):
    """A constraint that prevents modifications of words with less frequency than the threshold delta.

    :param freq_dict: Frequency dictionary produced by the train data.
    :param delta: Frequency threshold
    """

    def __init__(self, freq_dict, delta):

        if delta < 0:
            raise ValueError("Frequency threshold (delta) needs to be positive")
        self.delta = delta
        self.freq_dict = freq_dict

    def _get_modifiable_indices(self, current_text):
        idxs = []
        for i, word in enumerate(current_text.words):
            freq = 0
            if word in self.freq_dict:
              freq = int(self.freq_dict[word])
            if phi_value(freq) < self.delta:
              idxs.append(i)

        return set(idxs)



class DetectionSearch(SearchMethod):


    def __init__(self,freq_dict,type,gamma=None):
        print("--Detection SEARCH " + type + "--")
        self.type = type
        self.freq_dict = freq_dict
        self.gamma = gamma
        
    def _perform_search(self, initial_result):
        
        
        #print(initial_result.attacked_text)
        modifiable_indices = get_modifiable_indices(initial_result.attacked_text,self.pre_transformation_constraints )
        #print(modifiable_indices)
        cur_text = initial_result.attacked_text
        cur_result = initial_result
        
        for index in modifiable_indices:
                
            word_freq  = get_word_phi(initial_result.attacked_text.words[index],self.freq_dict)
                
            #print("WORD INDEX")
            #print(index)
            
            transformed_text_candidates = self.get_transformations(
                cur_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index]
            )
            
            #print("CANDIDATES")
            #print(transformed_text_candidates)

            transformed_text_candidates = Union(transformed_text_candidates[0],transformed_text_candidates[1])
                
            #print("Union CANDIDATES")
            #print(transformed_text_candidates)
           
            if len(transformed_text_candidates) is 0:
                    
                #print("NO TRANSFORMATION FOUND")
                continue
            
            frequencies = [get_word_phi(candidate.words[index],self.freq_dict) for candidate in transformed_text_candidates]
            
            #print("Frequncies ")
            #print(frequencies)
            max_freq = np.argmax(frequencies)
            #print("Max Frequency")
            #print(max_freq)
                
            if max_freq < word_freq:
                
                #print("Frequency is not better")
                continue
            
            else:
                
                cur_text = transformed_text_candidates[max_freq]
                
                
        results = self.get_goal_results([cur_text])
            
        cur_result = results[0][0]
                
        if self.type == 'continuous':
                
            cur_score = results[0][0].score
            if cur_score - initial_result.score>self.gamma:
                # Make sure that status is changed
                cur_result.goal_status = GoalFunctionResultStatus.SUCCEEDED
            else:
                cur_result.goal_status = GoalFunctionResultStatus.SEARCHING
                
        else:
            if not self.type == 'discrete':
                raise ValueError('Type must be either discrete or continuous')


        return cur_result
        
        def check_transformation_compatibility(self, transformation):
            """The genetic algorithm is specifically designed for word
            substitutions."""
        return transformation_consists_of_word_swaps(transformation)


class FWGS(AttackRecipe):


    @staticmethod
    def build(model,freq_dict,delta,type,gamma,candidates):

        transformations = [WordSwapEmbedding(max_candidates=candidates),WordSwapWordNet()]
        
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off

        constraints = [RepeatModification()]
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        constraints.append(FrequencyConstraint(freq_dict,delta))

        #constraints.append(PartOfSpeech())
        #
        # Universal Sentence Encoder with a minimum angular similarity of ÃŽÂµ = 0.5.
        #

        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        
        search_method = DetectionSearch(freq_dict,type,gamma)

        return Attack(goal_function, constraints, transformations, search_method)
        
        


class VocabularySearch(SearchMethod):


    def __init__(self,freq_dict):
        print("-- Vocabulary SEARCH --")
        self.freq_dict = freq_dict

    def _perform_search(self, initial_result):
        
        
        #print(initial_result.attacked_text)
        modifiable_indices = get_modifiable_indices(initial_result.attacked_text,self.pre_transformation_constraints )
        #print(modifiable_indices)
        cur_text = initial_result.attacked_text
        cur_result = initial_result
        
        for index in modifiable_indices:
            
            if cur_text.words[index] in self.freq_dict:
                continue
                
            #print("WORD INDEX")
            #print(index)
            
            transformed_text_candidates = self.get_transformations(
                cur_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index]
            )
            
            #print("CANDIDATES")
            #print(transformed_text_candidates)

            transformed_text_candidates = Union(transformed_text_candidates[0],transformed_text_candidates[1])
                
            #print("Union CANDIDATES")
            #print(transformed_text_candidates)
           
            if len(transformed_text_candidates) is 0:
                    
                #print("NO TRANSFORMATION FOUND")
                continue
            
            
            
            random_choice = np.random.choice(transformed_text_candidates)

                
            if random_choice.words[index] not in self.freq_dict:
                continue
                
            else:
                
                cur_text = random_choice
                
                
        results = self.get_goal_results([cur_text])
        cur_result = results[0][0]
         
        return cur_result
        
        def check_transformation_compatibility(self, transformation):
            """The genetic algorithm is specifically designed for word
            substitutions."""
        return transformation_consists_of_word_swaps(transformation)




class NWS(AttackRecipe):


    @staticmethod
    def build(model,freq_dict,delta=None,type=None,gamma=None,candidates=8):

        transformations = [WordSwapEmbedding(max_candidates=candidates),WordSwapWordNet()]
        
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off

        constraints = [RepeatModification()]
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)

        #constraints.append(PartOfSpeech())
        #
        # Universal Sentence Encoder with a minimum angular similarity of ÃŽÂµ = 0.5.
        #

        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        
        search_method = VocabularySearch(freq_dict)

        return Attack(goal_function, constraints, transformations, search_method)