from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import MaxWordsPerturbed

from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
    MinWordLength,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import SearchMethod
from textattack.transformations import WordSwapWordNet
from textattack.transformations import WordSwapEmbedding


from textattack.shared.attack import Attack

from textattack.attack_recipes import AttackRecipe
from textattack.shared.validators import transformation_consists_of_word_swaps


import numpy as np

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.utils import get_modifiable_indices
import corpus

np.random.seed(21)

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

class FrequencySearchSNLI(SearchMethod):


    def __init__(self,type):
        print("--FREQUNECY SEARCH " + type + "--")
        self.type = type
        
    def _perform_search(self, initial_result):
        
        print('Try loading SNLI training Corpus')
        freq_dict = corpus.load_csv('../snli_train_word_freq.csv')
        

        modifiable_indices = get_modifiable_indices(initial_result.attacked_text,self.pre_transformation_constraints )
        try_indices  = set()
        cur_result = initial_result
        
     
        
        if len(modifiable_indices) is 0:
            
            return cur_result
        
        while not cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            
            random_choice = np.random.choice(list(modifiable_indices))
            
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[random_choice]
            )

            transformed_text_candidates = Union(transformed_text_candidates[0],transformed_text_candidates[1])
           
            if len(transformed_text_candidates) is 0:
                
                return cur_result
            
            frequencies = get_word_frequencies(transformed_text_candidates,freq_dict,random_choice)
            lowest_freq = np.argmin(frequencies)
            try_indices.add(random_choice)
            
            results = self.get_goal_results([transformed_text_candidates[lowest_freq]])
            
            if self.type == 'R':
                cur_result = results[0][0]
            elif self.type == 'P':
                
                cur_score = results[0][0].score
                if cur_score > initial_result.score:
                    cur_result = results[0][0]
                
                
                '''
                else:
                    frequencies.remove(lowest_freq)
                    transformed_text_candidates.remove(lowest_freq)
                    while len(frequencies) > 0:
                        lowest_freq = np.argmin(frequencies)
                        print("FREQUENCIES")
                        print(frequencies)
                        print("LOWEST")
                        print(lowest_freq)
                        results = self.get_goal_results([transformed_text_candidates[lowest_freq]])
                        cur_score = results[0][0].score
                        print("CUR SCORE")
                        print(cur_score)
                        if cur_score > initial_result.score:
                            cur_result = results[0][0]
                            print("SUCCESS")
                            break
                '''
            else:
                raise ValueError('Type must be either P or R for frequency search')

            modifiable_indices = get_modifiable_indices(cur_result.attacked_text,self.pre_transformation_constraints)
            modifiable_indices = modifiable_indices - try_indices

            if len(modifiable_indices) is 0:
                return cur_result

        return cur_result
        
        def check_transformation_compatibility(self, transformation):
            """The genetic algorithm is specifically designed for word
            substitutions."""
        return transformation_consists_of_word_swaps(transformation)


class FrequencyR(AttackRecipe):


    @staticmethod
    def build(model):

        transformations = [WordSwapEmbedding(max_candidates=8),WordSwapWordNet()]
        
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off

        constraints = [RepeatModification()]
        #constraints = [StopwordModification()]
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
        search_method = FrequencySearchSNLI('R')

        return Attack(goal_function, constraints, transformations, search_method)


class FrequencyP(AttackRecipe):


    @staticmethod
    def build(model):

        transformations = [WordSwapEmbedding(max_candidates=8),WordSwapWordNet()]
        
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off

        constraints = [RepeatModification()]
        #constraints = [StopwordModification()]
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
        search_method = FrequencySearchSNLI('P')

        return Attack(goal_function, constraints, transformations, search_method)