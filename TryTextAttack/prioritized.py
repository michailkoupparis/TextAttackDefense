import torch
from torch.nn.functional import softmax
import numpy as np

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

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.shared.utils import get_modifiable_indices

from textattack.shared.attack import Attack

from textattack.attack_recipes import AttackRecipe

from textattack.shared.validators import transformation_consists_of_word_swaps


np.random.seed(21)

class PriotirizedSearch(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.

    Args:
        wir_method: method for ranking most important words
    """

    def __init__(self):
        print("Priotirized Search")


    def _perform_search(self, initial_result):
        
        modifiable_indices = get_modifiable_indices(initial_result.attacked_text,self.pre_transformation_constraints )
        
        #not_tried_modifiable_indices = modifiable_indices
        
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
        
            if len(transformed_text_candidates) is 0:
                
                return cur_result
            
            results, search_over = self.get_goal_results(transformed_text_candidates)
            #print([r.attacked_text for r in results])
            #print([r.score for r in results])
            results = sorted(results, key=lambda x: -x.score)
            #print([r.score for r in results])
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
            else:
                return cur_result
             
            modifiable_indices = get_modifiable_indices(cur_result.attacked_text,self.pre_transformation_constraints)
            if len(modifiable_indices) is 0:
                return cur_result
    
        return cur_result
        
        def check_transformation_compatibility(self, transformation):
            """The genetic algorithm is specifically designed for word
            substitutions."""
        return transformation_consists_of_word_swaps(transformation)
        

class Priotirized(AttackRecipe):


    @staticmethod
    def build(model):

        transformations = WordSwapWordNet()
        
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
        #

        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = PriotirizedSearch()

        return Attack(goal_function, constraints, transformations, search_method)

