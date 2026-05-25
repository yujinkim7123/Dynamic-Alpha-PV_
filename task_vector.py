import torch
import torch.nn as nn

from utils.utils import get_param_names_to_merge

class TaskVector:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, 
                 exclude_param_names_regex: list = None, task_vector_param_dict: dict = None):
        """
        Task vector. Initialize the task vector from a pretrained model and a finetuned model, or
        directly passing the task_vector_param_dict dictionary.
        :param pretrained_model: nn.Module, pretrained model
        :param finetuned_model: nn.Module, finetuned model
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param task_vector_param_dict: dict, task vector to initialize self.task_vector_param_dict
        """
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), 
                                                            exclude_param_names_regex=exclude_param_names_regex)
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]

    def __add__(self, other):
        """
        Add task vector.
        :param other: TaskVector to add
        :return: 새로운 TaskVector (두 TaskVector의 합)
        """
        assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: TaskVector to add
        :return: 새로운 TaskVector (두 TaskVector의 합)
        """
        return self.__add__(other)

    def __mul__(self, scalar):
        """
        TaskVector와 스칼라 간의 곱셈을 지원합니다.
        :param scalar: int 또는 float
        :return: 각 파라미터에 스칼라를 곱한 새로운 TaskVector
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError("TaskVector multiplication only supports int or float as scalar.")
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name, tensor in self.task_vector_param_dict.items():
                new_task_vector_param_dict[param_name] = tensor * scalar
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __rmul__(self, scalar):
        """
        스칼라 * TaskVector를 지원하기 위해 __rmul__를 정의합니다.
        """
        return self.__mul__(scalar)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        """
        combine the task vector with pretrained model
        :param pretrained_model: nn.Module, pretrained model
        :param scaling_coefficient: float, scaling coefficient to merge the task vector
        :return: 병합된 파라미터 딕셔너리
        """
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]

        return merged_params
