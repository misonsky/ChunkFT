import re
import numpy as np
from transformers.utils import logging
from peft import TaskType
logger = logging.get_logger(__name__)
class ChunkCallBack(object):
    def __init__(self,freeze_layers,strategy,chunk_num,taskType,peft_type) -> None:
        self.strategy = strategy
        self.chunk_num = chunk_num
        self.freeze_layers = freeze_layers
        self.peft_type = peft_type
        self.taskType = taskType
        self.pattern_list= self.GetSpecialLayer()
        self.chunk_dict = {}
    @property
    def get_chunk_dict(self):
        return self.chunk_dict
    @property
    def emb_pattern(self):
        return list()
    @property
    def others_pattern(self):
        return list()
    @property
    def seq_cls_head(self):
        return list()
    @property
    def token_cls_head(self):
        return list()
    @property
    def qa_cls_head(self):
        return list()
    @property
    def causal_head(self):
        return list()
    @property
    def others_pattern(self):
        return list()
    def embed_strategy(self):
        return self.emb_pattern,"column"
    def head_strategy(self):
        if self.taskType == TaskType.SEQ_CLS:
            return self.seq_cls_head,"row"
        if self.taskType == TaskType.TOKEN_CLS:
            return self.token_cls_head,"row"
        if self.taskType == TaskType.QUESTION_ANS:
            return self.qa_cls_head,"row"
        if self.taskType == TaskType.CAUSAL_LM:
            return self.causal_head,"column"
        else:
            raise ValueError("......unsupported task type......")
    def SequenceClassificationSpecialLayer(self):
        special_layers = []
        special_layers.extend(self.emb_pattern)
        special_layers.extend(self.others_pattern)
        special_layers.extend(self.seq_cls_head)
        return special_layers
    def TokenClassificationSpecialLayer(self):
        special_layers = []
        special_layers.extend(self.emb_pattern)
        special_layers.extend(self.others_pattern)
        special_layers.extend(self.token_cls_head)
        return special_layers
    def QuestionAnsweringSpecialLayer(self):
        special_layers = []
        special_layers.extend(self.emb_pattern)
        special_layers.extend(self.others_pattern)
        special_layers.extend(self.qa_cls_head)
        return special_layers
    def CausalLMSpecialLayer(self):
        special_layers = []
        special_layers.extend(self.emb_pattern)
        special_layers.extend(self.others_pattern)
        special_layers.extend(self.causal_head)
        return special_layers
    def check_selection(self,elements,name_search):
        if len(name_search)<=0:
            return False
        elements = elements = [element if '\\' in element else re.escape(element) for element in elements]
        signal_value = [1 if len(re.compile(element).findall(name_search[0]))>0 else 0 for element in elements]
        if sum(signal_value)<=0:
            return False
        else:
            return True
    def split_block(self,hidden_size):
        base_size = hidden_size // self.chunk_num
        remainder = hidden_size % self.chunk_num
        chunk_sizes = [base_size + 1 if i < remainder else base_size for i in range(self.chunk_num)]
        positions = np.cumsum([0] + chunk_sizes[:-1]).tolist()
        assert len(positions) == self.chunk_num
        positions.append(hidden_size)
        ranges = [(positions[i], positions[i+1]) for i in range(self.chunk_num)]
        return ranges
    def get_split_dimension(self,p):
        tensor_shape = p.shape
        if len(tensor_shape)==1 or self.strategy == "column":
            p.strategy = 0
            if tensor_shape[0] not in self.chunk_dict:
                self.chunk_dict[tensor_shape[0]] = self.split_block(tensor_shape[0])
        elif p.strategy == "row":
            p.strategy = 1
            if tensor_shape[1] not in self.chunk_dict:
                self.chunk_dict[tensor_shape[1]] = self.split_block(tensor_shape[1])
    def set_parameter(self,model):
        assert self.strategy in ["row","column"]
        e_pattern,e_strategy = self.embed_strategy()
        h_pattern,h_strategy = self.head_strategy()
        for name,p in model.named_parameters():
            if self.check_selection(e_pattern,[name]):
                p.strategy = e_strategy
            elif self.check_selection(h_pattern,[name]):
                p.strategy = h_strategy
            else:
                p.strategy = self.strategy
            self.get_split_dimension(p)
    def check_task_type(self,taskType,model_name,TaskTInterface):
        logger.warning("For {} the HiTaskType should be {}".format(model_name," , ".join(TaskTInterface)))
        assert taskType in TaskTInterface
    def GetSpecialLayer(self):
        if self.taskType == TaskType.SEQ_CLS:
            return self.SequenceClassificationSpecialLayer()
        if self.taskType == TaskType.TOKEN_CLS:
            return self.TokenClassificationSpecialLayer()
        if self.taskType == TaskType.QUESTION_ANS:
            return self.QuestionAnsweringSpecialLayer()
        if self.taskType == TaskType.CAUSAL_LM:
            return self.CausalLMSpecialLayer()
        else:
            raise ValueError("......unsupported task type......")
    def group_model(self,model):
        group_parameters = []
        for name,p in model.named_parameters():
            if not p.requires_grad:continue
            for pattern in self.pattern_list:
                matches = re.compile(pattern).findall(name)
                if len(matches)>0:
                    if matches[0] not in group_parameters:
                        group_parameters.append(matches[0])
        if hasattr(self,"merge_param"):
            group_parameters = self.merge_param(group_parameters)
        if len(self.freeze_layers)>0:
            for index in self.freeze_layers:
                group_parameters[int(index)]=-1
        group_parameters = [element for element in group_parameters if element != -1]
        return group_parameters

class RobertaCallBack(ChunkCallBack):
    def __init__(self,freeze_layers,strategy,chunk_num,taskType,peft_type=None):
        super().__init__(freeze_layers,strategy,chunk_num,taskType,peft_type)
        self.TaskTInterface = [TaskType.SEQ_CLS,TaskType.TOKEN_CLS,TaskType.QUESTION_ANS]
        self.check_task_type(taskType,"RoBERTa",self.TaskTInterface)
    @property
    def emb_pattern(self):
        if self.peft_type:
            return [rf'\.embedding\.']
        else:
            return [rf'\.embeddings\.']
    @property
    def seq_cls_head(self):
        if self.peft_type:
            return ["classifier"]
        else:
            return ["classifier"]
    @property
    def token_cls_head(self):
        if self.peft_type:
            return ["classifier"]
        else:
            return ["classifier"]
    @property
    def qa_cls_head(self):
        if self.peft_type:
            return ["qa_outputs"]
        else:
            return ["qa_outputs"]
    @property
    def others_pattern(self):
        if self.peft_type:
            return [rf'\.\d+\.']
        else:
            return [rf'\.\d+\.']
class BERTCallBack(ChunkCallBack):
    def __init__(self,freeze_layers,strategy,chunk_num,taskType,peft_type=None):
        super().__init__(freeze_layers,strategy,chunk_num,taskType,peft_type)
        self.TaskTInterface = [TaskType.SEQ_CLS,TaskType.TOKEN_CLS,TaskType.QUESTION_ANS]
        self.check_task_type(taskType,"BERTa",self.TaskTInterface)
    @property
    def emb_pattern(self):
        if self.peft_type:
            return [rf'\.embedding\.']
        else:
            return [rf'\.embeddings\.']
    @property
    def seq_cls_head(self):
        if self.peft_type:
            return ["classifier"]
        else:
            return ["pooler","classifier"]
    @property
    def token_cls_head(self):
        if self.peft_type:
            return ["classifier"]
        else:
            return ["pooler","classifier"]
    @property
    def qa_cls_head(self):
        if self.peft_type:
            return ["qa_outputs"]
        else:
            return ["pooler","qa_outputs"]
    @property
    def others_pattern(self):
        if self.peft_type:
            return [rf'\.\d+\.']
        else:
            return [rf'\.\d+\.']
class GPT2CallBack(ChunkCallBack):
    def __init__(self,freeze_layers,strategy,chunk_num,taskType,peft_type=None):
        super().__init__(freeze_layers,strategy,chunk_num,taskType,peft_type)
        self.TaskTInterface = [TaskType.SEQ_CLS,TaskType.TOKEN_CLS,TaskType.QUESTION_ANS,TaskType.CAUSAL_LM]
        self.check_task_type(taskType,"GPT2",self.TaskTInterface)
    def merge_param(self,group_parameters):
        group_parameters = self.emb_pattern + [param for param in group_parameters if len(re.compile(self.emb_pattern[0]).findall(param))<=0]
        
        return group_parameters
    @property
    def emb_pattern(self):
        if self.peft_type:
            return [rf"\.embedding\."]
        else:
            return [rf"\.w[^ ]e\."]
    @property
    def seq_cls_head(self):
        if self.peft_type:
            return ["score"]
        else:
            return ["score"]
    @property
    def token_cls_head(self):
        if self.peft_type:
            return ["classifier"]
        else:
            return ["classifier"]
    @property
    def qa_cls_head(self):
        if self.peft_type:
            return ["qa_outputs"]
        else:
            return ["qa_outputs"]
    @property
    def causal_head(self):
        if self.peft_type:
            return [rf"\.ln_f\."]
        else:
            return [rf"\.ln_f\."]
    @property
    def others_pattern(self):
        if self.peft_type:
            return [rf'\.\d+\.']
        else:
            return [rf'\.\d+\.']
            
class GPTNeoXCallBack(ChunkCallBack):
    def __init__(self,freeze_layers,strategy,chunk_num,taskType,peft_type=None):
        super().__init__(freeze_layers,strategy,chunk_num,taskType,peft_type)
        self.TaskTInterface = [TaskType.SEQ_CLS,TaskType.TOKEN_CLS,TaskType.QUESTION_ANS,TaskType.CAUSAL_LM]
        self.check_task_type(taskType,"GPTNeoX",self.TaskTInterface)
    def merge_param(self,group_parameters):
        group_parameters = self.emb_pattern + [param for param in group_parameters if len(re.compile(self.emb_pattern[0]).findall(param))<=0]
        
        return group_parameters
    @property
    def emb_pattern(self):
        if self.peft_type:
            return [rf"\.embedding\."]
        else:
            return [rf"\.w[^ ]e\."]
    @property
    def seq_cls_head(self):
        if self.peft_type:
            return ["score"]
        else:
            return ["score"]
    @property
    def token_cls_head(self):
        if self.peft_type:
            return ["classifier"]
        else:
            return ["classifier"]
    @property
    def qa_cls_head(self):
        if self.peft_type:
            return ["qa_outputs"]
        else:
            return ["qa_outputs"]
    @property
    def causal_head(self):
        if self.peft_type:
            return [rf"\.ln_f\."]
        else:
            return [rf"\.ln_f\."]
    @property
    def others_pattern(self):
        if self.peft_type:
            return [rf'\.\d+\.']
        else:
            return [rf'\.\d+\.']

class OPTCallBack(ChunkCallBack):
    def __init__(self,freeze_layers,strategy,chunk_num,taskType,peft_type=None):
        super().__init__(freeze_layers,strategy,chunk_num,taskType,peft_type)
        self.TaskTInterface = [TaskType.SEQ_CLS,TaskType.QUESTION_ANS,TaskType.CAUSAL_LM]
        self.check_task_type(taskType,"OPT",self.TaskTInterface)
    
    def merge_param(self,group_parameters):
        group_parameters = self.emb_pattern + [param for param in group_parameters if len(re.compile(self.emb_pattern[0]).findall(param))<=0]
        return group_parameters
    @property
    def emb_pattern(self):
        if self.peft_type:
            return [rf"\.embedding\."]
        else:
            return [rf"\.embed_[^ ]+\."]
    @property
    def seq_cls_head(self):
        if self.peft_type:
            return ["score"]
        else:
            return ["score"]
    @property
    def qa_cls_head(self):
        if self.peft_type:
            return ["qa_outputs"]
        else:
            return ["qa_outputs"]
    @property
    def causal_head(self):
        if self.peft_type:
            return ["final_layer_norm"]
        else:
            return ["final_layer_norm"]
    @property
    def others_pattern(self):
        if self.peft_type:
            return [rf'\.\d+\.']
        else:
            return [rf'\.\d+\.']

class LLaMaFamilyCallBack(ChunkCallBack):
    def __init__(self,freeze_layers,strategy,chunk_num,taskType,peft_type=None):
        super().__init__(freeze_layers,strategy,chunk_num,taskType,peft_type)
        self.TaskTInterface = [TaskType.SEQ_CLS,TaskType.CAUSAL_LM]
        self.check_task_type(taskType,"LLaMA",self.TaskTInterface)
    @property
    def emb_pattern(self):
        if self.peft_type:
            return [rf"\.embedding\."]
        else:
            return ["embed_tokens"]
    @property
    def seq_cls_head(self):
        if self.peft_type:
            return ["score"]
        else:
            return ["model.norm.weight","score"]
    @property
    def causal_head(self):
        if self.peft_type:
            return ["lm_head"]
        else:
            return ["model.norm.weight","lm_head"]
    @property
    def others_pattern(self):
        if self.peft_type:
            return [rf'\.\d+\.']
        else:
            return [rf'\.\d+\.']

MODDELS_HiFT_PROCESS={
    "roberta":RobertaCallBack,
    "bert":BERTCallBack,
    "gpt2":GPT2CallBack,
    "gptneox":GPTNeoXCallBack,
    "gptneo":GPTNeoXCallBack,
    "opt":OPTCallBack,
    "llamafamily":LLaMaFamilyCallBack,
}

def GetCallBack(model_name_path):
    if "roberta" in model_name_path.lower():
        return MODDELS_HiFT_PROCESS["roberta"]
    if "bert" in model_name_path.lower():
        return MODDELS_HiFT_PROCESS["bert"]
    if "gpt2" in model_name_path.lower():
        return MODDELS_HiFT_PROCESS["gpt2"]
    if "gptneox" in model_name_path.lower() or "gpt-neox" in model_name_path.lower():
        return MODDELS_HiFT_PROCESS["gptneox"]
    if "gptneo" in model_name_path.lower() or "gpt-neo" in model_name_path.lower():
        return MODDELS_HiFT_PROCESS["gptneo"]
    if "opt" in model_name_path.lower():
        return MODDELS_HiFT_PROCESS["opt"]
    if "llamafamily" in model_name_path.lower() or "llama" in model_name_path.lower():
        return MODDELS_HiFT_PROCESS["llamafamily"]
    else:
        pass
    