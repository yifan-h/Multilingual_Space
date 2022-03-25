import os
import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Conv1D
import transformers.adapters.composition as ac


class MLKGLM(nn.Module):
    """docstring for ClassName"""
    def __init__(self, bert_path):
        super(MLKGLM, self).__init__()
        # load pretrained MLLM
        self.MLLM = AutoModel.from_pretrained(bert_path, 
                                            return_dict=True,
                                            output_hidden_states=True)
        hidden_num = self.MLLM.get_input_embeddings().embedding_dim
        # set three extra modules
        self.universal_mapping = nn.Sequential(Conv1D(hidden_num, hidden_num),
                                                    nn.ELU(),
                                                    nn.LayerNorm(hidden_num, eps=1e-12),
                                                    nn.Dropout(0.1),
                                                    Conv1D(4*hidden_num, hidden_num),
                                                    nn.ELU(),
                                                    nn.LayerNorm(4*hidden_num, eps=1e-12),
                                                    nn.Dropout(0.1),
                                                    Conv1D(hidden_num, 4*hidden_num),
                                                    nn.ELU(),
                                                    nn.LayerNorm(hidden_num, eps=1e-12),
                                                    nn.Dropout(0.1),
                                                    Conv1D(hidden_num, hidden_num),
                                                    nn.ELU(),
                                                    nn.LayerNorm(hidden_num, eps=1e-12),
                                                    nn.Dropout(0.1))
        self.universal_aggregator = nn.Sequential(Conv1D(hidden_num, 2*hidden_num),
                                                    nn.LayerNorm(hidden_num, eps=1e-12),
                                                    nn.Dropout(0.1))
        self.triple_mapping = nn.Sequential(Conv1D(hidden_num, hidden_num),
                                                nn.ELU(),
                                                nn.LayerNorm(hidden_num, eps=1e-12),
                                                nn.Dropout(0.1),
                                                Conv1D(4*hidden_num, hidden_num),
                                                nn.ELU(),
                                                nn.LayerNorm(4*hidden_num, eps=1e-12),
                                                nn.Dropout(0.1),
                                                Conv1D(hidden_num, 4*hidden_num),
                                                nn.ELU(),
                                                nn.LayerNorm(hidden_num, eps=1e-12),
                                                nn.Dropout(0.1),
                                                Conv1D(hidden_num, hidden_num),
                                                nn.ELU(),
                                                nn.LayerNorm(hidden_num, eps=1e-12),
                                                nn.Dropout(0.1))
        self.triple_aggregator = nn.Sequential(Conv1D(hidden_num, 2*hidden_num),
                                                nn.LayerNorm(hidden_num, eps=1e-12),
                                                nn.Dropout(0.1))
        # for testing
        self.all_aggregator = nn.Linear(3*hidden_num, hidden_num, bias=False)
        # self.all_aggregator.weight.data = self.weight_init_sum(self.all_aggregator.weight.data)

    def weight_init_sum(self, t):
        hidden_num = int(t.shape[-1]/3)
        return 0.0003 + torch.cat((0.333*torch.eye(hidden_num,hidden_num),
                                    0.333*torch.eye(hidden_num,hidden_num),
                                    0.333*torch.eye(hidden_num,hidden_num)),dim=1)

    def forward(self, **inputs):
        # get MLLM output
        outputs_MLLM = self.MLLM(**inputs).hidden_states
        # take last layer hidden state: (batch_size, sequence_length, hidden_size)
        outputs_MLLM = outputs_MLLM[-1]
        # objective 1: universal space
        outputs_universal = self.universal_mapping(outputs_MLLM)
        outputs_universal = self.universal_aggregator(torch.cat((outputs_MLLM, outputs_universal), dim=-1))
        # objective 2: transformer layers
        outputs_MLKGLM = self.triple_mapping(outputs_universal)
        outputs_MLKGLM = self.triple_aggregator(torch.cat((outputs_MLLM, outputs_MLKGLM), dim=-1))
        return self.all_aggregator(torch.cat((outputs_MLLM, outputs_universal, outputs_MLKGLM), dim=-1))

class LMQA(nn.Module):
    """docstring for ClassName"""
    def __init__(self, args):
        super(LMQA, self).__init__()
        # load pretrained MLLM
        self.model_name = args.model_name
        if args.model_name[-2:] == "KG":
            self.base_model = MLKGLM(args)
            load_model(self.base_model, args.modelkg_dir)
        else:
            self.base_model = AutoModel.from_pretrained(args.model_dir, 
                                                        return_dict=True,
                                                        output_hidden_states=True)
        self.new_all_aggregator = nn.Linear(768, 128)

    def forward(self, **inputs):
        if self.model_name[-2:] == "KG":
            outputs = self.base_model(**inputs)
        else:
            outputs = self.base_model(**inputs).hidden_states[-1]
        outputs = torch.mean(outputs, dim=1)
        outputs = F.elu(self.new_all_aggregator(outputs))
        return outputs



class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )