import unittest
import numpy as np
import torch
from torch import nn
from src.two_stage_inference import attach_adapters, centered_features, infer_groups
from src.two_stage_lora import inferred_groups, centered


class InferenceTests(unittest.TestCase):
    def test_unmerged_adapter_matches_peft_arithmetic(self):
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(4)
        raw = nn.Sequential(nn.Linear(5, 7, bias=False))
        peft_base = nn.Sequential(nn.Linear(5, 7, bias=False))
        peft_base.load_state_dict(raw.state_dict())
        reference = get_peft_model(peft_base, LoraConfig(r=2, lora_alpha=4, target_modules=['0'], lora_dropout=0)).eval()
        a, b = torch.randn(2, 5), torch.randn(7, 2)
        with torch.no_grad():
            reference.base_model.model[0].lora_A.default.weight.copy_(a)
            reference.base_model.model[0].lora_B.default.weight.copy_(b)
        attach_adapters(raw, {'base_model.model.0.lora_A.weight': a, 'base_model.model.0.lora_B.weight': b}, {'r':2, 'lora_alpha':4})
        x = torch.randn(3, 5)
        torch.testing.assert_close(raw(x), reference(x), rtol=0, atol=0)

    def test_grouping_and_centering_match_training(self):
        texts=['quantum physics electron', 'quantum physics electron', 'tomato pasta cooking']
        np.testing.assert_array_equal(infer_groups(texts), inferred_groups(texts))
        z=np.arange(12,dtype=np.float32).reshape(3,4)
        a,b=centered(z, infer_groups(texts)); c,d=centered_features(z, infer_groups(texts))
        np.testing.assert_array_equal(a,c); np.testing.assert_array_equal(b,d)

    def test_empty_vocabulary_has_no_centered_groups(self):
        groups=infer_groups(['', 'the'])
        self.assertEqual(len(set(groups)),2)


if __name__=='__main__': unittest.main()
