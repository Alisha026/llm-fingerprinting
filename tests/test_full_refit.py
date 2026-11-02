import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.two_stage_lora import validate_full_refit


class FullRefitTests(unittest.TestCase):
    def test_exact_union_and_public_checkpoint_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); checkpoint=root/'public'; encoder=root/'base'
            encoder.mkdir(); checkpoint.mkdir(); (checkpoint/'adapter').mkdir(); (checkpoint/'tokenizer').mkdir()
            provenance={'repo_id':'Qwen/Qwen3-8B-Base','revision':'test'}
            (encoder/'download_provenance.json').write_text(json.dumps(provenance))
            meta={'labels':['a','b'],'cfg':{'encoder_provenance':provenance,'keep':24,'layers':[2,4,6,8,12,16,20,24]}}
            (checkpoint/'meta.json').write_text(json.dumps(meta))
            for name in ['heads.safetensors','adapter/adapter_config.json','adapter/adapter_model.safetensors','tokenizer/tokenizer.json']:
                (checkpoint/name).write_text('test fixture')
            frame=pd.DataFrame({'id':['1','2'],'prompt_id':['x','y'],'text':['one','two'],'model':['a','b']})
            actual=validate_full_refit(frame,frame.iloc[:1],frame.iloc[1:],checkpoint,encoder)
            pd.testing.assert_frame_equal(actual,frame)
            changed=frame.copy(); changed.loc[0,'text']='altered'
            with self.assertRaisesRegex(ValueError,'exactly equal'):
                validate_full_refit(changed,frame.iloc[:1],frame.iloc[1:],checkpoint,encoder)
            with self.assertRaisesRegex(ValueError,'public stage'):
                validate_full_refit(frame,frame.iloc[:1],frame.iloc[1:],root/'official',encoder)
            meta['cfg']['encoder_provenance']['revision']='wrong'
            (checkpoint/'meta.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'revision'):
                validate_full_refit(frame,frame.iloc[:1],frame.iloc[1:],checkpoint,encoder)


if __name__=='__main__': unittest.main()
