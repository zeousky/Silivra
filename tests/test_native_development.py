import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from silivra.agent import build_system_prompt
from silivra.tools import ToolRegistry

SPEC = importlib.util.spec_from_file_location('native_development',Path(__file__).resolve().parents[1]/'scripts/build-native-development.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class NativeDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.registry = ToolRegistry(Path(self.temporary.name),allow_network=True)
        self.row = {'id':'v6-copy-abc','source':{'name':'AI-authored synthetic fixture','license':'MIT','url':'local:test','revision':'native-english-v6'},
            'messages':[{'role':'system','content':build_system_prompt([])},
                        {'role':'user','content':'Copy amber.'},
                        {'role':'assistant','content':'{"final":"amber"}'}]}

    def test_stable_sample_excludes_targets_and_imported_rehearsal(self):
        rows = [{**self.row,'id':'v6-copy-'+str(i)} for i in range(10)]
        other = copy.deepcopy(self.row);other['source']['revision']='old-source'
        first = builder.cases_from_records(rows+[other],self.registry,3)
        self.assertEqual(first,builder.cases_from_records(list(reversed(rows)),self.registry,3))
        self.assertEqual(len(first),3)
        for case in first:
            self.assertEqual(case['messages'],self.row['messages'][1:2])
            self.assertEqual(case['expected'],{'final':'amber'})

    def test_tool_history_is_preserved_as_inert_observation(self):
        spec = next(s for s in self.registry.schemas() if s['function']['name']=='calculate')
        row = copy.deepcopy(self.row);row['id']='v6-calculate-abc'
        row['messages']=[{'role':'system','content':build_system_prompt([spec])},
            {'role':'user','content':'Use calculate for 2+3.'},
            {'role':'assistant','content':'{"tool":"calculate","arguments":{"expression":"2+3"}}'},
            {'role':'tool','name':'calculate','content':'{"ok":true,"result":{"result":"5"}}'},
            {'role':'assistant','content':'{"final":"5"}'}]
        cases = builder.cases_from_records([row],self.registry)
        observed = next(c for c in cases if c['category']=='calculate-observation')
        self.assertEqual(observed['messages'],row['messages'][1:4])
        self.assertEqual(observed['expected'],{'final':'5'})

    def test_invalid_limit_or_changed_policy_is_rejected(self):
        for limit in (0,True,13):
            with self.assertRaises(ValueError):builder.cases_from_records([self.row],self.registry,limit)
        row=copy.deepcopy(self.row);row['messages'][0]['content']='changed policy'
        with self.assertRaises(ValueError):builder.cases_from_records([row],self.registry)


if __name__=='__main__':unittest.main()
