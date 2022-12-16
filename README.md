This module containing algorithms and programs that is using to learn Z3 tactics.
Learning algorithms are implemented in python language, so this module is in python language.

To start learning run main.py file. If you want to learning start from beginning you must 
remove *.h5 files from 'learning/shared' directory. If you want to actions will be generated again, you must remove 
'randomly_selected_actions.json' file too.
Also, 'learning/shared' directory contain .txt files that keep paths of learned smt2 files to 
prevent duplicate learning in each run,
and if you want all smt2 files be used in learning you can remove this files too (if exists).

To solve inequalities with learned policy run 'renderer.py' file and pass inequalities json to it.

To test learned policy you can run 'test_learning.py' file. 'test_learning.py' file test learned policy
on test data.

Note: Don't remove randomly_selected_actions.json file in shared 
folder individually, because policy has been prepared based on 
actions in randomly_selected_actions.json, and changing actions during learning or rendering 
learned policy can cause problem. You must remove all policy files and randomly_selected_actions.json file
together or none of them.

