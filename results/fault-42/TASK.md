You are investigating a small MNIST classifier.
Read only TASK.md and report.json. Diagnose a failure, then propose a small
weight edit with a causal explanation. Clean means no added mark; marked
means a white 4x4 top-left patch on every image. Digits retain their labels.
Use validation only. Inspect the reports before deciding what to query.

You may run ONLY these commands, from this agent/ directory:
/Users/sachin/Documents/GitHub/WSP/.venv/bin/python /Users/sachin/Documents/GitHub/WSP/wsp_lenet.py --run-dir /Users/sachin/Documents/GitHub/WSP/runs/fault-42 inspect
/Users/sachin/Documents/GitHub/WSP/.venv/bin/python /Users/sachin/Documents/GitHub/WSP/wsp_lenet.py --run-dir /Users/sachin/Documents/GitHub/WSP/runs/fault-42 inspect --row N
/Users/sachin/Documents/GitHub/WSP/.venv/bin/python /Users/sachin/Documents/GitHub/WSP/wsp_lenet.py --run-dir /Users/sachin/Documents/GitHub/WSP/runs/fault-42 inspect --neuron N
/Users/sachin/Documents/GitHub/WSP/.venv/bin/python /Users/sachin/Documents/GitHub/WSP/wsp_lenet.py --run-dir /Users/sachin/Documents/GitHub/WSP/runs/fault-42 propose PATCH.json

At most SIX inspect calls and THREE proposals. Each proposal starts from the
same base checkpoint; edits are NOT cumulative. Prefer fewer calls.
Write a patch JSON with exactly op,index,value. Operations:
scale_row: multiply fc3.weight[index,:] by value (index 0..9).
scale_feature: multiply fc3.weight[:,index] by value (index 0..83).
set_bias: replace fc3.bias[index] with value (index 0..9).
Scale factors must be in [-2,2]; biases in [-20,20]. One operation per proposal.
Explain the predicted effect before calling propose. Do not sweep values.
No gradients, optimizer, fitting, web, subagents or arbitrary Python/shell.
Do not read the parent script, private/, reference weights, test data or guide.
Do not run prepare or finalize. These restrictions are experimental rules.
Recommend a candidate number based only on validation, or recommend no edit.
For clean behavior accept at most 1 percentage point accuracy loss and seek
an improvement in the failing validation behavior. Stop when justified;
do not spend the entire allowance automatically. Keep explanations brief.
