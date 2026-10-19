import json, sys, glob, os
K = ("hidden_size","intermediate_size","moe_intermediate_size","shared_expert_intermediate_size","n_shared_experts","num_shared_experts",
     "first_k_dense_replace","n_routed_experts","num_experts","num_local_experts","hidden_act","hidden_activation","activation",
     "mlp_hidden_act","intermediate_size_mlp","interleave_moe_layer_step","model_type","architectures","num_hidden_layers",
     "moe_layer_freq","mlp_layer_types","decoder_sparse_step","expert_intermediate_size","ffn_hidden_size","moe_ffn_hidden_size",
     "layers_block_type","hybrid_override_pattern","mlp_only_layers","num_experts_per_tok","glu","swiglu_limit","dense_intermediate_size")
def walk(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict): walk(v, pre + k + ".")
        elif k in K or "intermediate" in k or "act" in k.lower() and "fact" not in k:
            s = json.dumps(v); print(f"   {pre}{k} = {s[:140]}")
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    print("==", os.path.basename(f))
    try: walk(json.load(open(f)))
    except Exception as e: print("  ERR", e)
